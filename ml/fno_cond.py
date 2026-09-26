"""
Conditional FNO: takes eta/s as a scalar input alongside eps(x,y).
Uses FiLM (Feature-wise Linear Modulation) for viscosity conditioning.

Architecture:
  Input:  eps(x,y) [B, 1, 61, 61]  +  eta/s [B, 1]
  Output: T, ux, uy, rho [B, 4, 61, 61]

FiLM conditioning:
  eta/s -> MLP -> (gamma, beta) per FNO block
  x = gamma * x + beta  (modulates feature maps)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.in_ch = in_ch; self.out_ch = out_ch
        self.modes1 = modes1; self.modes2 = modes2
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_ch, H, W//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:,:,:self.modes1,:self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:,:,:self.modes1,:self.modes2], self.w1)
        out_ft[:,:,-self.modes1:,:self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:,:,-self.modes1:,:self.modes2], self.w2)
        return torch.fft.irfft2(out_ft, s=(H, W))


class FiLM(nn.Module):
    """FiLM conditioning: gamma, beta = MLP(eta_s)."""
    def __init__(self, cond_dim, feature_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.GELU(),
            nn.Linear(64, feature_dim * 2)  # gamma + beta
        )
        # small random init so gradients flow from start
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, cond):
        """
        x:    [B, C, H, W]
        cond: [B, cond_dim]
        """
        params = self.mlp(cond)           # [B, 2C]
        gamma, beta = params.chunk(2, dim=1)
        gamma = 1 + gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class FNOBlockCond(nn.Module):
    """FNO block with FiLM conditioning."""
    def __init__(self, d_model, modes1, modes2, cond_dim):
        super().__init__()
        self.spectral = SpectralConv2d(d_model, d_model, modes1, modes2)
        self.local    = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.norm     = nn.InstanceNorm2d(d_model)
        self.film     = FiLM(cond_dim, d_model)

    def forward(self, x, cond):
        x = F.gelu(self.norm(self.spectral(x) + self.local(x)))
        x = self.film(x, cond)  # modulate with eta/s
        return x


class HydroFNOCond(nn.Module):
    """
    Conditional FNO for viscous QGP hydrodynamics.
    Conditions on eta/s for continuous viscosity interpolation.
    """
    def __init__(self,
        in_channels  = 1,
        out_channels = 4,
        d_model      = 32,
        n_blocks     = 4,
        modes1       = 16,
        modes2       = 16,
        cond_dim     = 1,   # eta/s scalar
    ):
        super().__init__()
        self.cond_dim = cond_dim

        # lift input to d_model channels
        self.lift = nn.Conv2d(in_channels, d_model, kernel_size=1)

        # conditional FNO blocks
        self.blocks = nn.ModuleList([
            FNOBlockCond(d_model, modes1, modes2, cond_dim)
            for _ in range(n_blocks)
        ])

        # project to output
        self.project = nn.Sequential(
            nn.Conv2d(d_model, d_model//2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_model//2, out_channels, kernel_size=1),
        )

        self.eps_activation = nn.Softplus()

    def forward(self, eps_field, cond):
        """
        eps_field: [B, 1, 61, 61]
        cond:      [B, cond_dim]  (eta/s values)
        """
        x = self.lift(eps_field)
        for block in self.blocks:
            x = block(x, cond)
        out = self.project(x)
        out = torch.cat([self.eps_activation(out[:,0:1]), out[:,1:]], dim=1)
        return out


if __name__ == "__main__":
    # test
    model = HydroFNOCond(in_channels=1, out_channels=4,
                          d_model=32, n_blocks=4,
                          modes1=16, modes2=16, cond_dim=1)
    n = sum(p.numel() for p in model.parameters())
    print(f"Conditional FNO: {n/1e6:.2f}M parameters")

    # test forward pass
    B = 4
    eps = torch.randn(B, 1, 61, 61)
    etas = torch.tensor([[0.05],[0.08],[0.12],[0.20]])
    out = model(eps, etas)
    print(f"Input:  {eps.shape}")
    print(f"eta/s:  {etas.shape}")
    print(f"Output: {out.shape}")
    print("Forward pass: OK")

    # test that different eta/s gives different output
    eps_same = torch.randn(1, 1, 61, 61).expand(4,-1,-1,-1)
    out_diff = model(eps_same, etas)
    diffs = [(out_diff[i]-out_diff[0]).abs().mean().item() for i in range(4)]
    print(f"\nOutput difference for same IC, different eta/s:")
    for i,eta in enumerate([0.05,0.08,0.12,0.20]):
        print(f"  eta/s={eta}: mean diff from eta/s=0.05: {diffs[i]:.6f}")
    print("Conditioning works!" if diffs[3]>1e-6 else "WARNING: conditioning not working")
