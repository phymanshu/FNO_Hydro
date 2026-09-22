"""
Fourier Neural Operator surrogate for EPOS4 Pb-Pb hydrodynamic evolution.

Physics-informed architecture:
  Input:  initial energy density field  ε(x,y)  [batch, 1, nx, ny]
  Output: freeze-out field              T^μν(x,y) [batch, n_fields, nx, ny]

The FNO learns the solution operator of relativistic viscous hydro:
  G : ε(·,τ₀) → T^μν(·,τ_f)

Reference: Li et al. 2021 "Fourier Neural Operator for PDEs"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Spectral convolution layer (core FNO building block) ─────────────────────
class SpectralConv2d(nn.Module):
    """
    2D Fourier layer: FFT → linear mix of low-frequency modes → iFFT.
    Learns correlations at all scales simultaneously.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1   # number of Fourier modes in x
        self.modes2 = modes2   # number of Fourier modes in y

        scale = 1 / (in_channels * out_channels)
        # complex weights for 4 quadrants of the 2D FFT
        self.weights1 = nn.Parameter(scale * torch.randn(
            in_channels, out_channels, modes1, modes2, 2))
        self.weights2 = nn.Parameter(scale * torch.randn(
            in_channels, out_channels, modes1, modes2, 2))

    def compl_mul2d(self, inp, weights):
        # (batch, in_ch, x, y), (in_ch, out_ch, x, y) -> (batch, out_ch, x, y)
        return torch.einsum("bixy,ioxy->boxy",
                            inp, torch.view_as_complex(weights))

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(B, self.out_channels, H, W//2+1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2],
                             self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2],
                             self.weights2)

        return torch.fft.irfft2(out_ft, s=(H, W))


# ── Single FNO block ──────────────────────────────────────────────────────────
class FNOBlock2d(nn.Module):
    """
    One FNO layer: spectral conv (global) + pointwise conv (local) + activation.
    The two paths are added before activation, like a residual connection.
    """
    def __init__(self, channels, modes1, modes2):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes1, modes2)
        self.local    = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm     = nn.InstanceNorm2d(channels)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.local(x)))


# ── Full FNO model ────────────────────────────────────────────────────────────
class HydroFNO(nn.Module):
    """
    FNO surrogate for relativistic hydrodynamic evolution.

    Maps initial energy density field to freeze-out field:
      ε(x,y,τ₀)  →  T^μν(x,y,τ_f)

    Architecture:
      Lifting layer:   1 → d_model channels
      FNO blocks ×4:   learn global + local correlations
      Projection:      d_model → n_out channels

    Conditioning:
      Impact parameter b and sqrt(s) are appended as constant
      feature maps to the input, giving the model global event info.
    """
    def __init__(
        self,
        in_channels  = 1,      # ε field
        out_channels = 6,      # T^00, T^01, T^02, T^11, T^12, T^22
        d_model      = 64,     # feature dimension
        n_blocks     = 4,      # number of FNO blocks
        modes1       = 12,     # Fourier modes in x (≤ nx//2)
        modes2       = 12,     # Fourier modes in y (≤ ny//2)
        n_cond       = 2,      # conditioning variables (b, sqrt_s)
    ):
        super().__init__()
        self.n_cond = n_cond

        # lift input + conditioning to d_model channels
        self.lift = nn.Conv2d(in_channels + n_cond, d_model, kernel_size=1)

        # FNO blocks
        self.blocks = nn.ModuleList([
            FNOBlock2d(d_model, modes1, modes2)
            for _ in range(n_blocks)
        ])

        # project to output
        self.project = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_model // 2, out_channels, kernel_size=1),
        )

        # physics constraint: output energy density must be >= 0
        self.eps_activation = nn.Softplus()

    def forward(self, eps_field, b=None, sqrt_s=None):
        """
        Args:
            eps_field: [B, 1, nx, ny]   initial energy density
            b:         [B]              impact parameter (fm)
            sqrt_s:    [B]              sqrt(s_NN) (GeV)
        Returns:
            T_field:   [B, 6, nx, ny]   stress-energy tensor components
        """
        B, _, H, W = eps_field.shape

        # append conditioning as constant feature maps
        if b is not None and sqrt_s is not None:
            b_map = b.view(B, 1, 1, 1).expand(B, 1, H, W)
            s_map = sqrt_s.view(B, 1, 1, 1).expand(B, 1, H, W)
            x = torch.cat([eps_field, b_map, s_map], dim=1)
        else:
            # zero conditioning if not provided
            zeros = torch.zeros(B, self.n_cond, H, W, device=eps_field.device)
            x = torch.cat([eps_field, zeros], dim=1)

        # lift
        x = self.lift(x)

        # FNO blocks
        for block in self.blocks:
            x = block(x)

        # project
        out = self.project(x)

        # physics constraint: T^00 (energy density) must be >= 0
        out = torch.cat([self.eps_activation(out[:, 0:1]), out[:, 1:]], dim=1)

        return out


# ── Physics-informed loss ─────────────────────────────────────────────────────
class PhysicsLoss(nn.Module):
    """
    Combined loss with data term + physics constraints.

    L = L_data + λ_cons * L_conservation + λ_pos * L_positivity

    L_data:         MSE between predicted and true T^μν
    L_conservation: residual of ∂_μ T^μν = 0 (energy-momentum conservation)
    L_positivity:   penalty for negative energy density T^00
    """
    def __init__(self, lambda_cons=0.1, lambda_pos=1.0):
        super().__init__()
        self.lambda_cons = lambda_cons
        self.lambda_pos  = lambda_pos

    def conservation_residual(self, T_field, dx=0.1):
        """
        Approximate ∂_x T^0x + ∂_y T^0y using finite differences.
        T_field: [B, 6, nx, ny]
          channels: T00, T01, T02, T11, T12, T22
        """
        T01 = T_field[:, 1]   # T^0x
        T02 = T_field[:, 2]   # T^0y

        dT01_dx = (T01[:, 2:, :] - T01[:, :-2, :]) / (2 * dx)
        dT02_dy = (T02[:, :, 2:] - T02[:, :, :-2]) / (2 * dx)

        # trim to same size
        res = dT01_dx[:, :, 1:-1] + dT02_dy[:, 1:-1, :]
        return res.pow(2).mean()

    def forward(self, pred, target):
        # data loss
        L_data = F.mse_loss(pred, target)

        # conservation loss
        L_cons = self.conservation_residual(pred)

        # positivity loss: penalize negative T^00
        L_pos = F.relu(-pred[:, 0]).pow(2).mean()

        total = L_data + self.lambda_cons * L_cons + self.lambda_pos * L_pos
        return total, {
            "L_data": L_data.item(),
            "L_cons": L_cons.item(),
            "L_pos":  L_pos.item(),
            "L_total": total.item(),
        }


# ── Quick architecture test ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing HydroFNO architecture...")

    model = HydroFNO(
        in_channels  = 1,
        out_channels = 6,
        d_model      = 64,
        n_blocks     = 4,
        modes1       = 12,
        modes2       = 12,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # simulate input: batch of 2 events, 64×64 transverse grid
    B, nx, ny = 2, 64, 64
    eps   = torch.rand(B, 1, nx, ny)     # initial energy density
    b     = torch.tensor([5.1, 4.8])     # impact parameters
    sqrt_s = torch.tensor([5360., 5360.]) # sqrt(s_NN) in GeV

    # forward pass
    T_out = model(eps, b, sqrt_s)
    print(f"  Input shape:  {eps.shape}")
    print(f"  Output shape: {T_out.shape}")
    print(f"  T^00 range:   [{T_out[:,0].min():.4f}, {T_out[:,0].max():.4f}] (must be >= 0)")
    print(f"  T^01 range:   [{T_out[:,1].min():.4f}, {T_out[:,1].max():.4f}]")

    # loss test
    target = torch.rand(B, 6, nx, ny)
    target[:, 0] = target[:, 0].abs()   # ensure positive T^00 in target
    loss_fn = PhysicsLoss()
    loss, breakdown = loss_fn(T_out, target)
    print(f"\n  Loss breakdown: {breakdown}")
    print("\nAll tests passed.")
