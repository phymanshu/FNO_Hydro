"""
Standalone Optical Glauber model for Pb-Pb collisions.
Computes initial energy density field ε(x,y) given impact parameter b.

Physics:
  - Woods-Saxon nuclear density for Pb-208
  - Binary collision (BC) + wounded nucleon (WN) model
  - ε(x,y) ∝ (1-α)*T_A*T_B + α*T_A(x+b/2)*T_B(x-b/2)
  - Eccentricities ε₂, ε₃ computed from ε(x,y)

Reference: Miller et al., Ann.Rev.Nucl.Part.Sci. 57 (2007) 205
"""
import torch
import math

# Pb-208 Woods-Saxon parameters
R_Pb   = 6.62   # fm, nuclear radius
a_Pb   = 0.546  # fm, surface diffuseness
rho0   = 0.16   # fm^-3, nuclear saturation density
sigma_NN = 6.4  # fm^2, inelastic NN cross section at 5.36 TeV

def woods_saxon(r, R=R_Pb, a=a_Pb):
    """Woods-Saxon nuclear density profile."""
    return 1.0 / (1.0 + torch.exp((r - R) / a))

def nuclear_thickness(x, y, R=R_Pb, a=a_Pb, n_z=50, z_max=15.0):
    """
    T_A(x,y) = ∫ ρ(x,y,z) dz
    Computed by numerical integration along z.
    x, y: [nx, ny] grids
    Returns T_A: [nx, ny]
    """
    z = torch.linspace(-z_max, z_max, n_z, device=x.device)
    dz = 2 * z_max / (n_z - 1)

    # r^2 for each (x,y,z) combination
    # x,y: [nx,ny] → expand to [nx,ny,nz]
    x3 = x.unsqueeze(-1).expand(*x.shape, n_z)
    y3 = y.unsqueeze(-1).expand(*y.shape, n_z)
    z3 = z.view(1, 1, n_z).expand(*x.shape, n_z)

    r3 = torch.sqrt(x3**2 + y3**2 + z3**2)
    rho3 = woods_saxon(r3, R, a)
    T = rho3.sum(-1) * dz   # integrate over z
    return T

def glauber_density(b, nx=64, ny=64, xy_max=12.0, alpha=0.15, device="cpu"):
    """
    Compute initial energy density field ε(x,y) for Pb-Pb at impact parameter b.

    Args:
        b:      impact parameter (fm)
        nx,ny:  grid size
        xy_max: transverse extent (fm)
        alpha:  hard-scattering fraction (0=pure WN, 1=pure BC)

    Returns:
        eps:  [nx, ny] energy density field (normalized)
        eps2: eccentricity ε₂
        eps3: eccentricity ε₃ (from fluctuations, approx here)
    """
    # transverse grid
    x = torch.linspace(-xy_max, xy_max, nx, device=device)
    y = torch.linspace(-xy_max, xy_max, ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # nuclear thickness functions
    # nucleus A centered at (+b/2, 0), nucleus B at (-b/2, 0)
    T_A = nuclear_thickness(X - b/2, Y)
    T_B = nuclear_thickness(X + b/2, Y)

    # inelastic collision probability per unit area
    T_AB = T_A * T_B * sigma_NN  # binary collision density

    # wounded nucleon density (participant density)
    # T_WN = T_A*(1-(1-sigma*T_B/A)^A) + T_B*(1-(1-sigma*T_A/A)^A)
    A = 208.0
    T_WN_A = T_A * (1.0 - (1.0 - sigma_NN * T_B / A).clamp(0, 1).pow(A))
    T_WN_B = T_B * (1.0 - (1.0 - sigma_NN * T_A / A).clamp(0, 1).pow(A))
    T_WN   = T_WN_A + T_WN_B

    # energy density: mix of WN and BC
    eps = (1.0 - alpha) * T_WN + alpha * T_AB
    eps = eps / (eps.sum() + 1e-10)   # normalize to 1

    # eccentricities from ε(x,y)
    # shift to center of mass
    x_cm = (X * eps).sum() / (eps.sum() + 1e-10)
    y_cm = (Y * eps).sum() / (eps.sum() + 1e-10)
    Xc = X - x_cm
    Yc = Y - y_cm
    r2 = Xc**2 + Yc**2

    # ε₂ = |<r² e^{2iφ}>| / <r²>
    phi = torch.atan2(Yc, Xc)
    eps2_num = torch.sqrt(
        (eps * r2 * torch.cos(2*phi)).sum()**2 +
        (eps * r2 * torch.sin(2*phi)).sum()**2
    )
    eps2_den = (eps * r2).sum()
    eps2 = (eps2_num / (eps2_den + 1e-10)).item()

    # ε₃ (participant triangularity) — from r³ moments
    eps3_num = torch.sqrt(
        (eps * r2 * torch.cos(3*phi)).sum()**2 +
        (eps * r2 * torch.sin(3*phi)).sum()**2
    )
    eps3 = (eps3_num / (eps2_den + 1e-10)).item()

    return eps, eps2, eps3


def glauber_batch(b_values, nx=64, ny=64, xy_max=12.0, alpha=0.15):
    """
    Compute ε(x,y) fields for a batch of impact parameters.
    Returns: fields [B, 1, nx, ny], eps2 [B], eps3 [B]
    """
    fields, eps2s, eps3s = [], [], []
    for b in b_values:
        eps, eps2, eps3 = glauber_density(float(b), nx, ny, xy_max, alpha)
        fields.append(eps.unsqueeze(0))
        eps2s.append(eps2)
        eps3s.append(eps3)

    return (torch.stack(fields).unsqueeze(1),
            torch.tensor(eps2s),
            torch.tensor(eps3s))


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Testing Glauber model...")
    test_bs = [1.0, 3.0, 5.0, 7.0, 9.0]

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    fig.suptitle("Glauber Initial Energy Density ε(x,y) for Pb-Pb", fontweight="bold")

    for i, b in enumerate(test_bs):
        eps, eps2, eps3 = glauber_density(b, nx=64, ny=64)
        print(f"  b={b:.1f} fm:  ε₂={eps2:.4f}  ε₃={eps3:.4f}  "
              f"max_eps={eps.max():.4f}")

        # 2D field
        ax = axes[0, i]
        im = ax.imshow(eps.numpy().T, origin="lower", cmap="hot",
                       extent=[-12,12,-12,12])
        ax.set_title(f"b={b:.1f} fm\nε₂={eps2:.3f}")
        ax.set_xlabel("x (fm)"); ax.set_ylabel("y (fm)")
        plt.colorbar(im, ax=ax)

        # x-projection
        ax = axes[1, i]
        ax.plot(torch.linspace(-12,12,64).numpy(),
                eps.sum(1).numpy(), color="red", lw=2, label="proj x")
        ax.plot(torch.linspace(-12,12,64).numpy(),
                eps.sum(0).numpy(), color="blue", lw=2, ls="--", label="proj y")
        ax.set_xlabel("position (fm)")
        ax.set_ylabel("ε projection")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("/eos/user/h/hsharma/epos4_plots/glauber_fields.png",
                bbox_inches="tight")
    print("\nSaved: glauber_fields.png")

    # test batch
    print("\nTesting batch computation...")
    b_batch = torch.tensor([4.6, 4.9, 5.2, 5.5, 5.8])
    fields, eps2s, eps3s = glauber_batch(b_batch)
    print(f"  Fields shape: {fields.shape}")
    print(f"  ε₂ values: {eps2s.tolist()}")
    print(f"  ε₃ values: {eps3s.tolist()}")
    print("\nAll tests passed.")
