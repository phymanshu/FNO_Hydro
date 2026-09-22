"""
Simplified Cooper-Frye particle sampler for FNO freeze-out field.

Input:  FNO output field [4, 61, 61]
        channels: T(x,y), ux(x,y), uy(x,y), cell_density(x,y)
Output: particle list [(px, py, pz, pid), ...]

Physics:
  For each active cell (T > T_min, density > 0):
    - Sample N particles from thermal distribution at T, boosted by (ux, uy)
    - Use Boltzmann approximation: dN/d³p ∝ exp(-p·u/T)
    - Sample species fractions from ALICE measurements
"""
import torch
import numpy as np
import math

# Particle masses in GeV
MASSES = {
    211:  0.13957,   # π+
   -211:  0.13957,   # π-
    321:  0.49368,   # K+
   -321:  0.49368,   # K-
   2212:  0.93827,   # p
  -2212:  0.93827,   # p-bar
}

# Species fractions from ALICE Pb-Pb 30-50% (approximate)
SPECIES = list(MASSES.keys())
FRACTIONS = np.array([0.35, 0.35, 0.08, 0.08, 0.07, 0.07])
FRACTIONS /= FRACTIONS.sum()

T_MIN   = 0.100   # GeV - minimum freeze-out T
T_FO    = 0.155   # GeV - nominal freeze-out T
XY_MAX  = 12.0    # fm  - grid extent
NX = NY = 61

def sample_thermal_momentum(T, ux, uy, mass, n_particles):
    """
    Sample momenta from boosted thermal distribution.
    Uses Boltzmann approximation with flow boost.
    Returns: [n_particles, 3] tensor of (px, py, pz)
    """
    # sample pT from Boltzmann: dN/dpT ∝ pT * exp(-mT/T)
    # use inverse transform: pT = -T*log(u1*u2) approximately
    # More accurate: sample from modified Bessel function K2
    n = n_particles
    
    # sample pT via accept-reject from Boltzmann
    pT_max = 5.0  # GeV cutoff
    pT_list = []
    while len(pT_list) < n:
        pT_try = torch.rand(n*3) * pT_max
        mT_try = torch.sqrt(pT_try**2 + mass**2)
        # Boltzmann weight with flow boost (simplified: no boost for sampling)
        weight = pT_try * torch.exp(-mT_try / T)
        weight /= weight.max()
        accept = torch.rand(len(weight)) < weight
        pT_list.extend(pT_try[accept].tolist())
    pT = torch.tensor(pT_list[:n])
    
    # sample phi uniformly
    phi = torch.rand(n) * 2 * math.pi
    px_cm = pT * torch.cos(phi)
    py_cm = pT * torch.sin(phi)
    
    # sample rapidity from Gaussian (approximate)
    eta = torch.randn(n) * 2.0  # rapidity width ~2
    mT = torch.sqrt(pT**2 + mass**2)
    pz = mT * torch.sinh(eta)
    
    # apply transverse flow boost
    gamma = 1.0 / math.sqrt(max(1 - ux**2 - uy**2, 0.01))
    # simplified boost
    px = px_cm + gamma * ux * mT
    py = py_cm + gamma * uy * mT
    
    return torch.stack([px, py, pz], dim=1)


def sample_event(freeze_field, n_target=None, seed=None):
    """
    Sample a full event from FNO freeze-out field.
    
    Args:
        freeze_field: tensor [4, 61, 61] — (T, ux, uy, cell_density)
        n_target:     target multiplicity (if None, estimated from field)
        seed:         random seed
    
    Returns:
        particles: list of (px, py, pz, pid) tuples
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    T_field   = freeze_field[0]   # [61,61]
    ux_field  = freeze_field[1]   # [61,61]
    uy_field  = freeze_field[2]   # [61,61]
    dens      = freeze_field[3]   # [61,61]
    
    # active cells: T above minimum and density > 0
    active = (T_field > T_MIN) & (dens > 0)
    if active.sum() == 0:
        return []
    
    # estimate total multiplicity from cell density
    total_density = dens[active].sum().item()
    if n_target is None:
        n_target = max(100, int(total_density * 500))
    
    # distribute particles across cells proportional to density
    cell_weights = dens[active] / (dens[active].sum() + 1e-10)
    n_per_cell = torch.multinomial(cell_weights,
                                   n_target,
                                   replacement=True)
    
    particles = []
    active_idx = active.nonzero()
    
    for k, (ix, iy) in enumerate(active_idx):
        n_k = (n_per_cell == k).sum().item()
        if n_k == 0:
            continue
        
        T  = T_field[ix, iy].item()
        ux = ux_field[ix, iy].item()
        uy = uy_field[ix, iy].item()
        
        if T < T_MIN:
            continue
        
        # sample species
        pids = np.random.choice(SPECIES, size=n_k, p=FRACTIONS)
        
        for pid in pids:
            mass = MASSES[pid]
            mom = sample_thermal_momentum(T, ux, uy, mass, 1)
            particles.append((
                mom[0,0].item(),
                mom[0,1].item(),
                mom[0,2].item(),
                pid
            ))
    
    return particles


def compute_observables(particles, eta_max=0.8, pt_min=0.2, pt_max=5.0):
    """
    Compute bulk observables from particle list.
    Returns: dict with Nch, mean_pT, v2, v3
    """
    if len(particles) < 10:
        return {"Nch": 0, "mean_pT": 0, "v2": 0, "v3": 0}
    
    pts, phis, etas = [], [], []
    for px, py, pz, pid in particles:
        pt = math.sqrt(px**2 + py**2)
        if pt < pt_min or pt > pt_max:
            continue
        p_tot = math.sqrt(px**2 + py**2 + pz**2)
        if p_tot < 1e-6:
            continue
        eta = 0.5 * math.log((p_tot+pz)/(p_tot-pz+1e-10))
        if abs(eta) > eta_max:
            continue
        pts.append(pt)
        phis.append(math.atan2(py, px))
    
    if len(pts) < 5:
        return {"Nch": 0, "mean_pT": 0, "v2": 0, "v3": 0}
    
    N = len(pts)
    v2 = math.sqrt(
        (sum(math.cos(2*phi) for phi in phis)/N)**2 +
        (sum(math.sin(2*phi) for phi in phis)/N)**2)
    v3 = math.sqrt(
        (sum(math.cos(3*phi) for phi in phis)/N)**2 +
        (sum(math.sin(3*phi) for phi in phis)/N)**2)
    
    return {
        "Nch":     N,
        "mean_pT": sum(pts)/N,
        "v2":      v2,
        "v3":      v3,
    }


if __name__ == "__main__":
    print("Testing Cooper-Frye sampler...")
    
    # load FNO model
    import sys
    sys.path.insert(0, "/afs/cern.ch/work/h/hsharma/epos4/ml")
    from fno import HydroFNO
    from vhlle_dataset import VHLLEDataset
    
    ckpt = torch.load(
        "/afs/cern.ch/work/h/hsharma/epos4/ml/checkpoints/fno_vhlle_best.pt",
        map_location="cpu")
    model = HydroFNO(in_channels=1, out_channels=4, d_model=32,
                     n_blocks=4, modes1=10, modes2=10, n_cond=0)
    model.load_state_dict(ckpt["model"]); model.eval()
    
    X_mean=ckpt["X_mean"]; X_std=ckpt["X_std"]
    Y_mean=ckpt["Y_mean"]; Y_std=ckpt["Y_std"]
    
    ds = VHLLEDataset(max_events=20, verbose=False)
    
    print(f"\n{'Event':>5} {'b(fm)':>6} "
          f"{'Nch_vhlle':>10} {'Nch_fno':>8} "
          f"{'v2_vhlle':>9} {'v2_fno':>7} "
          f"{'T_err':>7}")
    print("-"*65)
    
    for i in range(min(10, len(ds))):
        X, Y = ds[i]
        
        # vHLLE observables (from true freeze-out)
        obs_true = compute_observables(
            sample_event(Y, n_target=2000, seed=42))
        
        # FNO prediction
        Xn = (X - X_mean) / X_std
        with torch.no_grad():
            pred_n = model(Xn.unsqueeze(0))[0]
        pred = pred_n * Y_std[0,:,0,0].unsqueeze(-1).unsqueeze(-1) + \
               Y_mean[0,:,0,0].unsqueeze(-1).unsqueeze(-1)
        
        obs_fno = compute_observables(
            sample_event(pred, n_target=2000, seed=42))
        
        T_err = abs(Y[0].mean() - pred[0].mean()).item()
        
        print(f"{i:>5} {'?':>6} "
              f"{obs_true['Nch']:>10} {obs_fno['Nch']:>8} "
              f"{obs_true['v2']:>9.4f} {obs_fno['v2']:>7.4f} "
              f"{T_err:>7.4f}")
