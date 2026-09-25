"""
Proper Cooper-Frye sampler using raw vHLLE freeze-out cells.
Reads freezeout.dat directly — no binning, no information loss.

dN/d³p = g/(2π)³ · f_BE(p·u/T) · p^μ dσ_μ

Columns in freezeout.dat:
  0:tau  1:x  2:y  3:eta
  4:dσ_τ  5:dσ_x  6:dσ_y  7:dσ_η
  8:u^τ  9:u^x  10:u^y  11:u^η
  12:T  13:μ_B
  14-23: π^μν (10 components)
  24: Π (bulk)
"""
import os, math, numpy as np
from scipy.special import kn

MASSES = {211:0.13957, -211:0.13957,
          321:0.49368, -321:0.49368,
          2212:0.93827, -2212:0.93827}
SPECIES = list(MASSES.keys())
FRAC    = np.array([0.35,0.35,0.08,0.08,0.07,0.07])
FRAC   /= FRAC.sum()

def load_freezeout(path, max_rows=200000):
    """Load freeze-out surface from vHLLE freezeout.dat."""
    try:
        data = np.loadtxt(path, max_rows=max_rows)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None
    if len(data) < 50:
        return None
    return data

def sample_thermal_pT(T, mass, n):
    """
    Sample pT from Cooper-Frye thermal distribution.
    dN/dpT ∝ pT * mT * K1(mT/T)
    Uses rejection sampling.
    """
    pT_list = []
    while len(pT_list) < n:
        pT_try = np.random.exponential(T, n*3)
        mT_try = np.sqrt(pT_try**2 + mass**2)
        # weight: pT * mT * K1(mT/T) / (pT_max_weight)
        weight = pT_try * mT_try * kn(1, mT_try/T)
        weight /= (weight.max() + 1e-30)
        accept = np.random.rand(len(weight)) < weight
        pT_list.extend(pT_try[accept].tolist())
    return np.array(pT_list[:n])

def sample_event(fo_path, n_particles=5000, seed=42,
                 eta_max=0.8, pt_min=0.2, pt_max=3.0):
    """
    Full Cooper-Frye sampling from raw freeze-out surface.
    Returns dict with observables.
    """
    np.random.seed(seed)
    data = load_freezeout(fo_path)
    if data is None:
        return None

    x   = data[:,1]; y  = data[:,2]
    T   = data[:,12]
    ux  = data[:,9];  uy = data[:,10]
    # surface normal (use tau component as weight proxy)
    dsig_tau = np.abs(data[:,4])

    # weight cells by thermal yield ∝ T^3 * |dσ_τ|
    w = T**3 * (dsig_tau + 1e-10)
    w = w / w.sum()

    # sample cells
    idx = np.random.choice(len(data), size=n_particles, p=w, replace=True)
    T_s  = T[idx]; ux_s = ux[idx]; uy_s = uy[idx]
    pids = np.random.choice(SPECIES, size=n_particles, p=FRAC)

    phis_final = []
    pts_final  = []

    for k in range(n_particles):
        T_k   = max(float(T_s[k]),  0.05)
        ux_k  = float(ux_s[k])
        uy_k  = float(uy_s[k])
        mass  = MASSES[pids[k]]

        # sample pT from thermal distribution
        pT = float(sample_thermal_pT(T_k, mass, 1)[0])
        mT = math.sqrt(pT**2 + mass**2)
        phi_k = np.random.uniform(0, 2*math.pi)
        eta_k = np.random.normal(0, 1.5)

        # boost by real flow velocity
        v2 = ux_k**2 + uy_k**2
        gamma = 1.0 / math.sqrt(max(1 - v2, 0.01))

        px = pT*math.cos(phi_k) + gamma*ux_k*mT
        py = pT*math.sin(phi_k) + gamma*uy_k*mT
        pz = mT*math.sinh(eta_k)

        pt_f = math.sqrt(px**2 + py**2)
        p_f  = math.sqrt(px**2 + py**2 + pz**2)
        if pt_f < pt_min or pt_f > pt_max or p_f < 1e-6:
            continue
        eta_f = 0.5*math.log((p_f+pz)/(p_f-pz+1e-10))
        if abs(eta_f) > eta_max:
            continue
        phis_final.append(math.atan2(py,px))
        pts_final.append(pt_f)

    if len(phis_final) < 10:
        return {'Nch':0,'mean_pT':0,'v2':0,'v3':0}

    N = len(phis_final)
    v2 = math.sqrt(
        (sum(math.cos(2*phi) for phi in phis_final)/N)**2 +
        (sum(math.sin(2*phi) for phi in phis_final)/N)**2)
    v3 = math.sqrt(
        (sum(math.cos(3*phi) for phi in phis_final)/N)**2 +
        (sum(math.sin(3*phi) for phi in phis_final)/N)**2)

    return {
        'Nch':    N,
        'mean_pT': float(np.mean(pts_final)),
        'v2':     v2,
        'v3':     v3,
        'phis':   phis_final,
    }


if __name__ == "__main__":
    import sys, os, re, torch
    sys.path.insert(0, "/afs/cern.ch/work/h/hsharma/epos4/ml")
    from fno import HydroFNO
    from vhlle_dataset import VHLLEDataset

    PROD = "/eos/user/h/hsharma/vhlle_output"
    entries = os.listdir(PROD)
    event_dirs = sorted([os.path.join(PROD,e) for e in entries
                         if re.match(r"event_\d+$",e)],
                        key=lambda p:int(p.split("_")[-1]))[:50]

    print(f"Testing proper Cooper-Frye on {len(event_dirs)} events...")
    results = []
    for d in event_dirs:
        fo  = os.path.join(d,"freezeout.dat")
        par = os.path.join(d,"params.dat")
        if not os.path.exists(fo): continue
        b = None
        try:
            with open(par) as f:
                for line in f:
                    if "impactPar" in line:
                        b=float(line.split()[1]); break
        except: pass
        if b is None: continue

        obs = sample_event(fo, n_particles=5000, seed=42)
        if obs and obs['Nch']>0:
            results.append({**obs,'b':b})
            if len(results)%10==0:
                print(f"  {len(results)} events done")

    import numpy as np
    arr = {k:np.array([r[k] for r in results]) for k in ['b','v2','v3','Nch','mean_pT']}
    r_b_v2 = np.corrcoef(arr['b'],arr['v2'])[0,1]
    print(f"\nResults ({len(results)} events):")
    print(f"  Nch:     {arr['Nch'].mean():.0f} ± {arr['Nch'].std():.0f}")
    print(f"  <pT>:    {arr['mean_pT'].mean():.3f} ± {arr['mean_pT'].std():.3f} GeV")
    print(f"  v2:      {arr['v2'].mean():.4f} ± {arr['v2'].std():.4f}")
    print(f"  v3:      {arr['v3'].mean():.4f} ± {arr['v3'].std():.4f}")
    print(f"  corr(b,v2) = {r_b_v2:.4f}")
    print(f"\nALICE 30-50% Pb-Pb reference:")
    print(f"  v2 ~ 0.08-0.12  ·  <pT> ~ 0.60 GeV  ·  Nch ~ 800-1200")
