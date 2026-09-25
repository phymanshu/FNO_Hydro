"""
Extract integrated freeze-out scalars from vHLLE freezeout.dat files.
Stores per-event summary instead of binned 2D fields.

Output per event:
  b          - impact parameter (fm)
  T_mean     - mean freeze-out temperature (GeV)
  T_std      - temperature spread
  ux_mean    - mean x-flow
  uy_mean    - mean y-flow
  N_cells    - number of freeze-out cells
  eps2_fo    - eccentricity of freeze-out surface
  eps3_fo    - triangularity of freeze-out surface
"""
import os, glob, re
import numpy as np
import torch

PROD_DIR = "/eos/user/h/hsharma/vhlle_output"
OUT_FILE = "/eos/user/h/hsharma/vhlle_output/fo_scalars.npz"

def extract_scalars(fo_path, par_path):
    """Extract scalar summary from one freeze-out file."""
    # get b from params
    b = None
    try:
        with open(par_path) as f:
            for line in f:
                if "impactPar" in line:
                    b = float(line.split()[1]); break
    except: return None

    # load freeze-out surface
    try:
        data = np.loadtxt(fo_path, max_rows=200000)
    except: return None
    if len(data) < 50: return None

    # columns: tau x y eta dsig_tau dsig_x dsig_y dsig_eta
    #          utau ux uy ueta T mub ...
    x   = data[:, 1]
    y   = data[:, 2]
    T   = data[:, 12]
    ux  = data[:, 9]  if data.shape[1] > 9  else np.zeros(len(data))
    uy  = data[:, 10] if data.shape[1] > 10 else np.zeros(len(data))

    # weight by |dσ| ~ cell volume
    w = np.ones(len(data))  # equal weight for now

    T_mean  = np.average(T,  weights=w)
    T_std   = np.sqrt(np.average((T-T_mean)**2, weights=w))
    ux_mean = np.average(ux, weights=w)
    uy_mean = np.average(uy, weights=w)
    N_cells = len(data)

    # eccentricity of freeze-out surface
    x_cm = np.average(x, weights=w)
    y_cm = np.average(y, weights=w)
    xc = x - x_cm; yc = y - y_cm
    r2 = xc**2 + yc**2
    phi = np.arctan2(yc, xc)
    denom2 = np.average(r2, weights=w) + 1e-10
    eps2_fo = np.sqrt(np.average(r2*np.cos(2*phi), weights=w)**2 +
                      np.average(r2*np.sin(2*phi), weights=w)**2) / denom2
    r3 = r2 * np.sqrt(r2 + 1e-10)
    denom3 = np.average(r3, weights=w) + 1e-10
    eps3_fo = np.sqrt(np.average(r3*np.cos(3*phi), weights=w)**2 +
                      np.average(r3*np.sin(3*phi), weights=w)**2) / denom3

    return dict(b=b, T_mean=T_mean, T_std=T_std,
                ux_mean=ux_mean, uy_mean=uy_mean,
                N_cells=N_cells, eps2_fo=eps2_fo, eps3_fo=eps3_fo)


if __name__ == "__main__":
    entries = os.listdir(PROD_DIR)
    event_dirs = sorted([os.path.join(PROD_DIR, e) for e in entries
                         if re.match(r"event_\d+$", e)],
                        key=lambda p: int(p.split("_")[-1]))
    print(f"Found {len(event_dirs)} event directories")

    results = []
    for i, d in enumerate(event_dirs):
        fo  = os.path.join(d, "freezeout.dat")
        par = os.path.join(d, "params.dat")
        if not (os.path.exists(fo) and os.path.exists(par)):
            continue
        sc = extract_scalars(fo, par)
        if sc is not None:
            results.append(sc)
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(event_dirs)} done, {len(results)} valid")

    # save as npz
    keys = results[0].keys()
    out = {k: np.array([r[k] for r in results]) for k in keys}
    np.savez(OUT_FILE, **out)

    print(f"\nSaved {len(results)} events to {OUT_FILE}")
    print(f"\nSummary:")
    print(f"  b:       {out['b'].min():.2f} - {out['b'].max():.2f} fm")
    print(f"  T_mean:  {out['T_mean'].mean():.4f} ± {out['T_mean'].std():.4f} GeV")
    print(f"  ux_mean: {out['ux_mean'].mean():.4f} ± {out['ux_mean'].std():.4f}")
    print(f"  eps2_fo: {out['eps2_fo'].mean():.4f} ± {out['eps2_fo'].std():.4f}")
    print(f"  N_cells: {out['N_cells'].mean():.0f} ± {out['N_cells'].std():.0f}")
