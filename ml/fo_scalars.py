"""
Extract scalar summary from freeze-out surface per event.
Tests if FNO's predicted T(x,y) field captures freeze-out geometry.
"""
import os, re, torch, numpy as np
from vhlle_dataset import VHLLEDataset
from fno import HydroFNO

PROD_DIR = "/eos/user/h/hsharma/vhlle_output"

def extract_fo_scalars(fo_path):
    """Extract geometry and thermodynamics from raw freeze-out file."""
    try:
        data = np.loadtxt(fo_path, max_rows=200000)
    except: return None
    if len(data) < 50: return None

    x   = data[:, 1]
    y   = data[:, 2]
    T   = data[:, 12]
    ux  = data[:, 9]
    uy  = data[:, 10]

    # weight by T^3 (thermal weight ~ particle density)
    w = T**3 + 1e-10

    T_mean  = np.average(T,  weights=w)
    ux_mean = np.average(ux, weights=w)
    uy_mean = np.average(uy, weights=w)
    N_cells = len(data)

    # eccentricity of freeze-out surface
    x_cm = np.average(x, weights=w)
    y_cm = np.average(y, weights=w)
    xc = x - x_cm; yc = y - y_cm
    r2 = xc**2 + yc**2
    phi = np.arctan2(yc, xc)
    denom = np.average(r2, weights=w) + 1e-10
    eps2_fo = np.sqrt(np.average(r2*np.cos(2*phi), weights=w)**2 +
                      np.average(r2*np.sin(2*phi), weights=w)**2) / denom
    r3 = r2 * np.sqrt(r2 + 1e-10)
    denom3 = np.average(r3, weights=w) + 1e-10
    eps3_fo = np.sqrt(np.average(r3*np.cos(3*phi), weights=w)**2 +
                      np.average(r3*np.sin(3*phi), weights=w)**2) / denom3

    return dict(T_mean=T_mean, ux_mean=ux_mean, uy_mean=uy_mean,
                N_cells=N_cells, eps2_fo=eps2_fo, eps3_fo=eps3_fo,
                x_cm=x_cm, y_cm=y_cm)


def T_field_scalars(T_field):
    """Extract same scalars from FNO's predicted T(x,y) field."""
    import torch
    T = T_field.numpy()  # [61, 61]
    x = np.linspace(-12, 12, 61)
    y = np.linspace(-12, 12, 61)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # weight by T^3
    w = T**3 + 1e-10
    w_sum = w.sum()

    T_mean = (T * w).sum() / w_sum
    x_cm = (X * w).sum() / w_sum
    y_cm = (Y * w).sum() / w_sum
    Xc = X - x_cm; Yc = Y - y_cm
    r2 = Xc**2 + Yc**2
    phi = np.arctan2(Yc, Xc)
    denom = (w * r2).sum() / w_sum + 1e-10
    eps2 = np.sqrt(((w*r2*np.cos(2*phi)).sum()/w_sum)**2 +
                   ((w*r2*np.sin(2*phi)).sum()/w_sum)**2) / denom
    r3 = r2 * np.sqrt(r2 + 1e-10)
    denom3 = (w * r3).sum() / w_sum + 1e-10
    eps3 = np.sqrt(((w*r3*np.cos(3*phi)).sum()/w_sum)**2 +
                   ((w*r3*np.sin(3*phi)).sum()/w_sum)**2) / denom3

    return dict(T_mean=T_mean, eps2_fo=eps2, eps3_fo=eps3,
                x_cm=x_cm, y_cm=y_cm)


if __name__ == "__main__":
    # load model
    ckpt = torch.load(
        "/afs/cern.ch/work/h/hsharma/epos4/ml/checkpoints/fno_vhlle_large.pt",
        map_location="cpu")
    model = HydroFNO(in_channels=1, out_channels=4, d_model=64,
                     n_blocks=4, modes1=16, modes2=16, n_cond=0)
    model.load_state_dict(ckpt["model"]); model.eval()
    X_mean=ckpt["X_mean"]; X_std=ckpt["X_std"]
    Y_mean=ckpt["Y_mean"]; Y_std=ckpt["Y_std"]

    # load dataset
    ds = VHLLEDataset(max_events=100, verbose=True)

    # get event dirs
    entries = os.listdir(PROD_DIR)
    event_dirs = sorted([os.path.join(PROD_DIR, e) for e in entries
                         if re.match(r"event_\d+$", e)],
                        key=lambda p: int(p.split("_")[-1]))[:100]

    results = []
    for i, (d, ds_item) in enumerate(zip(event_dirs, ds)):
        fo_path = os.path.join(d, "freezeout.dat")
        if not os.path.exists(fo_path): continue

        # ground truth from raw freeze-out
        sc_true = extract_fo_scalars(fo_path)
        if sc_true is None: continue

        # FNO prediction
        X, Y = ds_item
        Xn = (X - X_mean) / X_std
        with torch.no_grad():
            pred_n = model(Xn.unsqueeze(0))[0]
        pred = pred_n * Y_std[0,:,0,0].unsqueeze(-1).unsqueeze(-1) + \
               Y_mean[0,:,0,0].unsqueeze(-1).unsqueeze(-1)

        # extract scalars from FNO T field
        sc_fno = T_field_scalars(pred[0])  # channel 0 = T

        results.append({
            "T_true": sc_true["T_mean"],
            "T_fno":  sc_fno["T_mean"],
            "eps2_true": sc_true["eps2_fo"],
            "eps2_fno":  sc_fno["eps2_fo"],
            "eps3_true": sc_true["eps3_fo"],
            "eps3_fno":  sc_fno["eps3_fo"],
            "N_true": sc_true["N_cells"],
        })

    arr = {k: np.array([r[k] for r in results]) for k in results[0]}
    print(f"\nResults for {len(results)} events:")
    print(f"\nTemperature:")
    print(f"  True T_mean: {arr['T_true'].mean():.4f} ± {arr['T_true'].std():.4f} GeV")
    print(f"  FNO  T_mean: {arr['T_fno'].mean():.4f} ± {arr['T_fno'].std():.4f} GeV")
    print(f"  Corr(T_true, T_fno): {np.corrcoef(arr['T_true'],arr['T_fno'])[0,1]:.4f}")

    print(f"\nEccentricity ε₂ (predicts v2):")
    print(f"  True ε₂: {arr['eps2_true'].mean():.4f} ± {arr['eps2_true'].std():.4f}")
    print(f"  FNO  ε₂: {arr['eps2_fno'].mean():.4f} ± {arr['eps2_fno'].std():.4f}")
    print(f"  Corr(ε₂_true, ε₂_fno): {np.corrcoef(arr['eps2_true'],arr['eps2_fno'])[0,1]:.4f}")

    print(f"\nTriangularity ε₃ (predicts v3):")
    print(f"  True ε₃: {arr['eps3_true'].mean():.4f} ± {arr['eps3_true'].std():.4f}")
    print(f"  FNO  ε₃: {arr['eps3_fno'].mean():.4f} ± {arr['eps3_fno'].std():.4f}")
    print(f"  Corr(ε₃_true, ε₃_fno): {np.corrcoef(arr['eps3_true'],arr['eps3_fno'])[0,1]:.4f}")

    # save
    np.savez("/eos/user/h/hsharma/epos4_plots/fo_scalar_comparison.npz", **arr)
    print("\nSaved to CERNBox.")
