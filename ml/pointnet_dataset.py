"""
PointNet dataset for Stage 2: predict ux, uy at raw freeze-out cells.
Uses FNO's T field as global context + raw cell positions as input.

Per sample returns:
  pts:    [N_pts, 3]  (x, y, T_fno_interp)  ← PointNet input
  ux_gt:  [N_pts]     true ux from vHLLE
  uy_gt:  [N_pts]     true uy from vHLLE
  T_gt:   [N_pts]     true T  from vHLLE  (for reference)
"""
import os, re, torch
import numpy as np
from torch.utils.data import Dataset
from scipy.interpolate import RegularGridInterpolator

PROD_DIR = "/eos/user/h/hsharma/vhlle_mcglauber"
CKPT_FNO = "/afs/cern.ch/work/h/hsharma/epos4/ml/checkpoints/fno_mcglauber_d32.pt"
N_PTS    = 4096   # subsample per event (fits in GPU RAM)
XY       = np.linspace(-12, 12, 61)

class PointNetDataset(Dataset):
    def __init__(self, max_events=5000, n_pts=N_PTS,
                 fno_ckpt=CKPT_FNO, verbose=True):
        self.n_pts = n_pts

        # load FNO model for T field prediction
        import sys; sys.path.insert(0,"/afs/cern.ch/work/h/hsharma/epos4/ml")
        from fno import HydroFNO
        ckpt = torch.load(fno_ckpt, map_location="cpu")
        self.fno = HydroFNO(in_channels=1, out_channels=4,
                            d_model=32, n_blocks=4,
                            modes1=16, modes2=16, n_cond=0)
        self.fno.load_state_dict(ckpt["model"])
        self.fno.eval()
        self.X_mean = ckpt["X_mean"]; self.X_std = ckpt["X_std"]
        self.Y_mean = ckpt["Y_mean"]; self.Y_std = ckpt["Y_std"]

        # collect event dirs
        entries = sorted([e for e in os.listdir(PROD_DIR)
                         if e.startswith("event_")],
                        key=lambda e: int(e.split("_")[1]))
        self.samples = []
        for ev in entries[:max_events]:
            ic = os.path.join(PROD_DIR, ev, "ic2D.dat")
            fo = os.path.join(PROD_DIR, ev, "freezeout.dat")
            if os.path.exists(ic) and os.path.exists(fo):
                self.samples.append((ic, fo))
        if verbose:
            print(f"PointNet Dataset: {len(self.samples)} events")
            print(f"  N_pts per event (train): {n_pts}")
            print(f"  FNO checkpoint: {os.path.basename(fno_ckpt)}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        ic_path, fo_path = self.samples[idx]

        # load IC and get FNO T field
        try:
            d = np.loadtxt(ic_path)
            eps = d[:,2].reshape(61,61).astype(np.float32)
        except:
            return self._empty()
        X_in = torch.tensor(eps).unsqueeze(0).unsqueeze(0)  # [1,1,61,61]
        with torch.no_grad():
            Xn = (X_in - self.X_mean) / self.X_std
            pred_n = self.fno(Xn)[0]
            pred = pred_n * self.Y_std[0,:,0,0].unsqueeze(-1).unsqueeze(-1) + \
                   self.Y_mean[0,:,0,0].unsqueeze(-1).unsqueeze(-1)
        T_fno = pred[0].numpy()  # [61,61]

        # interpolator for T_fno at arbitrary (x,y) positions
        T_interp = RegularGridInterpolator(
            (XY, XY), T_fno, method='linear',
            bounds_error=False, fill_value=0.0)

        # load raw freeze-out cells
        try:
            data = np.loadtxt(fo_path, max_rows=200000)
        except:
            return self._empty()
        if len(data) < 100:
            return self._empty()

        x_fo  = data[:,1].astype(np.float32)
        y_fo  = data[:,2].astype(np.float32)
        T_fo  = data[:,12].astype(np.float32)
        ux_fo = data[:,9].astype(np.float32)
        uy_fo = data[:,10].astype(np.float32)

        # filter active cells (T above threshold)
        mask = T_fo > 0.10
        if mask.sum() < 50:
            return self._empty()
        x_fo  = x_fo[mask];  y_fo  = y_fo[mask]
        T_fo  = T_fo[mask];  ux_fo = ux_fo[mask]; uy_fo = uy_fo[mask]

        # subsample N_pts cells randomly
        N = len(x_fo)
        if N > self.n_pts:
            idx_s = np.random.choice(N, self.n_pts, replace=False)
        else:
            idx_s = np.random.choice(N, self.n_pts, replace=True)
        x_s  = x_fo[idx_s];  y_s  = y_fo[idx_s]
        T_s  = T_fo[idx_s];  ux_s = ux_fo[idx_s]; uy_s = uy_fo[idx_s]

        # interpolate FNO T field at sampled cell positions
        pts_query = np.column_stack([x_s, y_s])
        T_fno_at_cells = T_interp(pts_query).astype(np.float32)

        # PointNet input: (x, y, T_fno)
        pts = np.column_stack([
            x_s / 12.0,              # normalise x to [-1, 1]
            y_s / 12.0,              # normalise y to [-1, 1]
            T_fno_at_cells / 0.165   # normalise T to ~[0, 1]
        ]).astype(np.float32)        # [N_pts, 3]

        return (torch.tensor(pts),
                torch.tensor(ux_s),
                torch.tensor(uy_s),
                torch.tensor(T_s))

    def _empty(self):
        return (torch.zeros(self.n_pts, 3),
                torch.zeros(self.n_pts),
                torch.zeros(self.n_pts),
                torch.zeros(self.n_pts))


if __name__ == "__main__":
    ds = PointNetDataset(max_events=5, verbose=True)
    pts, ux, uy, T = ds[0]
    print(f"\nSample 0:")
    print(f"  pts shape:  {pts.shape}   (x, y, T_fno)")
    print(f"  ux shape:   {ux.shape}    mean={ux.mean():.4f}  std={ux.std():.4f}")
    print(f"  uy shape:   {uy.shape}    mean={uy.mean():.4f}  std={uy.std():.4f}")
    print(f"  T shape:    {T.shape}     mean={T.mean():.4f}")
    print(f"\n  pts range: x=[{pts[:,0].min():.2f},{pts[:,0].max():.2f}]"
          f"  y=[{pts[:,1].min():.2f},{pts[:,1].max():.2f}]"
          f"  T=[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]")
