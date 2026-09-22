"""
vHLLE output parser and PyTorch dataset for FNO training.

Each training pair:
  X: ε(x,y) field from ic2D.dat        [1, 61, 61]
  Y: freeze-out summary field           [n_ch, 61, 61]
     binned onto same grid as IC
     channels: T, ux, uy, dσ_norm (4 channels)
"""
import os, glob, torch
import numpy as np
from torch.utils.data import Dataset
os.environ["OMP_NUM_THREADS"] = "1"

PROD_DIR = "/eos/user/h/hsharma/vhlle_output"
NX, NY   = 61, 61
XY_MAX   = 12.0

def parse_ic2d(path):
    """Load ic2D.dat → tensor [1, NX, NY]"""
    data = np.loadtxt(path)
    if data.shape[0] != NX * NY:
        return None
    eps = data[:, 2].reshape(NX, NY)
    return torch.tensor(eps, dtype=torch.float32).unsqueeze(0)

def parse_freezeout(path, nx=NX, ny=NY, xy_max=XY_MAX):
    """
    Load freezeout.dat → tensor [4, NX, NY]
    Bins freeze-out cells onto transverse grid.
    Channels: [T_mean, ux_mean, uy_mean, dσ_density]
    """
    try:
        data = np.loadtxt(path, max_rows=100000)
    except:
        return None
    if len(data) < 100:
        return None

    # columns: tau x y eta dsigma_tau dsigma_x dsigma_y dsigma_eta
    #          utau ux uy ueta T mub pi*10 Pi
    tau = data[:, 0];  x = data[:, 1];  y = data[:, 2]
    T   = data[:, 12] if data.shape[1] > 12 else data[:, -2]
    ux  = data[:, 9]  if data.shape[1] > 9  else np.zeros(len(data))
    uy  = data[:, 10] if data.shape[1] > 10 else np.zeros(len(data))

    # bin onto grid
    dx = 2 * xy_max / nx
    ix = ((x + xy_max) / dx).astype(int).clip(0, nx-1)
    iy = ((y + xy_max) / dx).astype(int).clip(0, ny-1)

    T_grid   = np.zeros((nx, ny))
    ux_grid  = np.zeros((nx, ny))
    uy_grid  = np.zeros((nx, ny))
    cnt      = np.zeros((nx, ny))

    np.add.at(T_grid,  (ix, iy), T)
    np.add.at(ux_grid, (ix, iy), ux)
    np.add.at(uy_grid, (ix, iy), uy)
    np.add.at(cnt,     (ix, iy), 1)

    mask = cnt > 0
    T_grid[mask]  /= cnt[mask]
    ux_grid[mask] /= cnt[mask]
    uy_grid[mask] /= cnt[mask]
    cnt_norm = cnt / (cnt.max() + 1e-10)

    field = np.stack([T_grid, ux_grid, uy_grid, cnt_norm])
    return torch.tensor(field, dtype=torch.float32)


class VHLLEDataset(Dataset):
    """
    Dataset of (ε_field, freeze-out_field) pairs from vHLLE production.
    X: [1, 61, 61]  initial energy density
    Y: [4, 61, 61]  freeze-out summary (T, ux, uy, cell density)
    """
    def __init__(self, prod_dir=PROD_DIR, max_events=None, verbose=True):
        self.pairs = []
        self.b_values = []

        # fast: build paths directly instead of glob (EOS glob is slow)
        import re
        try:
            all_entries = os.listdir(prod_dir)
            event_dirs = sorted([os.path.join(prod_dir, e) for e in all_entries
                                 if re.match(r"event_\d+$", e)],
                                key=lambda p: int(p.split("_")[-1]))
        except Exception:
            event_dirs = sorted(glob.glob(os.path.join(prod_dir, "event_*")))
        if max_events:
            event_dirs = event_dirs[:max_events]

        for i, d in enumerate(event_dirs):
            ic_path = os.path.join(d, "ic2D.dat")
            fo_path = os.path.join(d, "freezeout.dat")
            par_path = os.path.join(d, "params.dat")

            if not (os.path.exists(ic_path) and os.path.exists(fo_path)):
                continue

            # extract b value
            b = None
            try:
                with open(par_path) as f:
                    for line in f:
                        if "impactPar" in line:
                            b = float(line.split()[1])
                            break
            except:
                pass

            X = parse_ic2d(ic_path)
            Y = parse_freezeout(fo_path)

            if X is None or Y is None:
                continue

            self.pairs.append((X, Y))
            if b is not None:
                self.b_values.append(b)

            if verbose and (i+1) % 100 == 0:
                print(f"  {i+1}/{len(event_dirs)} events loaded")

        if verbose:
            print(f"\nVHLLE Dataset: {len(self.pairs)} events")
            if self.b_values:
                bs = np.array(self.b_values)
                print(f"  b range: {bs.min():.2f} - {bs.max():.2f} fm")
            print(f"  X shape: {self.pairs[0][0].shape}")
            print(f"  Y shape: {self.pairs[0][1].shape}")

    def __len__(self):  return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]


if __name__ == "__main__":
    print("Testing VHLLEDataset...")
    ds = VHLLEDataset(max_events=10)
    if len(ds) > 0:
        X, Y = ds[0]
        print(f"  X: {X.shape}  min={X.min():.3f}  max={X.max():.3f}")
        print(f"  Y: {Y.shape}  T_max={Y[0].max():.3f} GeV")
        print("  OK")
