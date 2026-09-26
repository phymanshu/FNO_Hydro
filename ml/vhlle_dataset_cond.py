"""
Conditional FNO dataset: combines MC Glauber events at multiple eta/s values.
Each sample returns (IC field, eta/s scalar, freeze-out fields).
"""
import os, torch
import numpy as np
from torch.utils.data import Dataset

# eta/s values and their EOS directories
ETAS_DIRS = {
    0.08: "/eos/user/h/hsharma/vhlle_mcglauber",
    0.05: "/eos/user/h/hsharma/vhlle_mcglauber_etascan/etas_0.05",
    0.12: "/eos/user/h/hsharma/vhlle_mcglauber_etascan/etas_0.12",
    0.20: "/eos/user/h/hsharma/vhlle_mcglauber_etascan/etas_0.20",
}

class VHLLEDatasetCond(Dataset):
    def __init__(self, max_per_etas=None, verbose=True):
        self.samples = []  # (ic_path, fo_path, etas)

        for etas, base in ETAS_DIRS.items():
            if not os.path.exists(base):
                if verbose: print(f"  eta/s={etas}: dir missing, skip")
                continue
            entries = sorted(
                [e for e in os.listdir(base) if e.startswith("event_")],
                key=lambda e: int(e.split("_")[1]))
            if max_per_etas:
                entries = entries[:max_per_etas]
            count = 0
            for ev in entries:
                ic = os.path.join(base, ev, "ic2D.dat")
                fo = os.path.join(base, ev, "freezeout.dat")
                if os.path.exists(ic) and os.path.exists(fo):
                    self.samples.append((ic, fo, etas))
                    count += 1
            if verbose:
                print(f"  eta/s={etas}: {count} events")

        if verbose:
            print(f"Total: {len(self.samples)} events across "
                  f"{len(set(s[2] for s in self.samples))} eta/s values")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        ic_path, fo_path, etas = self.samples[idx]

        # load IC
        try:
            d = np.loadtxt(ic_path)
            eps = d[:,2].reshape(61,61).astype(np.float32)
        except:
            eps = np.zeros((61,61), dtype=np.float32)
        X = torch.tensor(eps).unsqueeze(0)  # [1,61,61]

        # eta/s as scalar
        etas_t = torch.tensor([etas], dtype=torch.float32)  # [1]

        # load freeze-out
        try:
            data = np.loadtxt(fo_path, max_rows=200000)
        except:
            return X, etas_t, torch.zeros(4,61,61)
        if len(data) < 50:
            return X, etas_t, torch.zeros(4,61,61)

        x_fo=data[:,1]; y_fo=data[:,2]
        T=data[:,12]; ux=data[:,9]; uy=data[:,10]
        dsig=np.sqrt(data[:,4]**2+data[:,5]**2+
                     data[:,6]**2+data[:,7]**2)+1e-10

        xedges=np.linspace(-12,12,62); yedges=np.linspace(-12,12,62)
        def bin_field(vals, weights):
            grid=np.zeros((61,61),dtype=np.float32)
            cnt =np.zeros((61,61),dtype=np.float32)
            ix=np.clip(np.digitize(x_fo,xedges)-1,0,60)
            iy=np.clip(np.digitize(y_fo,yedges)-1,0,60)
            np.add.at(grid,(ix,iy),vals*weights)
            np.add.at(cnt, (ix,iy),weights)
            mask=cnt>0; grid[mask]/=cnt[mask]
            return grid

        w = T**3 * dsig
        Y = torch.tensor(np.stack([
            bin_field(T,  w),
            bin_field(ux, w),
            bin_field(uy, w),
            bin_field(np.ones(len(T)), w)
        ]))  # [4,61,61]

        return X, etas_t, Y
