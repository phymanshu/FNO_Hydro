"""VHLLEDataset for MC Glauber events."""
import os, torch
import numpy as np
from torch.utils.data import Dataset

PROD_DIR = "/eos/user/h/hsharma/vhlle_mcglauber"

class VHLLEDatasetMC(Dataset):
    def __init__(self, max_events=5000, verbose=True):
        self.samples = []
        entries = sorted([e for e in os.listdir(PROD_DIR)
                         if e.startswith("event_")],
                        key=lambda e: int(e.split("_")[1]))
        for ev in entries[:max_events]:
            ic = os.path.join(PROD_DIR, ev, "ic2D.dat")
            fo = os.path.join(PROD_DIR, ev, "freezeout.dat")
            bv = os.path.join(PROD_DIR, ev, "b_value.txt")
            if os.path.exists(ic) and os.path.exists(fo):
                self.samples.append((ic, fo, bv))
        if verbose:
            print(f"MC Glauber Dataset: {len(self.samples)} events")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        ic_path, fo_path, bv_path = self.samples[idx]

        # load IC: no header, raw x y eps (61x61=3721 rows)
        d = np.loadtxt(ic_path)          # no skiprows
        # take only rows with valid data (3 columns)
        if d.ndim==2 and d.shape[1]==3:
            eps = d[:,2].reshape(61,61).astype(np.float32)
        else:
            eps = np.zeros((61,61),dtype=np.float32)
        X = torch.tensor(eps).unsqueeze(0)  # [1,61,61]

        # load freeze-out
        try:
            data = np.loadtxt(fo_path, max_rows=200000)
        except:
            return X, torch.zeros(4,61,61)
        if len(data) < 50:
            return X, torch.zeros(4,61,61)

        x_fo=data[:,1]; y_fo=data[:,2]
        T=data[:,12]; ux=data[:,9]; uy=data[:,10]
        dsig=np.sqrt(data[:,4]**2+data[:,5]**2+
                     data[:,6]**2+data[:,7]**2)+1e-10

        # bin onto 61×61 grid
        xedges=np.linspace(-12,12,62)
        yedges=np.linspace(-12,12,62)
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
        T_grid  = bin_field(T,  w)
        ux_grid = bin_field(ux, w)
        uy_grid = bin_field(uy, w)
        rho_grid= bin_field(np.ones(len(T)), w)

        Y = torch.tensor(np.stack([T_grid,ux_grid,
                                    uy_grid,rho_grid]))
        return X, Y
