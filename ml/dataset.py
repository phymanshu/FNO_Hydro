"""
EPOS4 HepMC3 parser and PyTorch dataset.
Extracts b per event from GenHeavyIon record and filters by centrality.
"""
import os, torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
os.environ["OMP_NUM_THREADS"] = "1"

FINAL_STATE_PDGS = {
    211, -211, 111,
    321, -321, 311, -311, 130, 310,
    2212, -2212, 2112, -2112,
    3122, -3122, 3222, -3222, 3112, -3112,
    3312, -3312, 3334, -3334, 22,
}
PDG_TO_IDX = {pdg: i for i, pdg in enumerate(sorted(FINAL_STATE_PDGS))}
N_SPECIES   = len(PDG_TO_IDX)
BEAM_PZ_CUT = 100.0

# Pb-Pb centrality b ranges (fm) from iclpPb8T.optns
CENTRALITY_RANGES = {
    "0-5":   (0.00, 1.88),
    "5-10":  (1.88, 2.65),
    "10-20": (2.65, 3.73),
    "20-30": (3.73, 4.57),
    "30-40": (4.57, 5.27),
    "40-50": (5.27, 5.89),
    "50-60": (5.89, 6.45),
    "0-10":  (0.00, 2.65),
    "30-50": (4.57, 5.89),
    "minbias": (0.00, 99.0),
}

def parse_hepmc_file(filepath, b_min=0.0, b_max=99.0):
    """
    Parse HepMC3 file. Returns list of (b, particles_tensor) tuples.
    Filters events by impact parameter b in [b_min, b_max] fm.
    """
    events = []
    current = []
    current_b = None

    with open(filepath) as f:
        for line in f:
            if line.startswith("E "):
                # save previous event if b in range
                if current and current_b is not None:
                    if b_min <= current_b <= b_max:
                        t = torch.tensor(current, dtype=torch.float32)
                        events.append((current_b, t))
                current = []
                current_b = None

            elif line.startswith("A ") and "GenHeavyIon" in line:
                parts = line.split()
                try:
                    current_b = float(parts[14])  # col 15, 0-indexed=14
                except (IndexError, ValueError):
                    current_b = None

            elif line.startswith("P "):
                parts  = line.split()
                status = int(parts[9])
                pdg    = int(parts[3])
                if status != 1 or pdg not in FINAL_STATE_PDGS:
                    continue
                px, py, pz, E = map(float, parts[4:8])
                if abs(pz) > BEAM_PZ_CUT:
                    continue
                current.append([px, py, pz, E, float(PDG_TO_IDX.get(pdg, 0))])

    # save last event
    if current and current_b is not None:
        if b_min <= current_b <= b_max:
            events.append((current_b, torch.tensor(current, dtype=torch.float32)))

    return events


class EPOS4Dataset(Dataset):
    """
    PyTorch Dataset for EPOS4 HepMC3 files.
    Each item: (b, particles_tensor) where particles: [N, 5] (px,py,pz,E,pid)
    """
    def __init__(self, data_dir, centrality="30-50", max_files=None, verbose=True):
        self.centrality = centrality
        b_min, b_max = CENTRALITY_RANGES.get(centrality, (0, 99))
        self.b_min = b_min
        self.b_max = b_max
        self.events = []   # list of (b, tensor)

        files = sorted([f for f in os.listdir(data_dir) if f.endswith(".hepmc")])
        if max_files:
            files = files[:max_files]

        n_total = 0
        for i, fname in enumerate(files):
            evs = parse_hepmc_file(os.path.join(data_dir, fname), b_min, b_max)
            self.events.extend(evs)
            n_total += 1  # each file has 1 event (1 per job)
            if verbose and (i+1) % 50 == 0:
                print(f"  {i+1}/{len(files)} files, {len(self.events)} events passing centrality cut")

        if verbose:
            self._stats()

    def _stats(self):
        if not self.events:
            print(f"  No events passing centrality {self.centrality} cut!")
            return
        sizes = [e[1].shape[0] for e in self.events]
        bs    = [e[0] for e in self.events]
        pts   = [e[1][:, :2].norm(dim=1).mean().item() for e in self.events]
        print(f"\nDataset [{self.centrality}%]: {len(self.events)} events")
        print(f"  b range:         {min(bs):.2f} - {max(bs):.2f} fm  (mean={sum(bs)/len(bs):.2f})")
        print(f"  Particles/event: mean={sum(sizes)/len(sizes):.0f}  min={min(sizes)}  max={max(sizes)}")
        print(f"  Mean pT/event:   {sum(pts)/len(pts):.3f} GeV")

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        b, particles = self.events[idx]
        return b, particles


def collate_events(batch):
    bs        = torch.tensor([item[0] for item in batch])
    particles = [item[1] for item in batch]
    lengths   = [p.shape[0] for p in particles]
    N_max     = max(lengths)
    B         = len(batch)
    padded    = torch.zeros(B, N_max, 5)
    mask      = torch.zeros(B, N_max, dtype=torch.bool)
    for i, (p, n) in enumerate(zip(particles, lengths)):
        padded[i, :n] = p
        mask[i, :n]   = True
    return bs, padded, mask, torch.tensor(lengths)


def compute_observables(particles):
    px, py, pz, E = (particles[:, i] for i in range(4))
    pt  = torch.sqrt(px**2 + py**2)
    phi = torch.atan2(py, px)
    eta = torch.asinh(pz / (pt + 1e-8))
    v2  = torch.sqrt(torch.cos(2*phi).mean()**2 + torch.sin(2*phi).mean()**2)
    v3  = torch.sqrt(torch.cos(3*phi).mean()**2 + torch.sin(3*phi).mean()**2)
    return {"N": pt.shape[0], "mean_pT": pt.mean().item(),
            "v2": v2.item(), "v3": v3.item(),
            "eta_mean": eta.mean().item()}


if __name__ == "__main__":
    import sys
    data_dir   = sys.argv[1] if len(sys.argv) > 1 else "."
    centrality = sys.argv[2] if len(sys.argv) > 2 else "30-50"

    print(f"Loading {centrality}% centrality events from {data_dir}...")
    ds = EPOS4Dataset(data_dir, centrality=centrality)

    if len(ds) == 0:
        print("No events found. Try 'minbias' centrality.")
        sys.exit(1)

    print("\nSample observables:")
    for i in range(min(5, len(ds))):
        b, ev = ds[i]
        obs = compute_observables(ev)
        print(f"  Event {i}: b={b:.2f}fm  N={obs['N']}  "
              f"<pT>={obs['mean_pT']:.3f}  v2={obs['v2']:.4f}  v3={obs['v3']:.4f}")

    # show centrality breakdown of full dataset
    print(f"\nCentrality breakdown of all available events:")
    all_ds = EPOS4Dataset(data_dir, centrality="minbias", verbose=False)
    bs_all = [e[0] for e in all_ds.events]
    for cent, (blo, bhi) in CENTRALITY_RANGES.items():
        if cent == "minbias": continue
        n = sum(1 for b in bs_all if blo <= b < bhi)
        print(f"  {cent:8s}: {n:4d} events  (b={blo:.2f}-{bhi:.2f} fm)")
