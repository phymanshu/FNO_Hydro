"""
PointNet for Stage 2: predict ux, uy at raw freeze-out cells.
Input:  [B, N_pts, 3]  (x, y, T_fno)
Output: [B, N_pts, 2]  (ux, uy)

Architecture:
  per-point MLP → global max-pool → concat back → per-point MLP → ux, uy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetStage2(nn.Module):
    """
    PointNet for flow velocity prediction at freeze-out cells.
    
    Two-branch design:
      Local branch:  per-point features from (x, y, T_fno)
      Global branch: max-pool over all points → global context
      Concat → predict (ux, uy) per point
    """
    def __init__(self, in_dim=3, hidden=128, out_dim=2):
        super().__init__()

        # local feature extractor (shared MLP across points)
        self.local_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        # global feature extractor (after max-pool)
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        # prediction head (local + global context)
        self.pred_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, pts):
        """
        pts: [B, N, 3]  (x, y, T_fno) — normalised
        returns: [B, N, 2]  (ux, uy)
        """
        B, N, _ = pts.shape

        # local features per point
        local_feat = self.local_mlp(pts)          # [B, N, hidden]

        # global feature via max-pool
        global_feat = local_feat.max(dim=1)[0]    # [B, hidden]
        global_feat = self.global_mlp(global_feat) # [B, hidden]

        # broadcast global to each point
        global_exp = global_feat.unsqueeze(1).expand(-1, N, -1)  # [B, N, hidden]

        # concat local + global
        combined = torch.cat([local_feat, global_exp], dim=-1)   # [B, N, 2*hidden]

        # predict ux, uy
        out = self.pred_mlp(combined)              # [B, N, 2]
        return out


if __name__ == "__main__":
    model = PointNetStage2(in_dim=3, hidden=128, out_dim=2)
    n = sum(p.numel() for p in model.parameters())
    print(f"PointNet Stage 2: {n/1e6:.3f}M parameters")

    # test forward
    B, N = 4, 4096
    pts = torch.randn(B, N, 3)
    out = model(pts)
    print(f"Input:  {pts.shape}")
    print(f"Output: {out.shape}  (ux, uy per point)")
    print("Forward pass: OK")

    # test that different points give different outputs
    pts2 = pts.clone(); pts2[:,:,0] *= -1  # flip x
    out2 = model(pts2)
    diff = (out - out2).abs().mean().item()
    print(f"Output changes with input: {diff:.6f} (should be > 0)")
