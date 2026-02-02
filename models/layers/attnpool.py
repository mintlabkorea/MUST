# models/layers/attnpool.py (새 파일)
import torch, torch.nn as nn, torch.nn.functional as F

class AttnPool1D(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x, mask=None):
        # x: (B,T,D)
        s = self.score(x).squeeze(-1)            # (B,T)
        if mask is not None:
            s = s.masked_fill(~mask, -1e9)
        w = torch.softmax(s, dim=1)              # (B,T)
        out = (x * w.unsqueeze(-1)).sum(dim=1)   # (B,D)
        return out, w
