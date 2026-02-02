__import__('torch')
import torch
import torch.nn as nn
from config.config import Config

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn_weights = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        # x: (B, T, D)
        scores = self.attn_weights(x)             # (B, T, 1)
        weights = torch.softmax(scores, dim=1)    # (B, T, 1)
        return (x * weights).sum(dim=1)           # (B, D)

class PPGEncoder(nn.Module):
    """
    1) 2000-frame PPG → CNN + LSTM + Attn
    2) RR 시퀀스 → FC + Attn (동적 길이, 패딩)
    3) RMSSD, SDNN → FC
    → fused → proj → embed_dim
    """
    def __init__(self, cfg: Config):
        super().__init__()
        p = cfg.Encoders.ppg

        # --- 동적 시퀀스 길이 계산 ---
        sr   = cfg.Data.fs
        win  = cfg.Data.window_sec_emo
        self.seq_len       = int(sr * win)
        self.rr_target_len = int(sr * win * 1.5)

        # 1) CNN stack (unchanged)
        layers, in_ch = [], 1
        for out_ch in p['cnn_channels']:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # 2) LSTM + Attn (unchanged)
        self.lstm  = nn.LSTM(p['cnn_channels'][-1], p['lstm_hidden'],
                             p['lstm_layers'], batch_first=True)
        self.attn1 = AttentionLayer(p['lstm_hidden'])

        # --- RR-path: 목표 길이 self.rr_target_len 사용 ---
        self.fc_rr    = nn.Linear(self.rr_target_len, p['rr_dim'])
        self.attn_rr  = AttentionLayer(p['rr_dim'])

        # 3) float inputs (unchanged)
        self.fc_float = nn.Linear(2, p['float_dim'])

        # final projection
        fused_dim = p['lstm_hidden'] + p['rr_dim'] + p['float_dim']
        self.proj   = nn.Linear(fused_dim, p['embed_dim'])
        self.dropout= nn.Dropout(p['dropout'])

    def forward(self, seq, rr_seq, rmssd, sdnn):
        B = seq.size(0)

        # ── 0) NaN 처리 (혹시 collate_fn_unified 에서 빠졌다면 안전하게 한 번 더)
        rr_seq = torch.nan_to_num(rr_seq,    nan=-100.0, posinf=-100.0, neginf=-100.0)
        rmssd  = torch.nan_to_num(rmssd,     nan=-100.0, posinf=-100.0, neginf=-100.0)
        sdnn   = torch.nan_to_num(sdnn,      nan=-100.0, posinf=-100.0, neginf=-100.0)

        # ── 1) PPG CNN+LSTM+Attn (unchanged) ─────────────────────────────
        if seq.dim() == 3:
            B, N, T = seq.shape
            if T < self.seq_len:
                pad = torch.zeros((B*N, 1, self.seq_len - T),
                                  device=seq.device, dtype=seq.dtype)
                x_in = torch.cat([pad, seq.view(B*N,1,T)], dim=2)
            else:
                x_in = seq[..., -self.seq_len:].view(B*N,1,self.seq_len)
            x, _ = self.lstm(self.cnn(x_in).view(B*N,1,-1))
            x = self.attn1(x).view(B, N, -1).mean(dim=1)
        else:
            T = seq.size(-1)
            if T < self.seq_len:
                pad = torch.zeros((B,1,self.seq_len - T),
                                  device=seq.device, dtype=seq.dtype)
                x_in = torch.cat([pad, seq.unsqueeze(1)], dim=2)
            else:
                x_in = seq.unsqueeze(1)[..., -self.seq_len:]
            x, _ = self.lstm(self.cnn(x_in).view(B,1,-1))
            x = self.attn1(x)

        # ── 2) RR-path with mask + 뒤쪽 패딩 ─────────────────────────────
        B, T_rr = rr_seq.shape
        # 2-1) mask 생성 (True: 실제값, False: padding)
        mask = (rr_seq != -100.0)

        # 2-2) 뒤쪽 패딩: 길이 부족하면 뒤에 -100 붙이고, 길이 넘치면 뒤에서 자름
        if T_rr < self.rr_target_len:
            pad_len = self.rr_target_len - T_rr
            padding = torch.full((B, pad_len), -100.0,
                                 device=rr_seq.device, dtype=rr_seq.dtype)
            rr_padded = torch.cat([rr_seq, padding], dim=1)
            mask = torch.cat([mask, torch.zeros((B,pad_len), dtype=torch.bool,
                                                 device=rr_seq.device)], dim=1)
        else:
            rr_padded = rr_seq[:, :self.rr_target_len]
            mask = mask[:, :self.rr_target_len]

        # 2-3) mask 처리: invalid 위치(=False) → 0으로
        rr_padded = rr_padded.masked_fill(~mask, 0.0)

        # 2-4) FC → Attn
        r = self.fc_rr(rr_padded)         # (B, rr_dim)
        # AttentionLayer 가 mask 지원하면 넘겨주세요. 없으면, 이미 invalid 값 0으로 처리된 상태로!
        r = self.attn_rr(r.unsqueeze(1))  # (B, rr_dim)

        # ── 3) float path (RMSSD, SDNN) ────────────────────────────────
        # nan → -100 처리된 상태, 그대로 FC 에 투입해도 되지만
        # 스케일 차이가 크면 작은 계수 곱하거나 embedding 사용 권장
        f = self.fc_float(torch.stack([rmssd, sdnn], dim=1))  # (B, float_dim)

        # ── 4) fuse & project ─────────────────────────────────────────
        out = torch.cat([x, r, f], dim=1)   # (B, fused_dim)
        out = self.dropout(out)
        return self.proj(out)              # (B, embed_dim)
