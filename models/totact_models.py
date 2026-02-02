import torch
import torch.nn as nn

from models.encoder.ppg_TCN_encoder import PPGEncoder
from models.encoder.imu_encoder import IMUFeatureEncoder
from models.encoder.sc_encoder import ScenarioEmbedding
from models.encoder.survey_encoder import PreSurveyEncoder
from models.encoder.veh_encoder import VehicleTCNEncoder
from models.layers.attnpool import AttnPool1D
import torch.nn.functional as F

import traceback

# =========================
# Utils
# =========================
class _Proj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, max(out_dim, 32)),
            nn.ReLU(),
            nn.Linear(max(out_dim, 32), out_dim),
        )
    def forward(self, x):  # x: (B,T,in_dim) or (B,in_dim)
        if x.dim() == 3:
            B,T,_ = x.shape
            return self.net(x.view(B*T, -1)).view(B, T, -1)
        return self.net(x)

def _pretty(x):
    if x is None:
        return "None"
    if torch.is_tensor(x):
        shape = tuple(x.shape)
        dtype = x.dtype
        nz = float((x != 0).float().mean().item()) if x.numel() else 0.0

        if x.is_floating_point():
            mean_abs = x.abs().mean().item() if x.numel() else 0.0
            return f"shape={shape}, dtype={dtype}, |mean|={mean_abs:.6f}, nonzero%={100*nz:.2f}"
        else:
            # 정수 텐서는 통계 출력만 안전하게
            xf = x.to(torch.float32) if x.numel() else x
            mean_abs = xf.abs().mean().item() if x.numel() else 0.0
            minv = int(x.min().item()) if x.numel() else 0
            maxv = int(x.max().item()) if x.numel() else 0
            # unique 개수도 유용할 수 있음
            try:
                uniq = int(x.unique().numel()) if x.numel() and x.numel() <= 200000 else -1
                uinfo = f", unique={uniq}" if uniq >= 0 else ""
            except Exception:
                uinfo = ""
            return (f"shape={shape}, dtype={dtype}, min={minv}, max={maxv}, "
                    f"|mean|~{mean_abs:.6f}, nonzero%={100*nz:.2f}{uinfo}")
    return f"type={type(x)}"

def _all_ignored(x):
    # long/int 텐서에 -100만 가득한 경우(또는 None) → 무시
    return (x is None) or (torch.is_tensor(x) and x.dtype in (torch.long, torch.int64)
                           and x.numel() > 0 and (x == -100).all())

def _infer_dims_from_cfg(cfg):
    veh_out = int(cfg.Encoders.veh.get('embed_dim', 64))
    sc_out  = int(cfg.Encoders.sc.get('embed_dim', 16))
    imu_out = int(cfg.Encoders.imu.get('encoder_dim', 128))
    # PPG는 TCN 임베드 + HRV 6개(평균/표준편차/최소/최대/RMSSD/SDNN)
    ppg_out = int(cfg.Encoders.ppg.get('embed_dim', 64)) + 6
    srv_out = int(cfg.Encoders.survey.get('embed_dim', 8))
    return veh_out, sc_out, imu_out, ppg_out, srv_out

# 두 클래스(TOT/ACT) 공통으로 쓸 유틸
def _ppg_hrv_from_batch(batch, device):
    # batch에 'ppg_rr','ppg_rmssd','ppg_sdnn'이 들어온다는 가정(없으면 0 대체)
    zeros = lambda B: torch.zeros(B, 1, device=device)
    B = batch['ppg'].shape[0] if 'ppg' in batch else (batch['veh'].shape[0])
    rr = batch.get('ppg_rr', None)
    parts = []
    if rr is not None and rr.dim() > 1 and rr.size(1) > 0:
        parts.append(rr.mean(dim=1, keepdim=True))
        parts.append(torch.nan_to_num(rr.std(dim=1, keepdim=True), nan=0.0))
        parts.append(rr.min(dim=1, keepdim=True).values)
        parts.append(rr.max(dim=1, keepdim=True).values)
    else:
        parts += [zeros(B), zeros(B), zeros(B), zeros(B)]
    rmssd = batch.get('ppg_rmssd', None)
    sdnn  = batch.get('ppg_sdnn',  None)
    parts.append(torch.nan_to_num(rmssd.to(device).unsqueeze(1), nan=0.0) if rmssd is not None else zeros(B))
    parts.append(torch.nan_to_num(sdnn.to(device).unsqueeze(1),  nan=0.0) if sdnn  is not None else zeros(B))
    return torch.cat(parts, dim=1)  # (B,6)

def _auto_hidden(in_fused: int, task: str) -> int:
    """
    in_fused = feat_dim * (#enabled modalities)
    간단한 휴리스틱: TOT은 가볍게, ACT는 조금 크게.
    """
    if task == "tot":
        if in_fused <= 128: return 128
        if in_fused <= 192: return 160
        if in_fused <= 256: return 192
        if in_fused <= 320: return 224
        return 256
    else:  # act
        if in_fused <= 128: return 192
        if in_fused <= 192: return 224
        if in_fused <= 256: return 256
        if in_fused <= 320: return 256
        return 256
    
def _time_pool(x, mode: str = "mean"):
    """x: (B,C) 또는 (B,T,C) -> (B,C)"""
    if x is None:
        return None
    if x.dim() == 3:
        if mode == "max":
            return x.max(dim=1).values
        elif mode == "last":
            return x[:, -1]
        else:
            return x.mean(dim=1)
    return x  # already (B,C)

def _to_prob(x, use_prob: bool = True):
    if x is None:
        return None
    return F.softmax(x, dim=-1) if use_prob else x

# -----------------------------
# 잔차형 스택 헤드: y = base + gate * f([base, mot, emo])
# -----------------------------
class ResidualStackHead(nn.Module):
    def __init__(self, num_classes: int, gate_init: float = 0.1, norm_each: bool = True, use_prob: bool = True):
        super().__init__()
        C = num_classes
        self.use_prob = use_prob
        self.norm_each = norm_each
        if norm_each:
            self.norm_base = nn.LayerNorm(C)
            self.norm_mot  = nn.LayerNorm(C)
            self.norm_emo  = nn.LayerNorm(C)

        self.fuse = nn.Linear(3 * C, C)
        # ★ 바꿔주기: 완전 0 대신 아주 작은 Xavier
        nn.init.xavier_uniform_(self.fuse.weight, gain=1e-2)
        nn.init.zeros_(self.fuse.bias)

        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, base_logits, mot_2d, emo_2d):
        mot = _to_prob(mot_2d, self.use_prob)
        emo = _to_prob(emo_2d, self.use_prob)
        base_n = self.norm_base(base_logits) if self.norm_each else base_logits
        mot_n  = self.norm_mot(mot)          if self.norm_each else mot
        emo_n  = self.norm_emo(emo)          if self.norm_each else emo
        x = torch.cat([base_n, mot_n, emo_n], dim=-1)
        delta = self.fuse(x)
        return base_logits + self.gate * delta

    
# =========================
# TOT Baseline (Ablation)
# =========================
class TOT_Baseline(nn.Module):
    """
    선택된 모달리티들을 late-fuse → GRU(+AttnPool) → classifier(3-way)
    - 시간축 길이는 veh 기준(T)
    - cfg.TOT.use_modalities: ['veh','imu','sc','ppg','survey'] 중 일부
    - cfg.TOT.feat_dim: 각 모달을 동일 차원으로 투영
    - cfg.TOT.attn_pool: True/False (AttnPool 사용 여부)
    - cfg.TOT.num_classes: 기본 3
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use = set(cfg.TOT.use_modalities)
        self.num_classes = int(cfg.TOT.num_classes)
        D = int(cfg.TOT.feat_dim)

        # --- 인코더 내부 생성 ---
        self.veh_encoder = VehicleTCNEncoder(cfg.Encoders.veh)       # (B,C,T)->(B,Dveh,T)
        self.sc_encoder  = ScenarioEmbedding(cfg.Encoders.sc)        # (B,T,Dsc), T 인자 필요
        self.imu_encoder = IMUFeatureEncoder(cfg.Encoders.imu) if ('imu' in self.use) else None
        self.ppg_encoder = PPGEncoder(cfg.Encoders.ppg)              if ('ppg' in self.use) else None
        self.survey_mlp  = PreSurveyEncoder(cfg.Encoders.survey)     if ('survey' in self.use) else None

        # --- 출력 차원 추론 → 공통 feat_dim 프로젝션 ---
        veh_out, sc_out, imu_out, ppg_out, srv_out = _infer_dims_from_cfg(cfg)
        self.proj_veh = _Proj(in_dim=veh_out, out_dim=D)
        self.proj_sc  = _Proj(in_dim=sc_out,  out_dim=D)
        self.proj_imu = _Proj(in_dim=imu_out, out_dim=D) if self.imu_encoder else None
        self.proj_ppg = _Proj(in_dim=ppg_out, out_dim=D) if self.ppg_encoder else None
        self.proj_srv = _Proj(in_dim=srv_out, out_dim=D) if self.survey_mlp  else None

        # --- late-fuse 입력 차원 계산 ---
        in_fused = 0
        if 'veh'    in self.use: in_fused += D
        if 'sc'     in self.use: in_fused += D
        if 'imu'    in self.use and self.proj_imu: in_fused += D
        if 'ppg'    in self.use and self.proj_ppg: in_fused += D
        if 'survey' in self.use and self.proj_srv: in_fused += D
        if in_fused <= 0:
            raise ValueError("use_modalities가 비어있거나 모두 비활성입니다. 최소 1개 이상 켜주세요.")

        # hidden 자동/수동 결정
        if getattr(cfg.TOT, "hidden", None) is None:
            H = _auto_hidden(in_fused, task="tot")
        else:
            H = int(cfg.TOT.hidden)

        # GRU 레이어/드롭아웃 (경고 제거)
        num_layers = int(getattr(cfg.TOT, "gru_layers", 1))
        dropout = 0.2 if num_layers == 1 else 0.2

        self.gru  = nn.GRU(in_fused, H, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.pool = AttnPool1D(H) if bool(getattr(cfg.TOT, "attn_pool", False)) else None
        self.classifier = nn.Sequential(
            nn.Linear(H, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, self.num_classes)
        )

        print(f"[TOT] using modalities: {sorted(list(self.use))} (feat_dim={D}, in_fused={in_fused}, hidden={H}, attn_pool={bool(getattr(cfg.TOT,'attn_pool',False))})")


    def _zeros(self, B, T, D, device):  # (B,T,D)
        return torch.zeros(B, T, D, device=device)

    def _ensure_time_major(self, x):
        # (B,Ch,T) -> (B,T,Ch), (B,T,Ch)면 그대로
        if x.dim() == 3 and x.shape[1] < x.shape[2]:
            return x.permute(0, 2, 1).contiguous()
        return x

    def encode(self, batch):
        device = next(self.parameters()).device
        assert "veh" in batch, "veh 텐서는 시간축 길이 T를 결정하는 데 필수입니다."

        # veh로 시간축 길이 결정
        veh_feat_tcn = self.veh_encoder(batch["veh"].permute(0,2,1))   # (B,Dveh,T)
        veh_feat = self._ensure_time_major(veh_feat_tcn)               # (B,T,Dveh)
        B, T = veh_feat.shape[0], veh_feat.shape[1]
        veh_feat = self.proj_veh(veh_feat)                             # (B,T,D)

        feats = []
        if 'veh' in self.use:
            feats.append(veh_feat)

        if 'sc' in self.use:
            sc_feat = self.sc_encoder(
                batch["sc_evt"].to(device),
                batch["sc_type"].to(device),
                batch["sc_phase"].to(device),
                batch["sc_time"].to(device),
                T=T
            )                               # (B,T,Dsc_raw)
            feats.append(self.proj_sc(sc_feat))  # (B,T,D)

        if 'imu' in self.use and self.imu_encoder:
            if "imu" in batch:
                imu_feat = self.imu_encoder(batch["imu"])              # (B,T,Dim) or (B,Dim,T)
                imu_feat = self._ensure_time_major(imu_feat)
                feats.append(self.proj_imu(imu_feat))                  # (B,T,D)
            else:
                feats.append(self._zeros(B, T, self.cfg.TOT.feat_dim, device))

        if 'ppg' in self.use and self.ppg_encoder:
            if "ppg" in batch:
                tcn_out = self.ppg_encoder(batch["ppg"])               # (B, embed_dim) 정적/pooled
                hrv6    = _ppg_hrv_from_batch(batch, device)           # (B,6)
                ppg_cat = torch.cat([tcn_out, hrv6], dim=1)            # (B, embed+6)
                ppg_feat = self.proj_ppg(ppg_cat)                      # (B,D)
                ppg_feat = ppg_feat.unsqueeze(1).expand(-1, T, -1)     # (B,T,D)
            else:
                ppg_feat = self._zeros(B, T, self.cfg.TOT.feat_dim, device)
            feats.append(ppg_feat)

        if 'survey' in self.use and self.survey_mlp:
            if "survey" in batch:
                srv = self.survey_mlp(batch["survey"])                 # (B,Dsrv)
                srv = self.proj_srv(srv).unsqueeze(1).repeat(1, T, 1)  # (B,T,D)
                feats.append(srv)
            else:
                feats.append(self._zeros(B, T, self.cfg.TOT.feat_dim, device))

        fused = torch.cat(feats, dim=-1)   # (B,T,in_fused)
        h, _  = self.gru(fused)            # (B,T,128)
        if self.pool:
            pooled, _ = self.pool(h)       # (B,128)
        else:
            pooled = h[:, -1, :]           # (B,128)
        return pooled

    def forward(self, batch):
        x = self.encode(batch)              # (B,128)
        logits = self.classifier(x)         # (B,num_classes)
        return {"logits": logits, "feat": x}


# =========================
# ACT Baseline (Ablation)
# =========================
class ACT_Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use = set(cfg.ACT.use_modalities)
        D = int(cfg.ACT.feat_dim)  # ← H보다 먼저 D만 읽어둡니다. H는 나중에 결정!

        self.veh_encoder = VehicleTCNEncoder(cfg.Encoders.veh)
        self.sc_encoder  = ScenarioEmbedding(cfg.Encoders.sc)
        self.imu_encoder = IMUFeatureEncoder(cfg.Encoders.imu) if ('imu' in self.use) else None
        self.ppg_encoder = PPGEncoder(cfg.Encoders.ppg)              if ('ppg' in self.use) else None
        self.survey_mlp  = PreSurveyEncoder(cfg.Encoders.survey)     if ('survey' in self.use) else None

        veh_out, sc_out, imu_out, ppg_out, srv_out = _infer_dims_from_cfg(cfg)
        self.proj_veh = _Proj(in_dim=veh_out, out_dim=D)
        self.proj_sc  = _Proj(in_dim=sc_out,  out_dim=D)
        self.proj_imu = _Proj(in_dim=imu_out, out_dim=D) if self.imu_encoder else None
        self.proj_ppg = _Proj(in_dim=ppg_out, out_dim=D) if self.ppg_encoder else None
        self.proj_srv = _Proj(in_dim=srv_out, out_dim=D) if self.survey_mlp  else None

        # --- late-fuse 입력 차원 계산 ---
        in_fused = 0
        if 'veh'    in self.use: in_fused += D
        if 'sc'     in self.use: in_fused += D
        if 'imu'    in self.use and self.proj_imu: in_fused += D
        if 'ppg'    in self.use and self.proj_ppg: in_fused += D
        if 'survey' in self.use and self.proj_srv: in_fused += D
        if in_fused <= 0:
            raise ValueError("ACT.use_modalities가 비어있습니다. 최소 1개 이상 켜세요.")

        # --- 여기서 H를 결정(자동/수동) ---
        if getattr(cfg.ACT, "hidden", None) is None:
            H = _auto_hidden(in_fused, task="act")
        else:
            H = int(cfg.ACT.hidden)

        # GRU 레이어/드롭아웃 (경고 제거)
        num_layers = int(getattr(cfg.ACT, "gru_layers", 1))
        dropout = 0.2 if num_layers == 1 else 0.2

        self.gru = nn.GRU(in_fused, H, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.regressor = nn.Sequential(
            nn.Linear(H, H//2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(H//2, 1)
        )

        print(f"[ACT] using modalities: {sorted(list(self.use))} (feat_dim={D}, in_fused={in_fused}, hidden={H})")


    def _zeros(self, B, T, D, device):
        return torch.zeros(B, T, D, device=device)

    def _ensure_time_major(self, x):
        if x.dim() == 3 and x.shape[1] < x.shape[2]:
            return x.permute(0,2,1).contiguous()
        return x

    def forward(self, batch):
        device = next(self.parameters()).device
        assert "veh" in batch, "veh 텐서는 시간축 길이 T를 결정하는 데 필수입니다."

        veh_feat_tcn = self.veh_encoder(batch["veh"].permute(0,2,1))
        veh_feat = self._ensure_time_major(veh_feat_tcn)  # (B,T,Dveh)
        B, T = veh_feat.shape[0], veh_feat.shape[1]
        veh_feat = self.proj_veh(veh_feat)

        feats = []
        if 'veh' in self.use:
            feats.append(veh_feat)

        if 'sc' in self.use:
            sc_feat = self.sc_encoder(
                batch["sc_evt"].to(device),
                batch["sc_type"].to(device),
                batch["sc_phase"].to(device),
                batch["sc_time"].to(device),
                T=T
            )
            feats.append(self.proj_sc(sc_feat))

        if 'imu' in self.use and self.imu_encoder:
            if "imu" in batch:
                imu_feat = self.imu_encoder(batch["imu"])
                imu_feat = self._ensure_time_major(imu_feat)
                feats.append(self.proj_imu(imu_feat))
            else:
                feats.append(self._zeros(B, T, self.cfg.ACT.feat_dim, device))

        if 'ppg' in self.use and self.ppg_encoder:
            if "ppg" in batch:
                tcn_out = self.ppg_encoder(batch["ppg"])              # (B, embed_dim)
                hrv6    = _ppg_hrv_from_batch(batch, device)          # (B,6)
                ppg_cat = torch.cat([tcn_out, hrv6], dim=1)           # (B, embed+6)
                ppg_feat = self.proj_ppg(ppg_cat)                     # (B, D)
                ppg_feat = ppg_feat.unsqueeze(1).expand(-1, T, -1)    # (B, T, D)
            else:
                ppg_feat = self._zeros(B, T, self.cfg.ACT.feat_dim, device)
            feats.append(ppg_feat)

        if 'survey' in self.use and self.survey_mlp:
            if "survey" in batch:
                srv = self.survey_mlp(batch["survey"])                # (B,Dsrv)
                srv = self.proj_srv(srv).unsqueeze(1).repeat(1, T, 1) # (B,T,D)
                feats.append(srv)
            else:
                feats.append(self._zeros(B, T, self.cfg.ACT.feat_dim, device))

        fused = torch.cat(feats, dim=-1)  # (B,T,in_fused)
        h, _  = self.gru(fused)           # (B,T,H)
        y     = self.regressor(h)         # (B,T,1)
        return {"act_preds": y, "feat": h}
    
class StackHead(nn.Module):
    def __init__(self, cfg, in_dim_mot, in_dim_emo, num_classes):
        super().__init__()
        hid = getattr(cfg.TOT, "enh_hid", 128)
        self.proj_m = nn.Linear(in_dim_mot, hid)
        self.proj_e = nn.Linear(in_dim_emo, hid)
        self.fuser  = nn.Sequential(nn.ReLU(), nn.Linear(2*hid, num_classes))
        # 게이트와 스케일을 분리 (게이트는 [0,1], 스케일은 양수)
        self.gate   = nn.Parameter(torch.tensor(0.1))          # gating
        self.log_s  = nn.Parameter(torch.tensor(0.0))          # scale s = exp(log_s)
        self.residual_gain = nn.Parameter(torch.tensor(1.0))  # 학습 가능 스칼라

    def make_residual(self, mot, emo, base_logits):
        # 모양 맞추기 (시간축 존재/부재 모두 지원)
        def pool_if_needed(x, ref):
            return x if x.dim()==ref.dim() else x.unsqueeze(1).expand(-1, ref.size(1), -1)
        m = pool_if_needed(mot, base_logits)
        e = pool_if_needed(emo, base_logits)
        h = torch.cat([self.proj_m(m), self.proj_e(e)], dim=-1)
        res = self.fuser(h)
        return res

    def gated(self, residual):
        s = self.log_s.exp()            # 양수 스케일
        g = torch.sigmoid(self.gate)    # 0~1
        return (g * s) * residual
    
# totact_models.py
class EnhancedTOTModel(nn.Module):
    def __init__(self, cfg, pretrained_tot_baseline: nn.Module, pretrained_fusion_model: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.baseline = pretrained_tot_baseline.eval()
        for p in self.baseline.parameters():
            p.requires_grad = False

        self.fusion = pretrained_fusion_model.eval()
        for p in self.fusion.parameters():
            p.requires_grad = False

        # num_classes는 cfg 경로에 맞춰 가져오세요.
        self.num_classes = getattr(getattr(cfg, "TOT", cfg), "num_classes", getattr(cfg, "num_classes", 3))

        # 하이퍼(없으면 기본값 사용)
        TCFG = getattr(cfg, "TOT", cfg)
        self.enh_use_residual = getattr(TCFG, "enh_use_residual", True)
        self.time_pooling = getattr(TCFG, "enh_time_pool", "mean")
        self.fuse_mode    = getattr(TCFG, "enh_fuse_mode", "logit_add")
        
        use_prob              = getattr(TCFG, "enh_use_prob", True)
        norm_each             = getattr(TCFG, "enh_norm_each", True)
        gate_init             = getattr(TCFG, "enh_gate_init", 0.10)

        # 스택 헤드
        if self.enh_use_residual:
            self.stack_head = ResidualStackHead(
                num_classes=self.num_classes,
                gate_init=gate_init,
                norm_each=norm_each,
                use_prob=use_prob,
            )
        else:
            # (baseline, mot, emo) concat -> Linear
            self.stack_head = nn.Linear(3 * self.num_classes, self.num_classes)

        # 컨텍스트 소스 키
        self.ctx_source = getattr(TCFG, "enh_ctx_source", "prob")  # "prob"|"embed" (현재 코드는 logits/prob 가정)


    @torch.no_grad()
    def _baseline_logits(self, batch):
        out = self.baseline(batch)
        return out["logits"] if isinstance(out, dict) else out  # (B,C)

    @torch.no_grad()
    def _get_ctx_logits(self, batch):
        fb = dict()  # Fusion에 넘길 새 배치

        B = None
        # 배치에서 B 추출
        for v in batch.values():
            if torch.is_tensor(v):
                B = v.shape[0]; break
        if B is None: return None, None

        device = self.cfg.Project.device

        # 1) 시나리오: TOT → Fusion emotion 경로 & motion 경로 둘 다 커버
        sc_evt   = batch.get('sc_evt',   None)
        sc_type  = batch.get('sc_type',  None)
        sc_phase = batch.get('sc_phase', None)
        sc_time  = batch.get('sc_time',  None)

        # emotion 경로에서 기대하는 키
        if not _all_ignored(sc_evt):   fb['scenario_evt_e']   = sc_evt.to(torch.long)
        if not _all_ignored(sc_type):  fb['scenario_type_e']  = sc_type.to(torch.long)
        if not _all_ignored(sc_phase): fb['phase_evt_e']      = sc_phase.to(torch.long)
        if sc_time is not None:        fb['scenario_time_e']  = sc_time.float()

        # motion 경로에서 기대하는 키(있을 때만)
        if not _all_ignored(sc_evt):   fb['sc_motion_evt']   = sc_evt.to(torch.long)
        if not _all_ignored(sc_type):  fb['sc_motion_type']  = sc_type.to(torch.long)
        if not _all_ignored(sc_phase): fb['sc_motion_phase'] = sc_phase.to(torch.long)
        if sc_time is not None:        fb['sc_motion_time']  = sc_time.float()

        # 2) veh/imu/ppg/survey 매핑
        if 'veh' in batch:
            fb['veh_motion']  = batch['veh']            # 모션 경로
            fb['veh_emotion'] = batch['veh']            # 감정 경로에도 동일 소스 사용
        if 'imu' in batch:
            fb['imu_motion']  = batch['imu']
        # ppg는 Fusion이 HRV 보조특징도 기대함. 없으면 0으로 채워서 안전하게 패스
        if 'ppg' in batch:
            fb['ppg_emotion']       = batch['ppg']
            fb['ppg_rr_emotion']    = batch.get('ppg_rr_emotion',    torch.zeros(B, 0, device=device))
            fb['ppg_rmssd_emotion'] = batch.get('ppg_rmssd_emotion', torch.zeros(B,   device=device))
            fb['ppg_sdnn_emotion']  = batch.get('ppg_sdnn_emotion',  torch.zeros(B,   device=device))
        if 'survey_e' in batch:
            fb['survey_e'] = batch['survey_e']

        # 3) 모션 길이 강제(시퀀스-투-원): 불필요한 expand 폭주 방지
        fb['label_motion'] = torch.ones(B, 1, dtype=torch.long, device=device)

        # 4) 실제로 배치에 "존재하는" 모달만 사용하도록 일시적으로 목록 축소
        orig_mmods = getattr(self.fusion, 'motion_modalities', [])
        orig_emods = getattr(self.fusion, 'emotion_modalities', [])

        # 각 경로에서 feature가 생성될 수 있는지 판단할 최소키 조건
        can_motion = {
            'imu':   ('imu_motion' in fb),
            'veh':   ('veh_motion' in fb),
            'sc':    all(k in fb for k in ['sc_motion_evt','sc_motion_type','sc_motion_phase','sc_motion_time']),
            'ppg':   ('ppg_emotion' in fb),
            'survey':('survey_e' in fb),
        }
        can_emotion = {
            'ppg':   ('ppg_emotion' in fb),
            'veh':   ('veh_emotion' in fb),
            'sc':    all(k in fb for k in ['scenario_evt_e','scenario_type_e','phase_evt_e','scenario_time_e']),
            'imu':   ('imu_motion' in fb),      # 감정 경로에서 imu를 평균 pooling해서 씀
            'survey':('survey_e' in fb),
        }

        self.fusion.motion_modalities  = [m for m in orig_mmods if can_motion.get(m, False)]
        self.fusion.emotion_modalities = [m for m in orig_emods if can_emotion.get(m, False)]

        # 5) 호출 (안전장치 포함)
        try:
            fout = self.fusion(fb)
        except Exception as e:
            print("[DBG][fusion call failed in enhancer]:", repr(e))
            # 원래 목록 복원하고 안전하게 리턴
            self.fusion.motion_modalities  = orig_mmods
            self.fusion.emotion_modalities = orig_emods
            return None, None

        # 목록 복원
        self.fusion.motion_modalities  = orig_mmods
        self.fusion.emotion_modalities = orig_emods

        # 6) 컨텍스트 로짓/확률 추출
        mot = fout.get('motion_logits', None)
        # 감정은 valence/arousal 따로라서, TOT 3-분류 컨텍스트로 쓰려면 9점척도를 3-bin으로 압축
        emo_logits_v = fout.get('valence_logits', None)
        emo_logits_a = fout.get('arousal_logits', None)

        def _val9_to_val3(logits9):
            # (B,9) -> (B,3): [0-2]=low, [3-5]=mid, [6-8]=high, 로짓을 합산
            if logits9 is None: return None
            B,C = logits9.shape
            if C == 3: return logits9
            if C != 9: return None
            low  = logits9[:, 0:3].logsumexp(dim=1, keepdim=True)
            mid  = logits9[:, 3:6].logsumexp(dim=1, keepdim=True)
            high = logits9[:, 6:9].logsumexp(dim=1, keepdim=True)
            return torch.cat([low, mid, high], dim=1)

        emo = _val9_to_val3(emo_logits_v)  # valence만 컨텍스트로 사용(원하면 a도 mix 가능)

        return mot, emo
    
    def forward(self, batch):
        base_logits = self._baseline_logits(batch)
        mot, emo = self._get_ctx_logits(batch)

        def _prep_ctx(x):
            if x is None: return torch.zeros_like(base_logits)
            if x.dim()==3 and base_logits.dim()==2:
                x = _time_pool(x, self.time_pooling)
            if x.dim()==2 and base_logits.dim()==3:
                x = x.unsqueeze(1).expand(-1, base_logits.size(1), -1)
            return x

        mot = _prep_ctx(mot); emo = _prep_ctx(emo)
        logits = self.stack_head(base_logits, mot, emo)

        # 첫 배치 한정 요약 로그
        if not hasattr(self, "_did_log"):
            self._did_log = True
            def stat(t): return f"shape={tuple(t.shape)}, |mean|={t.abs().mean().item():.6f}"
            print("[DBG][fwd] base:", stat(base_logits))
            print("[DBG][fwd] mot :", stat(mot))
            print("[DBG][fwd] emo :", stat(emo))
            print("[DBG][fwd] res :", stat(logits - base_logits))
        return {"logits": logits, "base_logits": base_logits,
                "fuse_residual": logits - base_logits, "mot_ctx": mot, "emo_ctx": emo}


class EnhancedACTModel(nn.Module):
    """
    베이스라인(ACT) 동결 + 모션/감정 컨텍스트로 '잔차'를 예측해 보정.
    - baseline 예측 y_base(B,T,1)에 delta(B,T,1)를 더하는 residual 설계
    - 파라미터/용량 작게 유지 → 과적합 억제
    """
    def __init__(self, cfg, pretrained_act_baseline, pretrained_fusion_model):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.Project.device
        self.ctx_dim = int(cfg.FusionModel.hidden_dim)

        # --- 동결 백본 ---
        self.baseline = pretrained_act_baseline.eval()
        self.fusion   = pretrained_fusion_model.eval()
        for p in self.baseline.parameters(): p.requires_grad = False
        for p in self.fusion.parameters():   p.requires_grad = False

        # --- 잔차 헤드 ---
        # 입력: [baseline_feat(H_b), mot_ctx(D), emo_ctx(D)]  → 작은 GRU/MLP
        # baseline의 히든 차원 추정(프린트 로그를 그대로 따름)
        self.Hb = getattr(cfg.ACT, "hidden", 128) or 128
        in_dim = self.Hb + 2*self.ctx_dim

        # 아주 작게: 1층 GRU 없이 MLP로도 충분 (시퀀스 처리 위해 시점별 선형)
        self.delta_head = nn.Sequential(
            nn.Linear(in_dim, max(64, in_dim//4)),
            nn.ReLU(),
            nn.Dropout(getattr(cfg.EnhancerTask, "dropout", 0.1)),
            nn.Linear(max(64, in_dim//4), 1)
        )

    def _fusion_ready(self, batch) -> bool:
        need_any = ("imu" in batch) or ("ppg" in batch) or ("veh" in batch)
        has_sc = all(k in batch for k in ("sc_evt","sc_type","sc_phase","sc_time"))
        return need_any and has_sc

    @torch.no_grad()
    def _get_ctx_embed_seq(self, batch, T, B):
        D = self.ctx_dim; dev = self.device
        if not self._fusion_ready(batch):
            z = torch.zeros(B, T, D, device=dev)
            return z, z
        try:
            out = self.fusion(batch)
            mot = (out.get("fused_motion") or out.get("motion_ctx") or out.get("motion"))
            emo = (out.get("fused_emotion") or out.get("emotion_ctx") or out.get("emotion"))
            if mot is not None and mot.dim()==3: mot = mot.mean(1)
            if emo is not None and emo.dim()==3: emo = emo.mean(1)
            if mot is None: mot = torch.zeros(B, D, device=dev)
            if emo is None: emo = torch.zeros(B, D, device=dev)
            # 시퀀스 길이로 확장
            mot = mot.unsqueeze(1).expand(-1, T, -1)  # (B,T,D)
            emo = emo.unsqueeze(1).expand(-1, T, -1)  # (B,T,D)
            return mot, emo
        except Exception:
            z = torch.zeros(B, T, D, device=dev)
            return z, z

    def forward(self, batch):
        dev = self.device

        # 1) 베이스라인 예측/피처 (동결)
        with torch.no_grad():
            b_out = self.baseline(batch)                  # {"act_preds": (B,T,1), "feat": (B,T,Hb)}
            y_base = b_out["act_preds"] if isinstance(b_out, dict) else b_out
            h_base = b_out.get("feat", None)
            if h_base is None:
                # 백업: veh/sc 재인코딩 (비상용)
                veh_feat_tcn = self.baseline.veh_encoder(batch["veh"].permute(0,2,1))
                veh_feat = veh_feat_tcn.permute(0,2,1)
                B,T = veh_feat.shape[0], veh_feat.shape[1]
                sc_feat = self.baseline.sc_encoder(
                    batch["sc_evt"].to(dev), batch["sc_type"].to(dev),
                    batch["sc_phase"].to(dev), batch["sc_time"].to(dev), T=T
                )
                fused = torch.cat([veh_feat, sc_feat], dim=-1)
                # GRU 은닉을 직접 끌지 못할 때는 선형으로 투영
                if not hasattr(self, "_proj_backup"):
                    self._proj_backup = nn.Linear(fused.size(-1), self.Hb).to(dev)
                h_base = self._proj_backup(fused)

        B, T = y_base.shape[0], y_base.shape[1]

        # 2) 모션/감정 임베딩 → (B,T,D)
        mot_ctx, emo_ctx = self._get_ctx_embed_seq(batch, T, B)

        # 3) 잔차(delta) 예측
        x = torch.cat([h_base, mot_ctx, emo_ctx], dim=-1)  # (B,T,Hb+2D)
        delta = self.delta_head(x)                         # (B,T,1)

        return y_base + delta   # residual: 최종 예측
