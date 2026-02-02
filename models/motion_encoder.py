import torch
import torch.nn as nn
from types import SimpleNamespace

from models.encoder.ppg_TCN_encoder import PPGEncoder
from models.encoder.imu_encoder import IMUFeatureEncoder
from models.encoder.sc_encoder import ScenarioEmbedding
from models.encoder.survey_encoder import PreSurveyEncoder
from models.encoder.veh_encoder import VehicleTCNEncoder
    
class MotionEncoder(nn.Module): 
    """
    모든 모달리티 Concat하여 사용
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        hidden_dim = cfg.PretrainMotion.hidden_dim
        self.modalities_to_use = set(cfg.PretrainMotion.modalities_to_use)
        print(f"Pre-train Motion with: {self.modalities_to_use}, Concat")
        # --- 모든 인코더와 프로젝션 레이어를 여기서 명확하게 정의 ---
        if 'imu' in self.modalities_to_use:
            self.imu = IMUFeatureEncoder(cfg.Encoders.imu)
            self.p_imu = nn.Linear(cfg.Encoders.imu['encoder_dim'], hidden_dim)
        if 'ppg' in self.modalities_to_use:
            self.ppg = PPGEncoder(cfg.Encoders.ppg)
            ppg_in_dim = cfg.Encoders.ppg['embed_dim'] + 6 
            self.p_ppg = nn.Linear(ppg_in_dim, hidden_dim)
        if 'veh' in self.modalities_to_use:
            self.veh = VehicleTCNEncoder(cfg.Encoders.veh)
            self.p_veh = nn.Linear(cfg.Encoders.veh['embed_dim'], hidden_dim)
        if 'sc' in self.modalities_to_use:
            self.sc = ScenarioEmbedding(cfg.Encoders.sc)
            self.p_sc = nn.Linear(cfg.Encoders.sc['embed_dim'],hidden_dim)
        # Survey
        if 'survey' in self.modalities_to_use:
            self.survey = PreSurveyEncoder(cfg.Encoders.survey)
            self.p_survey = nn.Linear(cfg.Encoders.survey['embed_dim'], hidden_dim)

        # ——— 3) Fusion Conv 레이어 ———
        # in_channels = hidden_dim × (활성화된 모달리티 개수)
        in_ch = hidden_dim * len(self.modalities_to_use)
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=hidden_dim, kernel_size=1),
            nn.BatchNorm1d(num_features=hidden_dim),nn.ReLU())

    def _process_hrv(self, batch, device):
        ppg_rr = batch['ppg_rr'].to(device)
        
        hrv_list = []
        if ppg_rr.dim() > 1 and ppg_rr.shape[1] > 0:
            hrv_list.append(ppg_rr.mean(dim=1, keepdim=True))
            std_dev = torch.std(ppg_rr, dim=1, keepdim=True)
            hrv_list.append(torch.nan_to_num(std_dev, nan=0.0))
            hrv_list.append(torch.min(ppg_rr, dim=1, keepdim=True).values)
            hrv_list.append(torch.max(ppg_rr, dim=1, keepdim=True).values)
        else: # 데이터가 없는 경우 0으로 채움
            hrv_list.extend([torch.zeros(ppg_rr.shape[0], 1, device=device)] * 4)

        for key in ['ppg_rmssd', 'ppg_sdnn']:
            # 다른 HRV 특징들도 nan이 될 가능성이 있다면 동일하게 처리해줄 수 있습니다.
            hrv_list.append(torch.nan_to_num(batch[key].to(device).unsqueeze(1), nan=0.0))
        
        return torch.cat(hrv_list, dim=1)
    
    def forward(self, **kwargs):
        parts = []
        T = None
        device = self.cfg.Project.device

        # IMU
        if 'imu' in self.modalities_to_use and 'imu_seq' in kwargs:
            out = self.imu(kwargs['imu_seq'], kwargs['imu_len'])    # (B,T,enc_dim)
            h   = self.p_imu(out)                                  # (B,T,H)
            parts.append(h)
            T = h.size(1)

        # PPG
        if 'ppg' in self.modalities_to_use and 'ppg_seq' in kwargs:
            tcn_out = self.ppg(kwargs['ppg_seq'])  # (B, tcn_embed_dim)
            hrv_out = self._process_hrv(kwargs, device) # (B, 6)
            combined = torch.cat([tcn_out, hrv_out], dim=1) # (B, tcn_embed_dim + 6)
            h = self.p_ppg(combined) # (B, H)
            if T is None and 'imu_seq' not in kwargs:
                raise ValueError("Cannot determine sequence length T without a temporal sequence like IMU.")
            h = h.unsqueeze(1).expand(-1, T, -1) # (B, T, H)
            parts.append(h)
            
        # VEH
        if 'veh' in self.modalities_to_use and 'veh_seq' in kwargs:
            # 1. VehicleTCNEncoder로 64차원 특징 추출
            pooled = self.veh(kwargs['veh_seq'], return_pooled=True)  # (B, 64)
            h = self.p_veh(pooled) # (B, 128) 차원으로 투영
            
            # 2. 시간 축으로 확장
            h = h.unsqueeze(1).expand(-1, T, -1) # (B, T, 128)
            parts.append(h)

        # Scenario
        if 'sc' in self.modalities_to_use and 'scenario_ids' in kwargs:
            sc_out = self.sc(
                kwargs['scenario_ids'],
                kwargs['scenario_types'],
                kwargs['phase_ids'],
                kwargs['timestamps'],
                T=T
            )                        # (B,T, sc_embed_dim)
            # hidden_dim 차원으로 투영
            h_sc = self.p_sc(sc_out)  # (B,T, hidden_dim)
            parts.append(h_sc)

        # Survey
        if 'survey' in self.modalities_to_use and 'survey' in kwargs:
            out = self.survey(kwargs['survey'])          # (B,enc_dim)
            h   = self.p_survey(out).unsqueeze(1).expand(-1, T, -1)  # (B,T,H)
            parts.append(h)

        # ——— 모든 파츠 합치고 Fusion ———
        cat = torch.cat(parts, dim=2)          # (B, T, in_ch)
        x   = cat.permute(0, 2, 1)             # (B, in_ch, T)
        return self.fusion_layer(x)            # (B, H, T)


# class MotionEncoder(nn.Module):
#     """
#     IMU를 Query로, 나머지 모달리티를 Context로 Cross-Attention 융합하는 인코더.
#     - Query: IMU 시퀀스 (B, T, H)
#     - Key/Value: VEH, PPG, SC, SURVEY (각각 pooled 또는 sequence)
#     """
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         hidden_dim = cfg.PretrainMotion.hidden_dim
#         # config에 정의된 사용할 모달리티 목록을 저장
#         self.modalities_to_use = set(cfg.PretrainMotion.modalities_to_use)
#         print(f"Pre-train Motion with: {self.modalities_to_use}, Cross-attention")

#         # 1) 각 모달리티 인코더
#         # IMU (시퀀스)
#         self.imu = IMUFeatureEncoder(cfg.Encoders.imu)
#         self.p_imu = nn.Linear(cfg.Encoders.imu['encoder_dim'], hidden_dim)
        
#         # VEH (다채널 시퀀스 -> pooled)
#         self.veh = VehicleTCNEncoder(cfg.Encoders.veh)
#         self.p_veh = nn.Linear(cfg.Encoders.veh['embed_dim'], hidden_dim)

#         # ======================= [수정된 부분 START] =======================
#         # PPG (static)
#         self.ppg = PPGEncoder(cfg.Encoders.ppg)
#         # Concat 기반 코드처럼, PPG TCN 출력(embed_dim)과 HRV 특징(6개)을 합친 차원을
#         # 프로젝션 레이어의 입력 차원으로 정확히 명시해줍니다.
#         ppg_in_dim = cfg.Encoders.ppg['embed_dim'] + 6
#         self.p_ppg = nn.Linear(ppg_in_dim, hidden_dim)
#         # ======================= [수정된 부분 END] =========================

#         # Scenario (pooled)
#         self.sc = ScenarioEmbedding(cfg.Encoders.sc)
#         self.p_sc = nn.Linear(cfg.Encoders.sc['embed_dim'], hidden_dim)
#         # Survey (static)
#         self.survey = PreSurveyEncoder(cfg.Encoders.survey)
#         self.p_survey = nn.Linear(cfg.Encoders.survey['embed_dim'], hidden_dim)

#         # 2) Cross-Attention 모듈
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=4,
#             dropout=0.3,
#             batch_first=True
#         )
#         self.norm_q = nn.LayerNorm(hidden_dim)
#         self.norm_kv = nn.LayerNorm(hidden_dim)
#         self.norm_out = nn.LayerNorm(hidden_dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )

#     def _process_hrv(self, batch, device):
#         ppg_rr = batch['ppg_rr'].to(device)
        
#         hrv_list = []
#         if ppg_rr.dim() > 1 and ppg_rr.shape[1] > 0:
#             hrv_list.append(ppg_rr.mean(dim=1, keepdim=True))
#             std_dev = torch.std(ppg_rr, dim=1, keepdim=True)
#             hrv_list.append(torch.nan_to_num(std_dev, nan=0.0))
#             hrv_list.append(torch.min(ppg_rr, dim=1, keepdim=True).values)
#             hrv_list.append(torch.max(ppg_rr, dim=1, keepdim=True).values)
#         else: # 데이터가 없는 경우 0으로 채움
#             hrv_list.extend([torch.zeros(ppg_rr.shape[0], 1, device=device)] * 4)

#         for key in ['ppg_rmssd', 'ppg_sdnn']:
#             value_tensor = batch[key].to(device).unsqueeze(1)
#             hrv_list.append(torch.nan_to_num(value_tensor, nan=0.0))
            
#         return torch.cat(hrv_list, dim=1)
   
#     def forward(self, **kwargs):
#         device = self.cfg.Project.device

#         # --- IMU Query 준비 ---
#         x_imu = kwargs['imu_seq']
#         len_imu = kwargs['imu_len']
#         imu_feat = self.imu(x_imu, len_imu)
#         q = self.p_imu(imu_feat)
#         q = self.norm_q(q)

#         # --- Context (Key/Value) 준비 ---
#         kv_list = []

#         if 'veh' in self.modalities_to_use:
#             veh_pooled = self.veh(kwargs['veh_seq'].permute(0, 2, 1), return_pooled=True)
#             kv_list.append(self.p_veh(veh_pooled).unsqueeze(1))
        
#         # ======================= [수정된 부분 START] =======================
#         if 'ppg' in self.modalities_to_use:
#             # TCN 출력과 HRV 특징을 결합하여 프로젝션 레이어에 전달합니다.
#             tcn_out = self.ppg(kwargs['ppg_seq'])
#             hrv_out = self._process_hrv(kwargs, device)
#             combined = torch.cat([tcn_out, hrv_out], dim=1)
#             kv_list.append(self.p_ppg(combined).unsqueeze(1))
#         # ======================= [수정된 부분 END] =========================

#         if 'sc' in self.modalities_to_use:
#             sc_out = self.sc(
#                 kwargs['scenario_ids'],
#                 kwargs['scenario_types'],
#                 kwargs['phase_ids'],
#                 kwargs['timestamps'],
#                 T=None
#             )
#             sc_emb = self.p_sc(sc_out)
#             kv_list.append(sc_emb.unsqueeze(1))

#         if 'survey' in self.modalities_to_use:
#             sv_out = self.survey(kwargs['survey'])
#             kv_list.append(self.p_survey(sv_out).unsqueeze(1))

#         if not kv_list:
#             out = q + self.ffn(q)
#             return out.permute(0, 2, 1)

#         kv = torch.cat(kv_list, dim=1)
#         kv = self.norm_kv(kv)

#         # --- Cross-Attention 실행 ---
#         attn_out, _ = self.cross_attn(q, kv, kv)
#         out = self.norm_out(q + attn_out)
#         out = out + self.ffn(out)

#         return out.permute(0, 2, 1)