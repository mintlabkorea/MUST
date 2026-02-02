# models/encoder/emotion_encoder.py

import torch
import torch.nn as nn
from types import SimpleNamespace

# 필요한 인코더 모듈들을 임포트합니다.
from models.encoder.ppg_TCN_encoder import PPGEncoder
from models.encoder.imu_encoder import IMUFeatureEncoder
from models.encoder.sc_encoder import ScenarioEmbedding
from models.encoder.survey_encoder import PreSurveyEncoder, StaticFeatureAttention
from models.encoder.veh_encoder import VehicleTCNEncoder

class EmotionEncoder(nn.Module):
    """
    다중 모달리티(PPG, IMU, SC, Survey) 특징을 인코딩하고 융합하여
    하나의 특징 벡터를 생성합니다.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # ——— 서브 인코더를 cfg.Encoders로부터 생성 ———
        self.imu    = IMUFeatureEncoder(cfg.Encoders.imu)
        self.ppg    = PPGEncoder(cfg.Encoders.ppg)
        self.veh    = VehicleTCNEncoder(cfg.Encoders.veh)
        self.sc     = ScenarioEmbedding(cfg.Encoders.sc)
        self.survey = PreSurveyEncoder(cfg.Encoders.survey)

        hidden_dim = cfg.PretrainEmotion.hidden_dim
        # --- 사용할 모달리티 목록을 set으로 저장하여 탐색 속도 향상 ---
        self.modalities_to_use = set(cfg.PretrainEmotion.modalities_to_use)
        print(f"[EmotionEncoder] Pre-training with modalities: {self.modalities_to_use}")
        # 1) 각 모달리티별 인코더 정의
        self.nets = nn.ModuleDict({
            'imu':    self.imu,
            'ppg':    self.ppg,
            'veh':    self.veh,
            'sc':     self.sc,
            'survey': self.survey
        })

        # PPG 입력 차원 = TCN 출력 차원  HRV 특징 6개 (rr_mean/std/min/max, rmssd, sdnn)
        ppg_in_dim = cfg.Encoders.ppg['embed_dim'] + 6
        
        # 2) 각 인코더 출력물을 공통 차원(hidden)으로 투영하는 프로젝션 레이어
        self.projs = nn.ModuleDict({
            'imu':    nn.Linear(cfg.Encoders.imu['encoder_dim'], hidden_dim),
            'veh':    nn.Linear(cfg.Encoders.veh['embed_dim'], hidden_dim),
            'ppg':    nn.Linear(ppg_in_dim, hidden_dim),
            'sc':     nn.Linear(cfg.Encoders.sc['embed_dim'], hidden_dim),
            'survey': nn.Linear(cfg.Encoders.survey['embed_dim'], hidden_dim),
        })

    def forward(self, batch: dict):

        device = next(self.parameters()).device
        # [수정] 기준 시퀀스 길이는 항상 존재하는 imu에서 가져옵니다.
        B, T = batch['imu_emotion'].shape[:2]
        
        temporal_parts = []
        static_part = None

        # --- 각 모달리티를 config에 따라 동적으로 처리 ---

        if 'imu' in self.modalities_to_use:
            imu_len = (batch['imu_emotion'].abs().sum(dim=-1) > 0).sum(dim=1)
            imu_out = self.nets.imu(batch['imu_emotion'].to(device), imu_len)
            temporal_parts.append(self.projs.imu(imu_out)) # (B, T, H)
        
        # --- Pooling이 필요한 특징들을 먼저 처리 ---
        pooled_parts = []
        if 'veh' in self.modalities_to_use:
            veh_out = self.nets.veh(batch['veh_emotion'].to(device).permute(0, 2, 1), return_pooled=True)
            pooled_parts.append(self.projs.veh(veh_out))

        if 'ppg' in self.modalities_to_use:
            ppg_tcn_out = self.nets.ppg(batch['ppg_emotion'].to(device).permute(0, 2, 1))
            hrv_features = self._process_hrv(batch, device) 
            combined_ppg = torch.cat([ppg_tcn_out, hrv_features], dim=1)
            pooled_parts.append(self.projs.ppg(combined_ppg))

        if 'sc' in self.modalities_to_use:
            sc_out = self.nets.sc(
                batch['scenario_evt_e'].to(device), batch['scenario_type_e'].to(device),
                batch['phase_evt_e'].to(device), batch['scenario_time_e'].to(device)
            )
            pooled_parts.append(self.projs.sc(sc_out))

        # --- Pooling된 특징들을 시간 축으로 확장하여 temporal_parts에 추가 ---
        if pooled_parts:
            # 모든 pooled 특징을 concat -> (B, H * N_pooled)
            concatenated_pooled = torch.cat(pooled_parts, dim=1)
            # (B, H*N) -> (B, 1, H*N) -> (B, T, H*N)
            expanded_pooled = concatenated_pooled.unsqueeze(1).expand(-1, T, -1)
            temporal_parts.append(expanded_pooled)

        # --- Survey (정적 특징) 처리 ---
        if 'survey' in self.modalities_to_use:
            static_part = self.projs.survey(self.nets.survey(batch['survey_e'].to(device)))
        else:
            static_part = torch.zeros(B, self.projs.survey.out_features, device=device)

        fused_temporal = torch.cat(temporal_parts, dim=2)

        return {
            'fused': fused_temporal,
            'static': static_part
        }

    def _process_hrv(self, batch, device):
        """
        HRV 관련 특징들을 하나의 텐서로 결합하는 헬퍼 함수.
        NaN 값을 0.0으로 안전하게 변환하는 로직이 추가되었습니다.
        """
        ppg_rr = batch['ppg_rr_emotion'].to(device)
        
        hrv_list = []
        if ppg_rr.dim() > 1 and ppg_rr.shape[1] > 0:
            # mean, std 계산 시 발생할 수 있는 NaN도 방지
            hrv_list.append(torch.nan_to_num(ppg_rr.mean(dim=1, keepdim=True), nan=0.0))
            std_dev = torch.std(ppg_rr, dim=1, keepdim=True)
            hrv_list.append(torch.nan_to_num(std_dev, nan=0.0))
            hrv_list.append(torch.min(ppg_rr, dim=1, keepdim=True).values)
            hrv_list.append(torch.max(ppg_rr, dim=1, keepdim=True).values)
        else: # 데이터가 없는 경우 0으로 채움
            hrv_list.extend([torch.zeros(ppg_rr.shape[0], 1, device=device)] * 4)

        # Dataloader에서 생성된 ppg_rmssd와 ppg_sdnn에 포함될 수 있는 NaN을
        # 0.0으로 안전하게 변환해줍니다.
        for key in ['ppg_rmssd_emotion', 'ppg_sdnn_emotion']:
            value_tensor = batch[key].to(device).unsqueeze(1)
            hrv_list.append(torch.nan_to_num(value_tensor, nan=0.0))
        
        return torch.cat(hrv_list, dim=1)
    
# class EmotionEncoder(nn.Module):
#     """
#     [수정됨] 다중 모달리티 특징을 인코딩하고 융합하며,
#     안정적인 학습을 위해 LayerNorm과 Dropout이 추가되었습니다.
#     """
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         # --- 서브 인코더 (기존과 동일) ---
#         self.imu    = IMUFeatureEncoder(cfg.Encoders.imu)
#         self.ppg    = PPGEncoder(cfg.Encoders.ppg)
#         self.veh    = VehicleTCNEncoder(cfg.Encoders.veh)
#         self.sc     = ScenarioEmbedding(cfg.Encoders.sc)
#         self.survey = PreSurveyEncoder(cfg.Encoders.survey)

#         hidden_dim = cfg.PretrainEmotion.hidden_dim
#         # ======================= [수정 1: hidden_dim 저장] =======================
#         # hidden_dim을 클래스 변수로 저장하여 forward 함수에서 참조할 수 있도록 합니다.
#         self.hidden_dim = hidden_dim
#         # =====================================================================
        
#         dropout_p = cfg.PretrainEmotion.dropout
#         self.modalities_to_use = set(cfg.PretrainEmotion.modalities_to_use)
#         print(f"[EmotionEncoder] Pre-training with modalities: {self.modalities_to_use}")
        
#         self.nets = nn.ModuleDict({
#             'imu': self.imu, 'ppg': self.ppg, 'veh': self.veh,
#             'sc': self.sc, 'survey': self.survey
#         })

#         ppg_in_dim = cfg.Encoders.ppg['embed_dim'] + 6
        
#         def create_projection_block(input_dim, output_dim):
#             return nn.Sequential(
#                 nn.Linear(input_dim, output_dim),
#                 nn.ReLU(),
#                 nn.LayerNorm(output_dim),
#                 nn.Dropout(dropout_p)
#             )

#         self.projs = nn.ModuleDict({
#             'imu':    create_projection_block(cfg.Encoders.imu['encoder_dim'], self.hidden_dim),
#             'veh':    create_projection_block(cfg.Encoders.veh['embed_dim'], self.hidden_dim),
#             'ppg':    create_projection_block(ppg_in_dim, self.hidden_dim),
#             'sc':     create_projection_block(cfg.Encoders.sc['embed_dim'], self.hidden_dim),
#             'survey': create_projection_block(cfg.Encoders.survey['embed_dim'], self.hidden_dim),
#         })

#         num_dynamic_modalities = len(self.modalities_to_use - {'survey'})
#         if 'imu' not in self.modalities_to_use: # imu가 없으면 pooled_parts만 존재
#             final_fusion_dim = self.hidden_dim * num_dynamic_modalities
#         elif not (self.modalities_to_use - {'imu', 'survey'}): # imu만 존재
#              final_fusion_dim = self.hidden_dim
#         else: # imu와 다른 pooled_parts가 함께 존재
#              final_fusion_dim = self.hidden_dim + self.hidden_dim * (num_dynamic_modalities -1)
        
#         self.final_norm = nn.LayerNorm(final_fusion_dim)
#         self.final_dropout = nn.Dropout(dropout_p)


#     def forward(self, batch: dict):
#         device = next(self.parameters()).device
        
#         # imu가 사용될 때만 T를 참조, 아닐 경우 기본값 설정
#         if 'imu_emotion' in batch and 'imu' in self.modalities_to_use:
#             B, T = batch['imu_emotion'].shape[:2]
#         else:
#             B, T = batch[list(batch.keys())[0]].shape[0], self.cfg.Data.seq_len # Fallback
        
#         temporal_parts = []
#         static_part = None

#         if 'imu' in self.modalities_to_use:
#             imu_len = (batch['imu_emotion'].abs().sum(dim=-1) > 0).sum(dim=1)
#             imu_out = self.nets.imu(batch['imu_emotion'].to(device), imu_len)
#             temporal_parts.append(self.projs.imu(imu_out))

#         pooled_parts = []
#         if 'veh' in self.modalities_to_use:
#             veh_out = self.nets.veh(batch['veh_emotion'].to(device).permute(0, 2, 1), return_pooled=True)
#             pooled_parts.append(self.projs.veh(veh_out))

#         if 'ppg' in self.modalities_to_use:
#             ppg_tcn_out = self.nets.ppg(batch['ppg_emotion'].to(device).permute(0, 2, 1))
#             hrv_features = self._process_hrv(batch, device) 
#             combined_ppg = torch.cat([ppg_tcn_out, hrv_features], dim=1)
#             pooled_parts.append(self.projs.ppg(combined_ppg))

#         if 'sc' in self.modalities_to_use:
#             sc_out = self.nets.sc(
#                 batch['scenario_evt_e'].to(device), batch['scenario_type_e'].to(device),
#                 batch['phase_evt_e'].to(device), batch['scenario_time_e'].to(device)
#             )
#             pooled_parts.append(self.projs.sc(sc_out))

#         if pooled_parts:
#             concatenated_pooled = torch.cat(pooled_parts, dim=1)
#             expanded_pooled = concatenated_pooled.unsqueeze(1).expand(-1, T, -1)
#             temporal_parts.append(expanded_pooled)

#         # --- Survey (정적 특징) 처리 ---
#         if 'survey' in self.modalities_to_use:
#             static_part = self.projs.survey(self.nets.survey(batch['survey_e'].to(device)))
#         else:
#             # ======================= [수정 2: 올바른 차원 참조] =======================
#             # self.projs.survey.out_features 대신 self.hidden_dim을 사용합니다.
#             static_part = torch.zeros(B, self.hidden_dim, device=device)
#             # =====================================================================

#         fused_temporal = torch.cat(temporal_parts, dim=2)
#         fused_temporal = self.final_norm(fused_temporal)
#         fused_temporal = self.final_dropout(fused_temporal)

#         return {
#             'fused': fused_temporal,
#             'static': static_part
#         }

#     def _process_hrv(self, batch, device):
#         # (이 함수는 수정 없이 그대로 유지)
#         ppg_rr = batch['ppg_rr_emotion'].to(device)
#         hrv_list = []
#         if ppg_rr.dim() > 1 and ppg_rr.shape[1] > 0:
#             hrv_list.append(torch.nan_to_num(ppg_rr.mean(dim=1, keepdim=True), nan=0.0))
#             std_dev = torch.std(ppg_rr, dim=1, keepdim=True)
#             hrv_list.append(torch.nan_to_num(std_dev, nan=0.0))
#             hrv_list.append(torch.min(ppg_rr, dim=1, keepdim=True).values)
#             hrv_list.append(torch.max(ppg_rr, dim=1, keepdim=True).values)
#         else:
#             hrv_list.extend([torch.zeros(ppg_rr.shape[0], 1, device=device)] * 4)
#         for key in ['ppg_rmssd_emotion', 'ppg_sdnn_emotion']:
#             value_tensor = batch[key].to(device).unsqueeze(1)
#             hrv_list.append(torch.nan_to_num(value_tensor, nan=0.0))
#         return torch.cat(hrv_list, dim=1)
    
    
# class EmotionEncoder(nn.Module):
#     """
#     Cross-Attention 기반 EmotionEncoder:
#     - Query: PPG + HRV features
#     - Context: modalities_to_use에 지정된 다른 피쳐들(imu, veh, sc, survey)을 pooling하여 사용
#     """

#     def __init__(self, cfg):
#         super().__init__()
#         self.hidden_dim = cfg.PretrainEmotion.hidden_dim
#         self.modalities = cfg.PretrainEmotion.modalities_to_use
#         print(f"[EmotionEncoderCrossAttention] Pre-train with modalities: {self.modalities}")

#         # dynamic 모달리티 (survey 제외)
#         self.dynamic_mods = [m for m in self.modalities if m != 'survey']
#         self.num_dyn = len(self.dynamic_mods)

#         # 1) 서브 인코더
#         self.encoders = nn.ModuleDict({'ppg': PPGEncoder(cfg.Encoders.ppg)})
#         if 'imu' in self.dynamic_mods:
#             self.encoders['imu'] = IMUFeatureEncoder(cfg.Encoders.imu)
#         if 'veh' in self.dynamic_mods:
#             self.encoders['veh'] = VehicleTCNEncoder(cfg.Encoders.veh)
#         if 'sc' in self.dynamic_mods:
#             self.encoders['sc'] = ScenarioEmbedding(cfg.Encoders.sc)
#         if 'survey' in self.modalities:
#             self.encoders['survey'] = PreSurveyEncoder(cfg.Encoders.survey)

#         # 2) 프로젝트: modality 개별 embed -> hidden_dim
#         ppg_dim = cfg.Encoders.ppg['embed_dim'] + 6
#         self.projectors = nn.ModuleDict({'ppg': nn.Linear(ppg_dim, self.hidden_dim)})
#         if 'imu' in self.dynamic_mods:
#             self.projectors['imu'] = nn.Linear(cfg.Encoders.imu['encoder_dim'], self.hidden_dim)
#         if 'veh' in self.dynamic_mods:
#             self.projectors['veh'] = nn.Linear(cfg.Encoders.veh['embed_dim'], self.hidden_dim)
#         if 'sc' in self.dynamic_mods:
#             self.projectors['sc'] = nn.Linear(cfg.Encoders.sc['embed_dim'], self.hidden_dim)
#         if 'survey' in self.modalities:
#             self.projectors['survey'] = nn.Linear(cfg.Encoders.survey['embed_dim'], self.hidden_dim)

#         # 3) Cross-Attention 레이어
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=self.hidden_dim,
#             num_heads=8,
#             dropout=0.3,
#             batch_first=True
#         )
#         self.norm_q = nn.LayerNorm(self.hidden_dim)
#         self.norm_kv = nn.LayerNorm(self.hidden_dim)
#         self.norm_out = nn.LayerNorm(self.hidden_dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim, self.hidden_dim)
#         )

#         # 4) 최종 projection: H -> H * num_dyn
#         self.final_proj = nn.Linear(self.hidden_dim, self.hidden_dim * self.num_dyn)

#     def forward(self, batch):
#         device = next(self.parameters()).device
#         B, T = batch['imu_emotion'].shape[0], batch['imu_emotion'].shape[1]

#         # --- Query: PPG + HRV ---
#         ppg_out = self.encoders['ppg'](batch['ppg_emotion'].to(device))  # (B, D_ppg)
#         hrv = self._process_hrv(batch, device)                            # (B, 6)
#         q0 = self.projectors['ppg'](torch.cat([ppg_out, hrv], dim=1))     # (B, H)
#         q = self.norm_q(q0).unsqueeze(1).expand(-1, T, -1)                 # (B, T, H)

#         # --- Context KV: pooled modalities (IMU, VEH, SC) ---
#         kv_list = []
#         if 'imu' in self.dynamic_mods:
#             imu_o = self.encoders['imu'](
#                 batch['imu_emotion'].to(device), batch['imu_e_lens'].to(device)
#             )  # (B, T, D_imu)
#             kv_list.append(self.projectors['imu'](imu_o.mean(dim=1)))  # (B, H)
#         if 'veh' in self.dynamic_mods:
#             veh_p = self.encoders['veh'](
#                 batch['veh_emotion'].to(device), return_pooled=True
#             )
#             kv_list.append(self.projectors['veh'](veh_p))             # (B, H)
#         if 'sc' in self.dynamic_mods:
#             sc_o = self.encoders['sc'](
#                 batch['scenario_evt_e'].to(device), batch['scenario_type_e'].to(device),
#                 batch['phase_evt_e'].to(device), batch['scenario_time_e'].to(device)
#             )
#             kv_list.append(self.projectors['sc'](sc_o))               # (B, H)

#         if not kv_list:
#             raise RuntimeError("No context modalities for cross-attention")

#         kv = torch.stack(kv_list, dim=1)      # (B, N_ctx, H)
#         kv_norm = self.norm_kv(kv)

#         # --- Cross-Attention 수행 ---
#         attn_out, _ = self.cross_attn(q, kv_norm, kv_norm)  # (B, T, H)
#         res = self.norm_out(q + attn_out)                    # (B, T, H)
#         out = res + self.ffn(res)                            # (B, T, H)

#         # --- 최종 dynamic features ---
#         fused_dyn = self.final_proj(out)  # (B, T, H * num_dyn)

#         # --- Static survey embedding ---
#         if 'survey' in self.modalities and 'static' in batch:
#             static = self.projectors['survey'](
#                 self.encoders['survey'](batch['static'].to(device))
#             )  # (B, H)
#         else:
#             static = torch.zeros(B, self.hidden_dim, device=device)

#         return {'fused': fused_dyn, 'static': static}

#     def _process_hrv(self, batch, device):
#         ppg_rr = batch['ppg_rr_emotion'].to(device)
#         hrv_list = []
#         if ppg_rr.dim() > 1 and ppg_rr.size(1) > 0:
#             hrv_list.extend([
#                 ppg_rr.mean(dim=1, keepdim=True),
#                 ppg_rr.std(dim=1, keepdim=True),
#                 ppg_rr.min(dim=1, keepdim=True).values,
#                 ppg_rr.max(dim=1, keepdim=True).values
#             ])
#         else:
#             hrv_list.extend([torch.zeros(ppg_rr.size(0), 1, device=device)] * 4)
#         for key in ['ppg_rmssd_emotion', 'ppg_sdnn_emotion']:
#             hrv_list.append(batch[key].to(device).unsqueeze(1))
#         return torch.cat(hrv_list, dim=1)  # (B, 6)
