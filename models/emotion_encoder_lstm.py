# models/encoder/emotion_encoder.py

import torch
import torch.nn as nn

# 필요한 인코더 모듈들을 임포트합니다.
from models.encoder.ppg_lstm_encoder import PPGEncoder
from models.encoder.imu_encoder import IMUFeatureEncoder
from models.encoder.sc_encoder import ScenarioEmbedding
from models.encoder.survey_encoder import PreSurveyEncoder

class EmotionEncoder(nn.Module):
    """
    다중 모달리티(PPG, IMU, SC, Survey) 특징을 인코딩하고 융합하여
    하나의 특징 벡터를 생성합니다.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 1) 각 모달리티별 인코더 정의
        self.nets = nn.ModuleDict({
            'imu': IMUFeatureEncoder(cfg),
            'ppg': PPGEncoder(cfg), # ppg_lstm_encoder.py의 PPGEncoder 사용
            'sc' : ScenarioEmbedding(cfg),
            'survey' : PreSurveyEncoder(cfg)
        })

        # 2) 각 인코더 출력물을 공통 차원(hidden)으로 투영하는 프로젝션 레이어
        # PPGEncoder는 이미 cfg.ppg_params['embed_dim'] 차원으로 최종 출력합니다.
        # 따라서 ppg_in_dim은 단순히 cfg.ppg_params['embed_dim']이 되어야 합니다.
        ppg_in_dim = cfg.ppg_params['embed_dim'] # <-- 이 부분을 수정
        self.projs = nn.ModuleDict({
            'imu'   : nn.Linear(cfg.imu_params['encoder_dim'], cfg.hidden),
            'ppg'   : nn.Linear(ppg_in_dim, cfg.hidden),
            'sc'    : nn.Linear(cfg.sc_params['embed_dim'],  cfg.hidden),
            'survey': nn.Linear(cfg.survey_params['input_dim'],cfg.hidden)
        })

    def forward(self, batch):
        """
        배치 데이터를 입력받아 모든 모달리티를 인코딩하고 융합합니다.

        Args:
            batch (dict): 데이터 로더로부터 받은 데이터 딕셔너리

        Returns:
            torch.Tensor: 최종 융합된 특징 벡터 (B, H)
        """
        device = next(self.parameters()).device

        # --- IMU 인코딩 & 프로젝션 ---
        imu_out = self.nets['imu'](
            batch['imu_emotion'].to(device),
            batch['imu_e_lens'].to(device)
        ).mean(dim=1)
        imu_emb = self.projs['imu'](imu_out)

        # --- PPG & HRV 인코딩 & 프로젝션 ---
        # ppg_lstm_encoder의 PPGEncoder는 RR 시퀀스, RMSSD, SDNN을 모두 직접 처리합니다.
        # 따라서 EmotionEncoder에서 별도로 HRV 특징을 추출하여 결합할 필요가 없습니다.
        ppg_emb_from_encoder = self.nets['ppg'](
            batch['ppg_emotion'].to(device),
            batch['ppg_rr_emotion'].to(device), # RR 시퀀스 전달
            batch['ppg_rmssd_emotion'].to(device), # RMSSD 전달
            batch['ppg_sdnn_emotion'].to(device) # SDNN 전달
        )
        ppg_emb = self.projs['ppg'](ppg_emb_from_encoder) # 인코더의 최종 출력 사용

        # --- Scenario & Survey 인코딩 & 프로젝션 ---
        sc_out = self.nets['sc'](
            batch['scenario_evt_e'].to(device),
            batch['scenario_type_e'].to(device),
            batch['phase_evt_e'].to(device),
            batch['scenario_time_e'].to(device)
        )
        sc_emb = self.projs['sc'](sc_out)
        surv_emb = self.projs['survey'](batch['survey_e'].to(device))

        # --- 특징 융합 (Feature Fusion) ---
        # 간단하게 모든 임베딩을 평균내어 융합합니다.
        emb_list = [ppg_emb, imu_emb, sc_emb, surv_emb]
        fused = torch.stack(emb_list, dim=0).mean(dim=0)

        return fused

