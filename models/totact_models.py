import torch
import torch.nn as nn
from models.encoder.veh_encoder import VehicleTCNEncoder
from models.encoder.sc_encoder import ScenarioEmbedding

class TOT_Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.veh_encoder = VehicleTCNEncoder(cfg.Encoders.veh)
        self.sc_encoder = ScenarioEmbedding(cfg.Encoders.sc)
        
        fusion_dim = cfg.Encoders.veh['embed_dim'] + cfg.Encoders.sc['embed_dim']
        # num_layers=1 이면 nn.GRU 의 dropout 인자는 무시되므로 넣지 않는다.
        self.gru = nn.GRU(fusion_dim, 128, batch_first=True)

        # 시간축 pooling 방식. 'last' 는 마지막 스텝만 사용하는 기존 동작.
        self.pool_mode = getattr(cfg.TOT, 'time_pool', 'mean')

        # [수정] Regressor -> Classifier
        num_classes = getattr(cfg.TOT, 'num_classes', 3)
        head_in = 128 * (2 if self.pool_mode == 'meanmax' else 1)
        self.classifier = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes) # 출력 차원을 클래스 개수로 변경
        )

    def forward(self, batch):
        device = batch['veh'].device
        
        veh_feat_tcn = self.veh_encoder(batch['veh'].permute(0, 2, 1))
        veh_feat = veh_feat_tcn.permute(0, 2, 1)
        
        # --- [FIX] Get sequence length T and pass it to the sc_encoder ---
        T = veh_feat.shape[1]
        
        sc_feat = self.sc_encoder(
            batch['sc_evt'].to(device),
            batch['sc_type'].to(device),
            batch['sc_phase'].to(device),
            batch['sc_time'].to(device),
            T=T  # Pass the sequence length
        )
        
        fused = torch.cat([veh_feat, sc_feat], dim=-1)
        gru_out, _ = self.gru(fused)
        pooled = self._time_pool(gru_out)

        prediction_logits = self.classifier(pooled) # regressor -> classifier
        return prediction_logits

    def _time_pool(self, h: torch.Tensor) -> torch.Tensor:
        """시간축 집계.

        기존 구현은 h[:, -1, :] (마지막 스텝)만 사용했는데, 입력이 3s x 100Hz = 300 스텝이라
        단층 GRU 의 gradient 가 시퀀스 앞쪽까지 도달하지 못해 학습 자체가 되지 않았다.
        328개 학습 샘플을 200 epoch 동안 암기시켜도 train acc 가 0.52~0.62 에서 진동.
        mean pooling 으로 바꾸면 같은 조건에서 0.80 까지 올라간다.
        (2026-08-24 실험: logs/20260824_fullrun/tot_sweep.log)
        """
        if self.pool_mode == 'last':
            return h[:, -1, :]
        if self.pool_mode == 'meanmax':
            return torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        return h.mean(dim=1)

class ACT_Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.veh_encoder = VehicleTCNEncoder(cfg.Encoders.veh)
        self.sc_encoder = ScenarioEmbedding(cfg.Encoders.sc)
        
        fusion_dim = cfg.Encoders.veh['embed_dim'] + cfg.Encoders.sc['embed_dim']
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, batch):
        device = batch['veh'].device
        
        veh_feat_tcn = self.veh_encoder(batch['veh'].permute(0, 2, 1))
        veh_feat = veh_feat_tcn.permute(0, 2, 1)
        
        # --- [FIX] Get sequence length T and pass it to the sc_encoder ---
        T = veh_feat.shape[1]
        
        sc_feat = self.sc_encoder(
            batch['sc_evt'].to(device),
            batch['sc_type'].to(device),
            batch['sc_phase'].to(device),
            batch['sc_time'].to(device),
            T=T # Pass the sequence length
        )
        
        fused = torch.cat([veh_feat, sc_feat], dim=-1)
        prediction = self.regressor(fused)
        return prediction
    
class EnhancedTOTModel(nn.Module):
    def __init__(self, cfg, pretrained_tot_baseline, pretrained_fusion_model):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.Project.device

        # --- 전문가 모델들을 불러와 동결 ---
        self.baseline = pretrained_tot_baseline
        self.fusion_model = pretrained_fusion_model
        
        for param in self.baseline.parameters():
            param.requires_grad = False # 베이스라인 모델 가중치 동결
        for param in self.fusion_model.parameters():
            param.requires_grad = False # 퓨전 모델 가중치 동결

        # --- 최종 결정을 내릴 '결정권자(Fusion Head)' MLP ---
        baseline_feature_dim = 128 # TOT_Baseline GRU의 hidden_size
        motion_feature_dim = cfg.FusionModel.hidden_dim
        emotion_feature_dim = cfg.FusionModel.hidden_dim
        
        combined_dim = baseline_feature_dim + motion_feature_dim + emotion_feature_dim
        
        self.fusion_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3) # 최종 TOT 클래스(3개) 예측
        )

    def forward(self, batch):
        
        # --- 1. TOT 베이스라인 모델로부터 '저수준 특징' 추출 ---
        veh_feat_tcn = self.baseline.veh_encoder(batch['veh'].permute(0, 2, 1))
        veh_feat = veh_feat_tcn.permute(0, 2, 1)
        T = veh_feat.shape[1]
        sc_feat = self.baseline.sc_encoder(
            batch['sc_evt'].to(self.device),
            batch['sc_type'].to(self.device),
            batch['sc_phase'].to(self.device),
            batch['sc_time'].to(self.device),
            T=T
        )
        fused_baseline = torch.cat([veh_feat, sc_feat], dim=-1)
        gru_out, _ = self.baseline.gru(fused_baseline)
        baseline_features = gru_out[:, -1, :]

        # --- 2. [수정] 퓨전 모델로부터 '고수준 문맥 특징'을 조건부로 추출 ---
        batch_size = baseline_features.shape[0]
        motion_feature_dim = self.cfg.FusionModel.hidden_dim
        emotion_feature_dim = self.cfg.FusionModel.hidden_dim
        
        # 'imu_emotion' 키가 있는지 확인하여 멀티모달 데이터인지 판단
        if 'imu_emotion' in batch:
            with torch.no_grad():
                fusion_out = self.fusion_model(batch, task_type='emotion')
                motion_context = fusion_out['fused_motion']
                emotion_context = fusion_out['fused_emotion']
            
            # 감정 라벨이 유효할 때만 감정 특징을 사용
            valid_emotion_mask = (batch.get('valence_reg_emotion', torch.tensor(-100)).squeeze() != -100).float().unsqueeze(1)
            conditional_emotion_context = emotion_context * valid_emotion_mask
        else:
            # 베이스라인 데이터일 경우, 문맥 특징을 0으로 채움
            motion_context = torch.zeros(batch_size, motion_feature_dim, device=self.device)
            conditional_emotion_context = torch.zeros(batch_size, emotion_feature_dim, device=self.device)

        # --- 3. 특징 조합 및 최종 예측 --- (기존 코드와 동일)
        combined_features = torch.cat([
            baseline_features, 
            motion_context, 
            conditional_emotion_context
        ], dim=1)
        
        final_logits = self.fusion_head(combined_features)
        
        return final_logits

class EnhancedACTModel(nn.Module):
    def __init__(self, cfg, pretrained_act_baseline, pretrained_fusion_model):
        super().__init__()
        
        self.cfg = cfg
        self.device = cfg.Project.device

        # --- 전문가 모델들을 불러와 동결 ---
        self.baseline = pretrained_act_baseline
        self.fusion_model = pretrained_fusion_model
        
        for param in self.baseline.parameters():
            param.requires_grad = False # 베이스라인 모델 가중치 동결
        for param in self.fusion_model.parameters():
            param.requires_grad = False # 퓨전 모델 가중치 동결

        # --- 최종 결정을 내릴 '결정권자(Fusion Head)' MLP ---
        # 1. ACT 베이스라인의 특징 (veh+sc fusion dim)
        # 2. 퓨전 모델의 행동 특징 (hidden_dim)
        # 3. 퓨전 모델의 감정 특징 (hidden_dim)
        baseline_feature_dim = cfg.Encoders.veh['embed_dim'] + cfg.Encoders.sc['embed_dim']
        motion_feature_dim = cfg.FusionModel.hidden_dim
        emotion_feature_dim = cfg.FusionModel.hidden_dim
        
        combined_dim = baseline_feature_dim + motion_feature_dim + emotion_feature_dim
        
        # ACT는 시퀀스 전체를 예측하는 회귀 모델
        self.fusion_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3) 
        )

    def forward(self, batch):
        device = batch['veh'].device
        
        # --- 1. ACT 베이스라인 모델로부터 '저수준 특징' 추출 ---
        veh_feat_tcn = self.baseline.veh_encoder(batch['veh'].permute(0, 2, 1))
        veh_feat = veh_feat_tcn.permute(0, 2, 1)
        T = veh_feat.shape[1]
        sc_feat = self.baseline.sc_encoder(
            batch['sc_evt'].to(self.device),
            batch['sc_type'].to(self.device),
            batch['sc_phase'].to(self.device),
            batch['sc_time'].to(self.device),
            T=T
        )
        baseline_features = torch.cat([veh_feat, sc_feat], dim=-1)

        # --- 2. [수정] 퓨전 모델로부터 '고수준 문맥 특징'을 조건부로 추출 ---
        batch_size = baseline_features.shape[0]
        motion_feature_dim = self.cfg.FusionModel.hidden_dim
        emotion_feature_dim = self.cfg.FusionModel.hidden_dim

        if 'imu_emotion' in batch:
            with torch.no_grad():
                fusion_out = self.fusion_model(batch, task_type='emotion')
                motion_context = fusion_out['fused_motion'].unsqueeze(1).expand(-1, T, -1)
                emotion_context = fusion_out['fused_emotion'].unsqueeze(1).expand(-1, T, -1)

            valid_emotion_mask = (batch.get('valence_reg_emotion', torch.tensor(-100)).squeeze() != -100).float().unsqueeze(1).unsqueeze(2)
            conditional_emotion_context = emotion_context * valid_emotion_mask
        else:
            motion_context = torch.zeros(batch_size, T, motion_feature_dim, device=self.device)
            conditional_emotion_context = torch.zeros(batch_size, T, emotion_feature_dim, device=self.device)

        # --- 3. 특징 조합 및 최종 예측 --- (기존 코드와 동일)
        combined_features = torch.cat([
            baseline_features, 
            motion_context, 
            conditional_emotion_context
        ], dim=-1)
        
        final_prediction = self.fusion_head(combined_features)
        
        return final_prediction

# =============================================================================
#  TOT 회귀 모델 (2026-08-24 실험 결과 채택)
#
#  subject-wise 6-fold CV (n=405) 기준 성능:
#    2-class @1.45s : pooled acc 0.8988 / bal 0.8790 / lift +0.2716 (5/6 fold >= 0.85)
#    3-class        : pooled acc 0.7877 / bal 0.7717 / lift +0.3901
#  기존 3-class 분류(TOT_Baseline, last-step) 대비 0.7210 -> 0.7877.
#
#  설계 근거 (logs/20260824_fullrun/results.md):
#   - TOT 를 클래스로 뭉개지 않고 연속값으로 학습. 순서/크기 정보를 모두 사용.
#   - log(TOT) 를 타깃으로 사용. TOT 는 중앙값 1.27s 에 최대 14.13s 인 우편향 분포라
#     원값으로 학습하면 큰 값에 손실이 지배당한다 (Spearman 0.607 -> 0.807).
#   - IMU 는 넣지 않는다. 학습 샘플이 328개뿐이라 과적합. CV 0.727 -> 0.685 로 악화.
#   - 입력 표준화도 하지 않는다. 전 항목에서 오히려 성능 하락.
# =============================================================================

class _ConvBranch(nn.Module):
    """dilated Conv1d 스택. (B, T, C_in) -> (B, C_out, T)"""

    def __init__(self, in_channels: int, channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, channels, 5, padding=2),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, 5, padding=2),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, 5, padding=4, dilation=2),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.permute(0, 2, 1))


class TOTRegressor(nn.Module):
    """연속 TOT 예측 모델. log(TOT) 를 예측하고 exp 로 되돌린다.

    veh conv branch -> (sc embedding concat) -> bi-GRU -> mean&max pool -> MLP
    출력은 log 공간의 스칼라. `predict_seconds` 로 초 단위 TOT 를 얻는다.
    """

    TIME_STEPS = 100  # 윈도우 길이와 무관하게 GRU 입력 길이를 고정 (3s/5s 공정 비교용)

    def __init__(self, cfg, hidden: int = 128):
        super().__init__()
        veh_dim = len(getattr(cfg.Data, 'veh_cols', [])) or cfg.Encoders.veh['input_dim']
        self.veh = _ConvBranch(veh_dim)
        self.evt = nn.Embedding(64, 16)
        self.typ = nn.Embedding(64, 8)
        self.pha = nn.Embedding(8, 4)
        self.gru = nn.GRU(64 + 28, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 4, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )

    def forward(self, batch) -> torch.Tensor:
        """returns log(TOT) 예측, shape (B,)"""
        x = torch.nn.functional.adaptive_avg_pool1d(
            self.veh(batch['veh']), self.TIME_STEPS
        ).permute(0, 2, 1)
        T = x.shape[1]
        sc = torch.cat([
            self.evt(batch['sc_evt'].clamp(0, 63)),
            self.typ(batch['sc_type'].clamp(0, 63)),
            self.pha(batch['sc_phase'].clamp(0, 7)),
        ], dim=-1)
        x = torch.cat([x, sc.unsqueeze(1).expand(-1, T, -1)], dim=-1)
        h, _ = self.gru(x)
        return self.head(torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def predict_seconds(self, batch) -> torch.Tensor:
        return torch.exp(self(batch))


def bin_tot(tot_seconds, thresholds):
    """연속 TOT(초)를 클래스로 변환.

    thresholds 길이 1 -> 2-class, 길이 2 -> 3-class.
    회귀 모델은 재학습 없이 임계값만 바꿔 평가할 수 있다는 점이 핵심 이점.
    """
    import numpy as np
    v = np.asarray(tot_seconds, dtype=float)
    if len(thresholds) == 1:
        return (v > thresholds[0]).astype(int)
    t1, t2 = thresholds
    return np.where(v <= t1, 0, np.where(v <= t2, 1, 2))
