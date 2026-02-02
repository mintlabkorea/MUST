# models/fusion/predictors.py

import torch
import torch.nn as nn
from .fusion_blocks import TaskSpecificFusion

class MotionPredictor(nn.Module):
    """
    모션 예측을 위한 최종 모듈.
    내부적으로 TaskSpecificFusion을 사용하여 Late-fusion을 수행합니다.
    """
    def __init__(self, feature_dim, output_dim):
        super().__init__()
        self.fusion = TaskSpecificFusion(
            query_modality="imu",
            context_modalities=["imu", "veh", "sc", "survey"],
            feature_dim=feature_dim,
            num_heads=4,
            output_dim=feature_dim
        )
        self.head = nn.Linear(feature_dim, output_dim)

    def forward(self, feature_dict, return_feature=False):
        # feature_dict: {'imu': (B,1,D), 'veh': (B,1,D), ...}
        mot_repr = self.fusion(feature_dict)  # (B, 1, D)
        mot_logits = self.head(mot_repr)      # (B, 1, output_dim)
        
        if return_feature:
            return mot_repr, mot_logits
        else:
            return mot_logits

class EmotionPredictor(nn.Module):
    """
    감정 예측을 위한 최종 모듈.
    MotionPredictor의 결과(mot_repr)를 컨텍스트로 받아 '행동->감정' 인과관계를 모델링합니다.
    """
    def __init__(self, feature_dim, num_valence, num_arousal):
        super().__init__()
        # 감정 관련 특징들을 1차 융합
        self.fusion = TaskSpecificFusion(
            query_modality="ppg",
            context_modalities=["ppg", "sc", "survey"], # 감정에 더 중요한 모달리티들
            feature_dim=feature_dim,
            num_heads=4,
            output_dim=feature_dim
        )
        # 1차 융합된 감정 특징 + 행동 컨텍스트를 최종 결합
        self.final_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim), # emo_repr + mot_repr
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.val_head = nn.Linear(feature_dim, num_valence)
        self.aro_head = nn.Linear(feature_dim, num_arousal)

    def forward(self, feature_dict, context):
        """
        feature_dict: {'ppg': (B,1,D), 'sc': (B,1,D), ...}
        context: MotionPredictor에서 온 mot_repr (B,1,D)
        """
        # 1. 감정 특징들을 1차 융합
        emo_repr = self.fusion(feature_dict)  # (B, 1, D)

        # 2. 행동 컨텍스트와 결합
        combined_repr = torch.cat([emo_repr, context], dim=-1) # (B, 1, 2D)
        final_repr = self.final_mlp(combined_repr).squeeze(1) # (B, D)
        
        # 3. 최종 예측
        val_logits = self.val_head(final_repr)
        aro_logits = self.aro_head(final_repr)
        
        return {
            'valence_logits': val_logits,
            'arousal_logits': aro_logits
        }