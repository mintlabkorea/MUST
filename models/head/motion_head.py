import torch.nn as nn
import torch
from models.encoder.ms_tcn2_v6 import MS_TCN2_PG
from models.fusion.fusion_block import TaskSpecificFusion

# class MotionHead(nn.Module):
#     """
#     Conv1d(head) : (B, H, T) → frame-wise logits (B, T, C)
#     """
#     def __init__(self, hidden, n_cls):
#         super().__init__()
#         self.cls = nn.Conv1d(hidden, n_cls, kernel_size=1)

#     def forward(self, features):         # (B, H, T)
#         return self.cls(features).permute(0, 2, 1)  # (B, T, C)

class MotionHead(nn.Module):
    """
    MS-TCN2의 PG(Prediction Generation) 모듈만을 사용하여
    시간적 특징을 분석하고 최종 로짓을 출력하는 헤드.
    """
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        
        self.tcn_predictor = MS_TCN2_PG(
            num_layers_PG=10,
            num_layers_R=10,
            num_R=3,
            num_f_maps=64,
            dim=feature_dim, 
            num_classes=num_classes
        )

    def forward(self, features):
        """
        Args:
            features (torch.Tensor): MotionEncoder로부터 온 융합 특징 (B, H, T)
        """
        # tcn_predictor 전체 대신 .PG 모듈만 호출하여 softmax를 회피
        logits = self.tcn_predictor.PG(features) # (B, num_classes, T)
        
        # 최종 출력을 (B, T, C) 형태로 변환하여 반환
        return logits.permute(0, 2, 1)
    
# class MotionHead(nn.Module):
#     """
#     Cross-attention 기반 MotionHead
#     - query: imu 특징
#     - context: imu, veh, sc 특징
#     - head: frame-wise 선형 분류기
#     """
#     def __init__(self, feature_dim, num_classes, num_heads=4):
#         super().__init__()
#         self.fusion = TaskSpecificFusion(
#             query_modality="imu",
#             context_modalities=["imu", "veh", "sc"],
#             feature_dim=feature_dim,
#             num_heads=num_heads,
#             output_dim=feature_dim
#         )
#         self.cls = nn.Linear(feature_dim, num_classes)

#     def forward(self, feature_input):
#         """
#         Args:
#             feature_dict: {
#                 "imu": Tensor (B, T, D),
#                 "veh": Tensor (B, T, D),
#                 "sc":  Tensor (B, T, D)
#             }
#         Returns:
#             logits: Tensor (B, T, num_classes)
#         """
#        # Tensor 만 넘어온다면—pre-train 모드일 때—바로 분류기로 통과
#         if isinstance(feature_input, torch.Tensor):
#             # feature_input: (B, T, D)
#             return self.cls(feature_input)          # (B, T, C)

#         # dict 으로 넘어온다면 cross-attention 융합 수행
#         mot_repr = self.fusion(feature_input)        # (B, T, D)
#         return self.cls(mot_repr)                    # (B, T, C)