# models/fusion/predictors.py

import torch
import torch.nn as nn
from .fusion_block import TaskSpecificFusion

# class MotionPredictor(nn.Module):
#     def __init__(self, feature_dim, output_dim):
#         super().__init__()
#         self.fusion = TaskSpecificFusion(
#             query_modality="imu",
#             context_modalities=["imu", "veh", "sc"], # <-- 'survey' 제거
#             feature_dim=feature_dim,
#             num_heads=4,
#             output_dim=feature_dim
#         )
#         self.head = nn.Linear(feature_dim, output_dim)

#     def forward(self, feature_dict, return_feature=False):
#         # feature_dict: {'imu': (B,1,D), 'veh': (B,1,D), ...}
#         mot_repr = self.fusion(feature_dict)  # (B, 1, D)
#         mot_logits = self.head(mot_repr)      # (B, 1, output_dim)
        
#         if return_feature:
#             return mot_repr, mot_logits
#         else:
#             return mot_logits

class MotionPredictor(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super().__init__()
        self.fusion = TaskSpecificFusion(
            query_modality="imu",
            context_modalities=["imu", "veh", "sc"],
            feature_dim=feature_dim,
            num_heads=4,
            output_dim=feature_dim
        )
        # --- [수정] ---
        # 1. context를 처리할 프로젝션 레이어를 추가합니다.
        self.context_proj = nn.Linear(feature_dim, feature_dim)
        # 2. 최종 예측을 위한 head는 그대로 둡니다. 입력 차원을 바꿀 필요가 없습니다.
        self.head = nn.Linear(feature_dim, output_dim)

    # 3. forward 함수 로직을 완전히 변경합니다.
    def forward(self, feature_dict, context=None, return_feature=False):
        # (1) self.fusion을 통해 모션 특징 시퀀스를 얻습니다.
        # mot_repr의 모양: (Batch, SeqLen, feature_dim)
        mot_repr = self.fusion(feature_dict)

        # (2) context(감정 정보)가 주어지면, 모션 특징 시퀀스에 더해줍니다.
        if context is not None:
            # context의 모양: (Batch, feature_dim)
            context_proj = self.context_proj(context)
            
            # context_proj를 (B, D) -> (B, 1, D)로 만든 후,
            # mot_repr의 시퀀스 길이(T)에 맞게 복제(expand)합니다. -> (B, T, D)
            context_expanded = context_proj.unsqueeze(1).expand_as(mot_repr)
            
            # 모션 시퀀스의 모든 타임스텝에 감정 컨텍스트를 더해줍니다.
            combined_repr = mot_repr + context_expanded
        else:
            # context가 없으면 원래의 모션 특징을 그대로 사용합니다.
            combined_repr = mot_repr

        # (3) 풍부해진 특징 시퀀스를 head에 통과시켜 최종 로짓을 계산합니다.
        # 입력이 (B, T, D_in)이면 출력은 (B, T, D_out)이 됩니다.
        mot_logits = self.head(combined_repr)

        if return_feature:
            return combined_repr, mot_logits # 풍부해진 특징을 반환
        else:
            return mot_logits

class EmotionPredictor(nn.Module):
    """
    GRU 기반의 감정 예측기.
    주어진 감정 특징(features)과 모션 컨텍스트(context)를
    Concat 기반으로 융합하여 감정을 예측합니다.
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.4,
            bidirectional=True
        )
        self.ln = nn.LayerNorm(hidden_dim * 2)
        # GRU 출력(H*2)과 컨텍스트(H)를 합친 크기를 입력으로 받습니다.
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, num_classes)
        )
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, features, context):
        features_tensor = features['fused']
        batch_size = features_tensor.shape[0]
        # 1. GRU로 시간적 특징 처리
        gru_out, _ = self.gru(features_tensor)
        
        # 2. 마지막 타임스텝의 은닉 상태를 사용
        last_hidden_state = self.ln(gru_out[:, -1, :])

        # 3. Context를 투영 (차원 맞추기)
        if context is not None:
            # context가 있으면 이전과 동일하게 처리
            if context.dim() == 3 and context.size(1) == 1:
                context = context.squeeze(1)
            context_proj = self.context_proj(context)
        else:
            # context가 없으면 (평가 시) 0으로 채워진 플레이스홀더 텐서를 생성합니다.
            hidden_dim = self.context_proj.out_features
            context_proj = torch.zeros(batch_size, hidden_dim, device=last_hidden_state.device)


        # 4. GRU 출력과 컨텍스트를 Concat으로 결합
        combined = torch.cat([last_hidden_state, context_proj], dim=1)
        
        # 5. 최종 로짓 계산
        logits = self.fc(combined)
        return logits

        
# class EmotionPredictor(nn.Module):
#     """
#     GRU 기반 감정 예측기.
#     시간적 특징(features)과 정적 특징(context)을 입력으로 받습니다.
#     """
#     def __init__(self, input_dim, hidden_dim, num_classes):
#         super().__init__()
#         # GRU는 시간적 특징을 처리합니다.
#         self.gru = nn.GRU(
#             input_size=input_dim,
#             hidden_size=hidden_dim,
#             num_layers=2,
#             batch_first=True,
#             dropout=0.4,
#             bidirectional=True
#         )
#         self.ln = nn.LayerNorm(hidden_dim * 2)
        
#         # 최종 FC 레이어는 GRU 출력과 context를 합친 것을 입력으로 받습니다.
#         # (GRU 양방향 출력) + (Context) -> num_classes
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.4), # 예시: FC 레이어 뒤에 Dropout 추가
#             nn.Linear(hidden_dim, num_classes)
#         )
#         self.context_proj = nn.Linear(hidden_dim, hidden_dim) # context_dim -> hidden_dim

#     def forward(self, features, context):
#         """
#         Args:
#             features (torch.Tensor): (B, T, D_features) 모양의 시간적 특징
#             context (torch.Tensor): (B, D_context) 모양의 정적 특징
#         """
#         # 1. GRU로 시간적 특징 처리
#         gru_out, _ = self.gru(features)  # (B, T, H*2)
        
#         # 2. 마지막 타임스텝의 은닉 상태를 사용
#         last_hidden_state = self.ln(gru_out[:, -1, :]) # (B, H*2)

#         # 3. Context를 GRU의 hidden_dim에 맞게 투영
#         #    context가 (B,1,H)로 들어올 수도 있으니, 2D로 변환
#         if context.dim() == 3 and context.size(1) == 1:
#             context = context.squeeze(1)  # -> (B, H)
#         context_proj = self.context_proj(context)  # (B, H)

#         # 4. GRU 출력과 context를 결합
#         combined = torch.cat([last_hidden_state, context_proj], dim=1) # (B, H*2 + H)
        
#         # 5. 최종 로짓 계산
#         logits = self.fc(combined)

#         return logits