import torch
import torch.nn as nn
import torch.nn.functional as F
from core.tft_encoder import TFTEncoder

class PPGEncoder(nn.Module):
    def __init__(self, static_input_size, tv_input_sizes, hidden_size,
                 num_targets=2, lstm_layers=1, dropout=0.1, max_seq_len=50):
        """
        PPG 데이터를 위한 TFT 기반 인코더.
        
        Args:
            static_input_size (int): static 입력(설문 등)의 feature 차원.
            tv_input_sizes (dict): time-varying 변수별 입력 차원. 
                                  예: {"ppg": 10, "rr": 1, "rmssd": 1, "sdnn": 1}
            hidden_size (int): 인코더 내부 hidden 차원 (예: 256).
            num_targets (int): 예측할 타깃 수. 여기서는 arousal와 valence → 2.
            lstm_layers (int): LSTM 층 수.
            dropout (float): dropout 확률.
            max_seq_len (int): 시퀀스 최대 길이 (positional encoding 용).
        """
        super(PPGEncoder, self).__init__()
        # TFTEncoder는 static 입력과 time-varying 입력을 받아 latent feature를 출력합니다.
        self.tft_encoder = TFTEncoder(static_input_size=static_input_size,
                                      tv_input_sizes=tv_input_sizes,
                                      hidden_size=hidden_size,
                                      num_targets=num_targets,
                                      lstm_layers=lstm_layers,
                                      dropout=dropout,
                                      max_seq_len=max_seq_len)

    def forward(self, static_input, tv_inputs):
        """
        Args:
            static_input: Tensor of shape (B, static_input_size)
            tv_inputs: dict mapping variable name -> Tensor of shape (B, T, input_dim)
        Returns:
            latent features: Tensor of shape (B, T, hidden_size)
        """
        latent = self.tft_encoder(static_input, tv_inputs)
        return latent


class MotionDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_motion_classes=5, dropout=0.3):
        """
        MotionDecoder for action classification.
        
        Args:
            input_dim (int): Fusion 모듈의 출력 feature 차원 (예: 256)
            hidden_dim (int): 공유 MLP의 hidden 차원
            num_motion_classes (int): 행동 분류 클래스 수 (예: 5)
            dropout (float): dropout 확률
        """
        super(MotionDecoder, self).__init__()
        # 시퀀스 전체 정보를 집약하기 위해 평균 풀링 사용 (입력: (B, T, input_dim))
        self.pool = nn.AdaptiveAvgPool1d(1)
        # 공유 MLP
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # 행동 분류를 위한 출력층
        self.motion_head = nn.Linear(hidden_dim, num_motion_classes)
        
    def forward(self, fused_features):
        # fused_features: (B, T, input_dim)
        x = fused_features.transpose(1, 2)  # → (B, input_dim, T)
        pooled = self.pool(x).squeeze(-1)     # → (B, input_dim)
        shared = self.shared_fc(pooled)       # → (B, hidden_dim)
        motion_logits = self.motion_head(shared)  # → (B, num_motion_classes)
        return motion_logits


class TOTDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        """
        TOTDecoder for takeover time regression.
        
        Args:
            input_dim (int): Fusion 모듈의 출력 feature 차원 (예: 256)
            hidden_dim (int): MLP의 hidden 차원
            dropout (float): dropout 확률
        """
        super(TOTDecoder, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # 회귀 output: 스칼라 값
        )
        
    def forward(self, fused_features):
        # fused_features: (B, T, input_dim)
        x = fused_features.transpose(1, 2)  # (B, input_dim, T)
        pooled = self.pool(x).squeeze(-1)     # (B, input_dim)
        takeover_time = self.fc(pooled)       # (B, 1)
        return takeover_time


class ACTDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        """
        ACTDecoder for anticipated collision time regression.
        
        Args:
            input_dim (int): Fusion 모듈의 출력 feature 차원 (예: 256)
            hidden_dim (int): MLP의 hidden 차원
            dropout (float): dropout 확률
        """
        super(ACTDecoder, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # 회귀 output: 스칼라 값
        )
        
    def forward(self, fused_features):
        # fused_features: (B, T, input_dim)
        x = fused_features.transpose(1, 2)  # (B, input_dim, T)
        pooled = self.pool(x).squeeze(-1)     # (B, input_dim)
        act_time = self.fc(pooled)            # (B, 1)
        return act_time


# 간단한 테스트 코드 (독립 실행 시)
if __name__ == "__main__":
    batch_size, seq_len, feature_dim = 8, 50, 256
    dummy_features = torch.randn(batch_size, seq_len, feature_dim)
    
    motion_decoder = MotionDecoder(input_dim=feature_dim, hidden_dim=128, num_motion_classes=5, dropout=0.3)
    tot_decoder = TOTDecoder(input_dim=feature_dim, hidden_dim=128, dropout=0.3)
    act_decoder = ACTDecoder(input_dim=feature_dim, hidden_dim=128, dropout=0.3)
    
    motion_logits = motion_decoder(dummy_features)
    tot_output = tot_decoder(dummy_features)
    act_output = act_decoder(dummy_features)
    
    print("MotionDecoder output shape:", motion_logits.shape)  # Expected: (8, 5)
    print("TOTDecoder output shape:", tot_output.shape)         # Expected: (8, 1)
    print("ACTDecoder output shape:", act_output.shape)         # Expected: (8, 1)

