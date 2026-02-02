import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes_arousal=3, num_classes_valence=3, dropout=0.3):
        """
        Args:
            input_dim (int): 인코더 출력의 feature 차원 (예: hidden_size)
            hidden_dim (int): 공유 MLP의 hidden 차원
            num_classes_arousal (int): arousal 분류 클래스 수 (예: 3)
            num_classes_valence (int): valence 분류 클래스 수 (예: 3)
            dropout (float): dropout 확률
        """
        super(MultiTaskHead, self).__init__()
        # 시간 축 평균 풀링 (입력: (B, T, input_dim) → (B, input_dim))
        self.pool = nn.AdaptiveAvgPool1d(1)
        # 공유 MLP
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # 각 타깃별 분리된 출력층
        self.arousal_head = nn.Linear(hidden_dim, num_classes_arousal)
        self.valence_head = nn.Linear(hidden_dim, num_classes_valence)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, input_dim) - 인코더의 latent feature
        Returns:
            dict: {
                "arousal_logits": (B, num_classes_arousal),
                "valence_logits": (B, num_classes_valence)
            }
        """
        # x: (B, T, input_dim) -> transpose to (B, input_dim, T) for pooling
        x_transposed = x.transpose(1, 2)
        pooled = self.pool(x_transposed).squeeze(-1)  # (B, input_dim)
        shared = self.shared_fc(pooled)              # (B, hidden_dim)
        arousal_logits = self.arousal_head(shared)
        valence_logits = self.valence_head(shared)
        return {"arousal_logits": arousal_logits, "valence_logits": valence_logits}

# 간단한 테스트 코드
if __name__ == "__main__":
    batch_size = 8
    seq_len = 50
    hidden_size = 256
    input_dim = hidden_size  # encoder 출력 차원
    dummy_encoder_output = torch.randn(batch_size, seq_len, input_dim)
    
    multitask_head = MultiTaskHead(input_dim=input_dim, hidden_dim=128, num_classes_arousal=3, num_classes_valence=3, dropout=0.3)
    outputs = multitask_head(dummy_encoder_output)
    print("Arousal logits shape:", outputs["arousal_logits"].shape)  # (8, 3)
    print("Valence logits shape:", outputs["valence_logits"].shape)  # (8, 3)
