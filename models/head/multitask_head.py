import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskHead(nn.Module):
    def __init__(self, seq_input_dim,
                 pooled_hidden_dim=128,
                 num_motion_classes=5,
                 num_valence_classes=3,
                 num_arousal_classes=3,
                 dropout_p=0.3,
                 include_motion=True,
                 use_relu_for_regression=False):
        """
        MultiTaskHead는 Fusion 인코더 (예: (B, T, seq_input_dim))의 출력을 받아
        모션 분류 (frame 단위), 감정 분류 (valence & arousal) 및 회귀 (TOT, ACT)를 처리합니다.
        
        Args:
            seq_input_dim (int): Fusion 인코더 출력의 feature 차원, 예: hidden_size.
            pooled_hidden_dim (int): 시퀀스 전체 평균 풀링 후 공유 FC layer의 hidden 차원.
            num_motion_classes (int): 모션 분류 클래스 수.
            num_valence_classes (int): Valence 분류 클래스 수.
            num_arousal_classes (int): Arousal 분류 클래스 수.
            dropout_p (float): shared FC에 적용할 Dropout 비율.
            include_motion (bool): True이면 motion head 계산, False이면 제외.
            use_relu_for_regression (bool): 회귀 태스크(TOT, ACT)에서 음수가 나오지 않도록 ReLU 적용 여부.
            
        입력:
            x: Tensor of shape (B, T, seq_input_dim)
            
        출력 (dict):
            "motion_logits": (B, T, num_motion_classes)  (include_motion=True 인 경우)
            "valence_logits": (B, num_valence_classes)
            "arousal_logits": (B, num_arousal_classes)
            "tot": (B,)
            "act": (B,)
        """
        super().__init__()
        self.include_motion = include_motion
        self.use_relu_for_regression = use_relu_for_regression
        
        # 1. Motion Head (frame 단위 분류)
        if include_motion:
            self.motion_head = nn.Linear(seq_input_dim, num_motion_classes)

        # 2. 시퀀스 전체 평균 풀링 후 shared feature extraction:
        # 입력 x: (B, T, seq_input_dim) → transpose → (B, seq_input_dim, T)
        # AdaptiveAvgPool1d: (B, seq_input_dim, T) → (B, seq_input_dim, 1)
        # squeeze: (B, seq_input_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.shared_fc = nn.Sequential(
            nn.Linear(seq_input_dim, pooled_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        
        # 3. 각 태스크별 헤드 (shared feature 사용)
        self.valence_head = nn.Linear(pooled_hidden_dim, num_valence_classes)
        self.arousal_head = nn.Linear(pooled_hidden_dim, num_arousal_classes)
        self.tot_head = nn.Linear(pooled_hidden_dim, 1)
        self.act_head = nn.Linear(pooled_hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, seq_input_dim) – Fusion 인코더의 출력.
        Returns:
            dict: {
                "motion_logits": (B, T, num_motion_classes)  (if include_motion is True),
                "valence_logits": (B, num_valence_classes),
                "arousal_logits": (B, num_arousal_classes),
                "tot": (B,),       # 회귀 예측 (Takeover Time)
                "act": (B,)        # 회귀 예측 (Anticipated Collision Time)
            }
        """
        out = {}
        if self.include_motion:
            # Motion head는 각 frame 별 분류 로그잇: (B, T, num_motion_classes)
            out["motion_logits"] = self.motion_head(x)
        
        # 전체 시퀀스 평균 풀링: (B, T, seq_input_dim) → (B, seq_input_dim)
        # x_transposed: (B, seq_input_dim, T)
        x_transposed = x.transpose(1, 2)
        pooled = self.pool(x_transposed).squeeze(-1)
        
        # Shared fully-connected layer: (B, seq_input_dim) → (B, pooled_hidden_dim)
        shared_feat = self.shared_fc(pooled)
        
        # 태스크별 예측
        out["valence_logits"] = self.valence_head(shared_feat)    # (B, num_valence_classes)
        out["arousal_logits"] = self.arousal_head(shared_feat)    # (B, num_arousal_classes)
        
        tot = self.tot_head(shared_feat).squeeze(-1)               # (B,)
        act = self.act_head(shared_feat).squeeze(-1)               # (B,)
        if self.use_relu_for_regression:
            tot = F.relu(tot)
            act = F.relu(act)
        out["tot"] = tot
        out["act"] = act
        return out

# ---------------------
# 간단한 테스트 코드
# ---------------------
if __name__ == "__main__":
    batch_size = 4
    seq_length = 50
    feature_dim = 256  # 예: Fusion encoder의 출력 feature 차원
    dummy_input = torch.randn(batch_size, seq_length, feature_dim)
    
    model = MultiTaskHead(seq_input_dim=feature_dim,
                          pooled_hidden_dim=128,
                          num_motion_classes=5,
                          num_valence_classes=3,
                          num_arousal_classes=3,
                          dropout_p=0.3,
                          include_motion=True,
                          use_relu_for_regression=True)
    
    output = model(dummy_input)
    for key, val in output.items():
        print(f"{key}: shape {val.shape}")
