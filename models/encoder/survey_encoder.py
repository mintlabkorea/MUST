import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.utils.data import Dataset

import os
import pandas as pd


class PreSurveyDataset(Dataset):
    """
    Loads pre-survey CSV, returns only the survey feature tensors.
    Skips the first row entirely, drops the first column (subject ID),
    and uses columns 2..end as features.
    """
    def __init__(self, csv_path: str):
        super().__init__()
        # ——— Config 전체가 넘어왔을 때도 처리해주기 ———
        if not isinstance(params, dict):
            # Config.Encoders.imu 에 정의된 파라미터 딕셔너리를 꺼냅니다.
            params = params.Encoders.survey
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Cannot find file: {csv_path}")
        # 1) 첫 번째 행 건너뛰기, 2) 전체 읽은 뒤 첫 번째 컬럼(subject ID) 제외
        df = pd.read_csv(csv_path, skiprows=1, header=None)
        # 두 번째 컬럼부터 끝까지 사용
        self.features = df.iloc[:, 1:].values.astype('float32')
        self.input_dim = self.features.shape[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32)
    
class PreSurveyEncoder(nn.Module):
    """
    Parametric survey encoder via Config.
    """
    def __init__(self, params):
        super().__init__()
        p = params
        prev = p['input_dim']
        layers = []
        
        for h in p['hidden_dims']:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.LayerNorm(h),
                nn.Dropout(p['dropout'])
            ]
            prev = h
        layers.append(nn.Linear(prev, p['embed_dim']))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, mask=None):

        return self.mlp(x)
    
class StaticFeatureAttention(nn.Module):
    """
    정적 피처 벡터의 각 요소에 어텐션 가중치를 학습하여
    중요한 피처를 강조하는 모듈.
    """
    def __init__(self, feature_dim, attention_dim=64):
        super().__init__()
        # 어텐션 스코어를 계산하기 위한 작은 신경망
        self.attention_mlp = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, feature_dim) # 각 피처별 스코어를 출력
        )

    def forward(self, x):
        # x shape: (batch_size, feature_dim)
        
        # 1. 각 피처의 중요도 점수(score)를 계산합니다.
        attention_scores = self.attention_mlp(x)
        
        # 2. Softmax를 적용하여 합이 1인 확률 가중치로 변환합니다.
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 3. 원본 피처에 학습된 가중치를 곱하여 중요한 피처를 강조합니다.
        weighted_features = x * attention_weights
        
        # 가중치를 함께 반환하여 나중에 분석용으로 사용할 수 있습니다.
        return weighted_features, attention_weights
