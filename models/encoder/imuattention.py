import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import Config

class ImuAttentionEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        # IMU 채널 수 (예: Config에서 imu_params['input_dim'] = 14)
        in_ch = cfg.imu_params['input_dim']  
        # 예시: CNN 채널 구성
        cnn_channels = [32, 64]  
        cnn_layers = []
        for out_ch in cnn_channels:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.ReLU(),
                # (선택) pooling 대신 그대로 시퀀스를 살려두고 싶으면 여기를 주석 처리
                # nn.MaxPool1d(kernel_size=2, stride=2)
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM 설정
        hidden = cfg.hidden    # 예: 128
        num_layers = 1         # 또는 cfg.imu_params에 따라
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],  
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        # Attention  
        self.attn = nn.Linear(hidden, 1, bias=False)

        # 최종 projection (원하는 출력 차원으로)
        self.proj = nn.Linear(hidden, cfg.num_motion)

    def forward(self, imu_seq):
        """
        imu_seq: (B, T, 14)  또는 (B, 14, T) 형태로 들어올 수 있음.
        여기선 (B, T, 14)라고 가정하고 내부에서 permute.
        """
        if imu_seq.dim() == 3 and imu_seq.size(1) != self.cnn[0].in_channels:
            # (B, T, 14) → (B, 14, T)
            x = imu_seq.permute(0, 2, 1)
        else:
            x = imu_seq  # (B, 14, T)

        # CNN → (B, C_last, T')   (만약 pooling을 쓰지 않았다면 T'=T, 
        # pooling이 있으면 T' = T/2 등으로 줄어듦)
        x = self.cnn(x)           

        # (B, C_last, T') → (B, T', C_last)
        x = x.permute(0, 2, 1)     

        # LSTM → (B, T', hidden)
        x, _ = self.lstm(x)

        # Attention 스코어 → (B, T', 1)
        scores = self.attn(x)     # (B, T', 1)
        weights = torch.softmax(scores, dim=1)  # (B, T', 1)

        # Attention 가중합 → (B, hidden)
        x_attn = (x * weights).sum(dim=1)  

        # Projection → (B, num_motion)
        logits = self.proj(x_attn)  
        return logits
