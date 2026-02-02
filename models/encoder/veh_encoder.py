import os, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config.config import Config

# =======================================
# 1) TCN 기반 인코더 (파라미터화)
# =======================================
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size, dropout):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=dilation,
            dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.conv_dilated(x)
        out = F.relu(out)
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out  # Residual

class VehicleTCNEncoder(nn.Module):
    """
    차량 데이터(시계열)를 TCN으로 임베딩 추출
      - 입력:  (B, input_dim, T)
      - 출력:  (B, embed_dim, T)
    """
    def __init__(self, params):
        super().__init__()
        # ——— Config 전체가 넘어왔을 때도 처리해주기 ———
        if not isinstance(params, dict):
            # Config.Encoders.imu 에 정의된 파라미터 딕셔너리를 꺼냅니다.
            params = params.Encoders.veh
        p=params
        # config에서 embed_dim 값을 읽어오고, 없으면 기본값 64를 사용
        embed_dim = p.get('embed_dim', 64)
        num_channels = p.get('num_channels', 64)


        self.conv_in = nn.Conv1d(
            in_channels=p['input_dim'],
            out_channels=p['num_channels'], # embed_dim이 아닌 num_channels로 초기 변환
            kernel_size=p['kernel_size'],
            padding=p['padding']
        )
        # Dilated Residual Layers
        self.layers = nn.ModuleList([
            DilatedResidualLayer(
                dilation=2**i,
                in_channels=p['num_channels'],
                out_channels=p['num_channels'],
                kernel_size=p['kernel_size'],
                dropout=p['dropout']
            ) for i in range(p['num_layers'])
        ])

        # 출력 프로젝션
        self.conv_out = nn.Conv1d(num_channels, embed_dim, kernel_size=1)

    def forward(self, x, input_lengths=None, return_pooled=False, mask=None):
        """
        Args:
            x: (B, input_dim, T)
            return_pooled: True면 전체 시퀀스를 평균내어 (B, embed_dim) 반환
            mask: optional, (B, T) boolean mask
        Returns:
            if return_pooled=False: (B, embed_dim, T)
            if return_pooled=True:  (B, embed_dim)
        """
        if x.shape[1] != self.conv_in.in_channels:
           x = x.transpose(1, 2)

        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)  # → (B, embed_dim, T)

        if return_pooled:
            if mask is not None:
                # 1. 패딩된 위치의 값을 0으로 만듭니다.
                masked_out = out * mask.unsqueeze(1).float()
                # 2. 유효한 값들만 더합니다.
                summed_out = masked_out.sum(dim=-1)
                # 3. 유효한 타임스텝의 개수를 세고, 0으로 나누는 것을 방지합니다.
                num_valid_steps = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                # 4. 합계를 개수로 나누어 평균을 계산합니다.
                out = summed_out / num_valid_steps
            else:
                out = F.adaptive_avg_pool1d(out, 1).squeeze(-1)  # (B, embed_dim)

        return out

# ======================================(레이블 X, 단순 임베딩용)
# =======================================
class VehicleEmbedDataset(Dataset):
    def __init__(self, base_dir):
        super().__init__()
        pattern = os.path.join(base_dir, '**', 'processed_carla.csv')
        self.file_paths = glob.glob(pattern, recursive=True)
        self.feature_cols = [
            "timestamps", "mode", "alarmtime", 
            "speed", "acceleration", "steer", "brake", "yaw",
            "veh1_role", "veh1_distance",
            "veh2_role", "veh2_distance",
            "veh3_role", "veh3_distance", "distance",
            "collision", "laneoffset"
        ]
        self.sequences = []
        self._load_data()

    def _load_data(self):
        for csv_path in self.file_paths:
            df = pd.read_csv(csv_path)
            df_features = df[self.feature_cols]
            self.sequences.append(df_features.values.astype(np.float32))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

# =======================================
# 3) collate_fn (패딩)
# =======================================
def collate_fn(batch):
    lengths = [arr.shape[0] for arr in batch]
    max_len = max(lengths)
    dim = batch[0].shape[1]

    pads, masks = [], []
    for arr in batch:
        T = arr.shape[0]
        pad = np.zeros((max_len, dim), dtype=np.float32)
        pad[:T] = arr
        pads.append(torch.tensor(pad))
        masks.append(torch.tensor([1]*T + [0]*(max_len-T)))

    padded = torch.stack(pads).permute(0,2,1)  # (B, D, T)
    mask = torch.stack(masks)                 # (B, T)
    return padded, mask

# =======================================
# 4) 임베딩 추출 스크립트
# =======================================
def extract_vehicle_embeddings(cfg):
    device = cfg.Project.device
    dataset = VehicleEmbedDataset("/path/to/your/data_directory")
    loader = DataLoader(dataset, batch_size=cfg.Data.batch_size, shuffle=False, collate_fn=collate_fn)


    model = VehicleTCNEncoder(cfg).to(device)
    model.eval()

    all_emb = []
    with torch.no_grad():
        for feats, mask in loader:
            feats = feats.to(device)
            emb = model(feats)  # (B, embed_dim, T)
            all_emb.extend(emb.cpu().numpy())

    return all_emb

if __name__ == '__main__':
    cfg = Config()
    embeddings = extract_vehicle_embeddings(cfg)
    # np.save(cfg.output_path, np.array(embeddings, dtype=object), allow_pickle=True)