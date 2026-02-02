import torch
import torch.nn as nn
from config.config import Config
from models.encoder.ms_tcn2 import MS_TCN2_PG

class PPGEncoder(nn.Module):
    """
    PPGEncoder using MS-TCN2 with Parallel Guidance as backbone.
    Input: (B, T) or (B, N, T)
    Output: (B, f_maps)
    """
    def __init__(self, params):
        super().__init__()
        # ——— Config 전체가 넘어왔을 때도 처리해주기 ———
        if not isinstance(params, dict):
            # Config.Encoders.imu 에 정의된 파라미터 딕셔너리를 꺼냅니다.
            params = params.Encoders.ppg
        p=params
        # Fetch MS-TCN2 params with defaults to avoid KeyError (3/2/2 or 4/2/3)
        num_layers_PG = p.get('num_layers_PG', 4)
        num_layers_R  = p.get('num_layers_R', 2)
        num_R         = p.get('num_R', 3)
        num_f_maps    = p.get('num_f_maps', p.get('embed_dim', 64))
        dropout_rate = p.get('dropout', 0.2)
        self.mstcn = MS_TCN2_PG(
            num_layers_PG=num_layers_PG, num_layers_R=num_layers_R,
            num_R=num_R, num_f_maps=num_f_maps,
            dim=1, num_classes=num_f_maps, drop_prob=dropout_rate 
        )
 
        # Global temporal pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, seq, return_pooled=True): 
        # 다양한 입력 형태 (B, T), (B, T, 1), (B, 1, T)를
        # Conv1d가 요구하는 (B, 1, T) 형태로 정규화합니다.
        if seq.dim() == 2:
            # Case 1: (B, T) -> (B, 1, T)
            x = seq.unsqueeze(1)
        elif seq.dim() == 3 and seq.shape[2] == 1:
            # Case 2: (B, T, 1) -> (B, 1, T)
            # 에러를 발생시키는 주된 원인입니다. 채널과 길이를 transpose합니다.
            x = seq.transpose(1, 2)
        else:
            # Case 3: (B, 1, T) 형태는 그대로 사용합니다.
            x = seq
        # =================================================================

        # 이제 x는 (B, 1, T) 형태임이 보장됩니다.
        out = self.mstcn(x)      # (stages, B, f_maps, T)
        last_stage_out = out[-1] # (B, f_maps, T)

        if return_pooled:
            pooled = self.global_pool(last_stage_out).squeeze(-1)  # (B, f_maps)
            return pooled
        else:
            # pooling 없이 시계열 특징 그대로 반환
            return last_stage_out # (B, f_maps, T)
        
# class PPGEncoder(nn.Module):
#     """
#     PPGEncoder using MS-TCN2 with Parallel Guidance as backbone.
#     Input: (B, T) or (B, N, T)
#     Output: (B, f_maps)
#     """
#     def __init__(self, params):
#         super().__init__()
#         # ——— Config 전체가 넘어왔을 때도 처리해주기 ———
#         if not isinstance(params, dict):
#             # Config.Encoders.imu 에 정의된 파라미터 딕셔너리를 꺼냅니다.
#             params = params.Encoders.ppg
#         p=params
#         # Fetch MS-TCN2 params with defaults to avoid KeyError (3/2/2 or 4/2/3)
#         num_layers_PG = p.get('num_layers_PG', 4)
#         num_layers_R  = p.get('num_layers_R', 2)
#         num_R         = p.get('num_R', 3)
#         # Default f_maps to embed_dim or last cnn channel if unspecified
#         num_f_maps    = p.get('num_f_maps', p.get('embed_dim', p.get('cnn_channels', [])[-1] if p.get('cnn_channels') else 64))
#         # MS-TCN2_PG expects input channels = 1, num_classes = f_maps
#         self.mstcn = MS_TCN2_PG(
#             num_layers_PG=num_layers_PG,
#             num_layers_R=num_layers_R,
#             num_R=num_R,
#             num_f_maps=num_f_maps,
#             dim=1,
#             num_classes=num_f_maps
#         )
#         # Global temporal pooling
#         self.global_pool = nn.AdaptiveAvgPool1d(1)

#     def forward(self, seq, *args, **kwargs):
#         # seq: Tensor of shape (B, T) or (B, N, T)
#         if seq.dim() == 3:
#             B, N, T = seq.shape
#             x = seq.view(B * N, T).unsqueeze(1)  # (B*N, 1, T)
#             out = self.mstcn(x)                      # (stages, B*N, f_maps, T)
#             last = out[-1]                           # (B*N, f_maps, T)
#             pooled = self.global_pool(last).squeeze(-1)  # (B*N, f_maps)
#             pooled = pooled.view(B, N, -1).mean(dim=1)
#         else:
#             x = seq.unsqueeze(1)                # (B, 1, T)
#             out = self.mstcn(x)                     # (stages, B, f_maps, T)
#             last = out[-1]                          # (B, f_maps, T)
#             pooled = self.global_pool(last).squeeze(-1)  # (B, f_maps)
#         return pooled  # (B, f_maps)
