# Champion Version: GatedTCN (v1) + Temporal Attention (v5) + LayerNorm (v6)
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- v5의 핵심: TemporalAttention ---
# (B, C, T) 입력을 받아 시간 축의 중요도를 학습합니다.
class TemporalAttention(nn.Module):
    def __init__(self, channels, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, 1, bias=False),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, channels, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)
        return x * w

# --- v1, v6의 아이디어를 결합한 새로운 TCN 블록 ---
class ChampionTCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.3):
        super().__init__()
        # v1: Gated Convolution
        self.conv = nn.Conv1d(in_ch, 2 * out_ch, kernel_size,
                              padding='same', dilation=dilation)
        # v6: LayerNorm (안정성)
        self.norm = nn.LayerNorm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x_conv = self.conv(x)
        
        # Gated Linear Unit
        x_act, x_gate = x_conv.chunk(2, dim=1)
        x = x_act * torch.sigmoid(x_gate)
        x = self.dropout(x)
        
        # LayerNorm을 위해 차원 변경: (B, C, T) -> (B, T, C)
        x_permuted = x.permute(0, 2, 1)
        x_norm = self.norm(x_permuted)
        # 다시 원래 차원으로 복원: (B, T, C) -> (B, C, T)
        x = x_norm.permute(0, 2, 1)
        
        return x + res

# --- 기존 파일 형식에 맞춘 메인 클래스들 ---

class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, dropout):
        super().__init__()
        self.conv_in = nn.Conv1d(dim, num_f_maps, 1)
        
        self.layers = nn.ModuleList([
            ChampionTCNBlock(num_f_maps, num_f_maps, kernel_size=5, dilation=2**i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.attentions = nn.ModuleList([
            TemporalAttention(num_f_maps, num_f_maps // 2) for _ in range(num_layers)
        ])
        
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_in(x)
        for i in range(len(self.layers)):
            f = self.layers[i](f)
            f = self.attentions[i](f)
        return self.conv_out(f)

class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, dropout):
        super().__init__()
        self.conv_in = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([
            ChampionTCNBlock(num_f_maps, num_f_maps, kernel_size=5, dilation=2**i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        return self.conv_out(out)

class MS_TCN2_PG(nn.Module):
    """
    [NEW] GatedTCN(v1), TemporalAttention(v5), LayerNorm(v6)을 결합한 챔피언 버전
    """
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, dropout=0.3):
        super().__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes, dropout)
        self.Rs = nn.ModuleList([
            copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes, num_classes, dropout))
            for _ in range(num_R)
        ])

    def forward(self, x):
        # 특징 추출기로 사용할 경우, softmax를 거치지 않은 .PG의 출력만 사용해야 합니다.
        # 예시: features = model.PG(x)
        out = self.PG(x)
        outputs = out.unsqueeze(0)
        for R in self.Rs:
            out = F.softmax(out, dim=1)
            out = R(out)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs