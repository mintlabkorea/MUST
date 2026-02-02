# Adaptive + SE + temporal atten
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------- Temporal Attention ----------------------
class TemporalAttention(nn.Module):
    """Simple temporal attention: produces weights over T and reweights features.
       Input:  (B, C, T)  -> Output: (B, C, T)
    """
    def __init__(self, channels, hidden=64):
        super().__init__()
        self.pool    = nn.AdaptiveAvgPool1d(1)  # to get global descriptor per channel
        self.fc_time = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.ReLU(),
            nn.Conv1d(hidden, channels, 1)
        )

    def forward(self, x):
        # x: (B,C,T)
        g = self.pool(x)                 # (B,C,1)
        w = self.fc_time(g)              # (B,C,1)
        w = torch.softmax(w, dim=-1)     # normalize over temporal dim (here only len 1, so broadcast)
        return x * w                     # broadcast to (B,C,T)

class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Sequential(
            nn.Conv1d(ch, ch // r, 1),
            nn.ReLU(),
            nn.Conv1d(ch // r, ch, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w

class AdaptiveDilatedConvSE(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, init_dilation=1, se_ratio=16, dropout=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_ch, in_ch, kernel_size))
        self.bias   = nn.Parameter(torch.zeros(out_ch))
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        self.dil = nn.Parameter(torch.tensor(float(init_dilation)))
        self.res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.conv1x1  = nn.Conv1d(out_ch, out_ch, 1)
        self.dropout  = nn.Dropout(dropout)
        self.se       = SEBlock(out_ch, r=se_ratio)

    def forward(self, x):
        d = max(1, int(torch.clamp(self.dil, min=1).item()))
        res = self.res_conv(x)
        out = F.conv1d(x, self.weight, self.bias, padding=d, dilation=d)
        out = F.relu(out)
        out = self.conv1x1(out)
        out = self.dropout(out)
        out = self.se(out)
        return res + out

class MS_TCN2_PG(nn.Module):
    """Adaptive dilation + SE + Temporal Attention"""
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes,
                 se_ratio=16, ta_hidden=64, dropout=0.0):
        super().__init__()
        self.PG = Prediction_Generation_SE_TA(num_layers_PG, num_f_maps, dim, num_classes,
                                              se_ratio, ta_hidden, dropout)
        self.Rs = nn.ModuleList([
            copy.deepcopy(Refinement_SE_TA(num_layers_R, num_f_maps, num_classes, num_classes,
                                           se_ratio, ta_hidden, dropout))
            for _ in range(num_R)
        ])

    def forward(self, x):
        out = self.PG(x)
        outputs = out.unsqueeze(0)
        for R in self.Rs:
            out = F.softmax(out, dim=1)
            out = R(out)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs

class Prediction_Generation_SE_TA(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, se_ratio, ta_hidden, dropout):
        super().__init__()
        self.conv_in = nn.Conv1d(dim, num_f_maps, 1)
        self.ad1 = nn.ModuleList([
            AdaptiveDilatedConvSE(num_f_maps, num_f_maps, kernel_size=3,
                                  init_dilation=2**(num_layers-1-i), se_ratio=se_ratio, dropout=dropout)
            for i in range(num_layers)
        ])
        self.ad2 = nn.ModuleList([
            AdaptiveDilatedConvSE(num_f_maps, num_f_maps, kernel_size=3,
                                  init_dilation=2**i, se_ratio=se_ratio, dropout=dropout)
            for i in range(num_layers)
        ])
        self.conv_fuse = nn.ModuleList([
            nn.Conv1d(2*num_f_maps, num_f_maps, 1)
            for _ in range(num_layers)
        ])
        self.ta_blocks = nn.ModuleList([
            TemporalAttention(num_f_maps, hidden=ta_hidden) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_in(x)
        for i in range(len(self.ad1)):
            f_in = f
            c1 = self.ad1[i](f)
            c2 = self.ad2[i](f)
            f = self.conv_fuse[i](torch.cat([c1, c2], dim=1))
            f = self.ta_blocks[i](f)
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in
        return self.conv_out(f)

class Refinement_SE_TA(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, se_ratio, ta_hidden, dropout):
        super().__init__()
        self.conv_in = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([
            AdaptiveDilatedConvSE(num_f_maps, num_f_maps, kernel_size=3,
                                   init_dilation=2**i, se_ratio=se_ratio, dropout=dropout)
            for i in range(num_layers)
        ])
        self.ta_blocks = nn.ModuleList([
            TemporalAttention(num_f_maps, hidden=ta_hidden) for _ in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_in(x)
        for i, layer in enumerate(self.layers):
            out = layer(out)
            out = self.ta_blocks[i](out)
        return self.conv_out(out)