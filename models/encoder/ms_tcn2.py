# V3 + Layernorm/DropBlock
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── DropBlock1D ───────────────────────────────────────────────
class DropBlock1D(nn.Module):
    def __init__(self, block_size=5, drop_prob=0.2): #(5~9, 0.05~0.2)
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        B, C, T = x.shape
        gamma = self.drop_prob * T / self.block_size
        mask = (torch.rand(B, 1, T, device=x.device) < gamma).float()
        mask = F.max_pool1d(mask, kernel_size=self.block_size, stride=1,
                            padding=self.block_size//2)
        keep = 1 - mask
        return x * keep * (keep.numel() / keep.sum().clamp(min=1.))

# ── Adaptive Dilated Conv + LN + DropBlock ────────────────────
class AdaptiveDilatedConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, init_dilation=1,
                 dropblock_size=5, drop_prob=0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_ch, in_ch, kernel_size))
        self.bias   = nn.Parameter(torch.zeros(out_ch))
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')

        self.dil = nn.Parameter(torch.tensor(float(init_dilation)))
        self.res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.proj1x1  = nn.Conv1d(out_ch, out_ch, 1)

        # 정규화 & 드롭
        self.ln        = nn.LayerNorm(out_ch)          # 채널 기준 (B,C,T) → LN over C
        self.dropblock = DropBlock1D(block_size=dropblock_size, drop_prob=drop_prob)

    def forward(self, x):
        d = max(1, int(torch.clamp(self.dil, min=1).item()))
        res = self.res_conv(x)

        out = F.conv1d(x, self.weight, self.bias, padding=d, dilation=d)
        out = F.relu(out)
        out = self.proj1x1(out)

        # LayerNorm expects (B,T,C) so transpose twice
        out = out.transpose(1, 2)          # (B,T,C)
        out = self.ln(out)
        out = out.transpose(1, 2)          # (B,C,T)

        out = self.dropblock(out)
        return res + out

# ── MS-TCN2-PG with Adaptive + LN/DropBlock ───────────────────
class MS_TCN2_PG(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes,
                 dropblock_size=5, drop_prob=0.2):
        super().__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes,
                                        dropblock_size, drop_prob)
        self.Rs = nn.ModuleList([
            copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes, num_classes,
                                     dropblock_size, drop_prob))
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

class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes,
                 dropblock_size, drop_prob):
        super().__init__()
        self.conv_in = nn.Conv1d(dim, num_f_maps, 1)
        self.ad1 = nn.ModuleList([
            AdaptiveDilatedConv(num_f_maps, num_f_maps, 3,
                                init_dilation=2**(num_layers-1-i),
                                dropblock_size=dropblock_size,
                                drop_prob=drop_prob)
            for i in range(num_layers)
        ])
        self.ad2 = nn.ModuleList([
            AdaptiveDilatedConv(num_f_maps, num_f_maps, 3,
                                init_dilation=2**i,
                                dropblock_size=dropblock_size,
                                drop_prob=drop_prob)
            for i in range(num_layers)
        ])
        self.conv_fuse = nn.ModuleList([
            nn.Conv1d(2*num_f_maps, num_f_maps, 1)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(drop_prob)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_in(x)
        for i in range(len(self.ad1)):
            f_in = f
            c1 = self.ad1[i](f)
            c2 = self.ad2[i](f)
            f = self.conv_fuse[i](torch.cat([c1, c2], dim=1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in
        return self.conv_out(f)

class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes,
                 dropblock_size, drop_prob):
        super().__init__()
        self.conv_in = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([
            AdaptiveDilatedConv(num_f_maps, num_f_maps, 3,
                                init_dilation=2**i,
                                dropblock_size=dropblock_size,
                                drop_prob=drop_prob)
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        return self.conv_out(out)
