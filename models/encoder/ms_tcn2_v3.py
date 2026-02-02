# V0에서 Adaptive Dilation 적용
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveDilatedConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, init_dilation=1):
        super().__init__()
        # weight and bias for conv
        self.weight = nn.Parameter(
            torch.Tensor(out_ch, in_ch, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_ch))
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        # learnable dilation scalar
        self.dil = nn.Parameter(torch.tensor(float(init_dilation)))
        # 1x1 for residual connection if needed
        self.res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.conv1x1 = nn.Conv1d(out_ch, out_ch, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        # compute integer dilation
        d = max(1, int(torch.clamp(self.dil, min=1).item()))
        padding = d
        # residual path
        res = self.res_conv(x)
        # dynamic dilated convolution
        out = F.conv1d(x, self.weight, self.bias,
                       padding=padding,
                       dilation=d)
        out = F.relu(out)
        # 1x1 projection + dropout
        out = self.conv1x1(out)
        out = self.dropout(out)
        return res + out

class MS_TCN2_PG(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        super().__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList([
            copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes, num_classes))
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
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super().__init__()
        self.conv_in = nn.Conv1d(dim, num_f_maps, 1)
        # adaptive dilated conv lists
        self.ad1 = nn.ModuleList([
            AdaptiveDilatedConv(num_f_maps, num_f_maps, kernel_size=3,
                                 init_dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ])
        self.ad2 = nn.ModuleList([
            AdaptiveDilatedConv(num_f_maps, num_f_maps, kernel_size=3,
                                 init_dilation=2**i)
            for i in range(num_layers)
        ])
        self.conv_fuse = nn.ModuleList([
            nn.Conv1d(2*num_f_maps, num_f_maps, 1)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout()
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
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super().__init__()
        self.conv_in = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([
            AdaptiveDilatedConv(num_f_maps, num_f_maps, kernel_size=3,
                                 init_dilation=2**i)
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        return self.conv_out(out)
