# V0 -> GatedTCN
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedTCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super(GatedTCNBlock, self).__init__()
        # conv outputs 2*out_ch for gating
        self.conv = nn.Conv1d(in_ch, 2*out_ch, kernel_size,
                              padding=(kernel_size-1)//2 * dilation,
                              dilation=dilation)
        self.norm = nn.BatchNorm1d(2*out_ch)
        # 1x1 for matching dims if needed
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        conv_out = self.norm(self.conv(x))  # (B,2*out_ch,T)
        a, b = conv_out.chunk(2, dim=1)
        gated = a * torch.sigmoid(b)
        # residual path
        res = self.residual(x)
        return res + gated

class MS_TCN2_PG(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        super(MS_TCN2_PG, self).__init__()
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
        super(Prediction_Generation, self).__init__()
        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)
        # -- replace basic convs with GatedTCNBlock --
        self.gated1 = nn.ModuleList([
            GatedTCNBlock(num_f_maps, num_f_maps, dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ])
        self.gated2 = nn.ModuleList([
            GatedTCNBlock(num_f_maps, num_f_maps, dilation=2**i)
            for i in range(num_layers)
        ])
        self.conv_fusion = nn.ModuleList([
            nn.Conv1d(2 * num_f_maps, num_f_maps, 1)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)
        for i in range(len(self.gated1)):
            f_in = f
            c1 = self.gated1[i](f)
            c2 = self.gated2[i](f)
            f = self.conv_fusion[i](torch.cat([c1, c2], dim=1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in
        out = self.conv_out(f)
        return out

class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        # -- use GatedTCNBlock for dilated residual --
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([
            copy.deepcopy(
                GatedTCNBlock(num_f_maps, num_f_maps, dilation=2**i)
            ) for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3,
                                      padding=dilation,
                                      dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out