import torch
import torch.nn as nn
import torch.nn.functional as F

class GMFusionBlock(nn.Module):
    """
    Gated Modality Fusion Block.
    Args:
        input_dims (dict): mapping from modality name to input feature dimension
        hidden_dim (int): hidden dimension for gating network
        output_dim (int): output fused feature dimension
        dropout_p (float): dropout probability
    """
    def __init__(self, input_dims: dict, hidden_dim: int, output_dim: int, dropout_p: float = 0.1):
        super(GMFusionBlock, self).__init__()
        self.modalities = list(input_dims.keys())
        self.num_modalities = len(self.modalities)
        # Per-modality linear projection to common output_dim
        self.projections = nn.ModuleDict({
            mod: nn.Linear(input_dims[mod], output_dim)
            for mod in self.modalities
        })
        # Gating network: takes concatenated projected features
        concat_dim = output_dim * self.num_modalities
        self.gating = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, self.num_modalities)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.output_dim = output_dim

    def forward(self, inputs: dict):
        """
        Args:
            inputs (dict): mapping modality name -> tensor of shape (batch, seq_len, dim)
        Returns:
            Tensor of shape (batch, seq_len, output_dim)
        """
        # Project each modality
        projected = []
        for mod in self.modalities:
            x = inputs[mod]  # (B, T, D_in)
            p = self.projections[mod](x)  # (B, T, output_dim)
            projected.append(p)
        # Concatenate along last dim
        concat_feats = torch.cat(projected, dim=-1)  # (B, T, M * output_dim)
        # Compute gating weights
        gates = self.gating(concat_feats)  # (B, T, M)
        gates = self.softmax(gates)  # normalize across modalities
        # Weighted sum of projected features
        fused = 0
        for i, p in enumerate(projected):  # each p: (B, T, output_dim)
            w = gates[..., i].unsqueeze(-1)  # (B, T, 1)
            fused = fused + p * w if i > 0 else p * w
        return fused


class SMFusionBlock(nn.Module):
    """
    Self-Modality Fusion Block using cross-modality self-attention.
    Args:
        feature_dim (int): input feature dimension for all modalities (must be equal)
        num_modalities (int): number of modalities to fuse
        num_heads (int): number of attention heads
        dropout_p (float): dropout probability
    """
    def __init__(self, feature_dim: int, num_modalities: int, num_heads: int, dropout_p: float = 0.1):
        super(SMFusionBlock, self).__init__()
        self.num_modalities = num_modalities
        self.feature_dim = feature_dim
        # Multi-head attention to fuse across modalities
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout_p)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout_p)
        # Final projection back to feature_dim
        self.output_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, inputs: dict):
        """
        Args:
            inputs (dict): mapping modality name -> tensor of shape (batch, seq_len, feature_dim)
        Returns:
            Tensor of shape (batch, seq_len, feature_dim)
        """
        # Stack modalities into a sequence of length num_modalities
        # inputs_list shape: list of (B, T, D)
        inputs_list = [inputs[mod] for mod in sorted(inputs.keys())]
        x = torch.stack(inputs_list, dim=0)  # (M, B, T, D)
        M, B, T, D = x.shape
        # Merge time and batch for attention: treat each modality across all positions
        x_flat = x.permute(2, 1, 0, 3).reshape(T * B, M, D)  # (T*B, M, D)
        # Prepare for multihead: transpose to (M, T*B, D)
        x_attn = x_flat.transpose(0, 1)  # (M, T*B, D)
        # Self-attention across modalities
        attn_out, _ = self.multihead_attn(x_attn, x_attn, x_attn)  # (M, T*B, D)
        # Dropout & residual
        attn_out = self.dropout(attn_out)
        attn_out = self.layer_norm(attn_out + x_attn)
        # Transpose back and reshape
        attn_out = attn_out.transpose(0, 1).reshape(T, B, M, D)  # (T, B, M, D)
        # Fuse by averaging modalities
        fused = attn_out.mean(dim=2)  # (T, B, D)
        # Reshape back to (B, T, D)
        fused = fused.permute(1, 0, 2)
        # Final projection
        fused = self.output_proj(fused)
        return fused

# Example usage:
# inputs = {'imu': imu_feats, 'ppg': ppg_feats, 'veh': veh_feats}
# gm = GMFusionBlock(input_dims={'imu':32, 'ppg':16, 'veh':64}, hidden_dim=128, output_dim=64)
# fused_gm = gm(inputs)  # (B, T, 64)
# sm = SMFusionBlock(feature_dim=64, num_modalities=3, num_heads=4)
# # first project modalities to same dim 64
# projected_inputs = {mod: proj(f_in) for mod, (proj, f_in) in zip(inputs.keys(), gm.projections.items())}
# fused_sm = sm(projected_inputs)  # (B, T, 64)
