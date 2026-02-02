import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fusion.fusion_block import TaskSpecificFusion

class MotionPredictor(nn.Module):
    def __init__(self, feature_dim=256, output_dim=132):
        super().__init__()
        self.fusion = TaskSpecificFusion(
            query_modality="imu",
            context_modalities=["imu","vehicle","scenario","survey"],
            feature_dim=feature_dim,
            num_heads=4,
            output_dim=feature_dim
        )
        self.head = nn.Linear(feature_dim, output_dim)

    def forward(self, feature_dict, return_feature=False):
        """
        Args:
            feature_dict: modalities → (B,1,D)
            return_feature: True면 (mot_repr, mot_logits), False면 mot_logits
        """
        mot_repr = self.fusion(feature_dict).squeeze(1)  # (B, feature_dim)
        mot_logits = self.head(mot_repr)                # (B, num_motion)
        if return_feature:
            return mot_repr, mot_logits
        else:
            return mot_logits

