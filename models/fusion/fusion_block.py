# models/fusion/fusion_blocks.py

import torch
import torch.nn as nn

class TaskSpecificFusion(nn.Module):
    """
    하나의 모달리티(Query)가 여러 다른 모달리티(Context)를 참조하여
    특징을 풍부하게 만드는 Cross-Attention 블록입니다.
    """
    def __init__(self, query_modality, context_modalities, feature_dim, num_heads, output_dim, dropout_p=0.1, use_residual=True):
        super().__init__()
        self.query_modality = query_modality
        self.context_modalities = context_modalities
        self.use_residual = use_residual
        self.dropout = nn.Dropout(dropout_p)
        
        self.attn_blocks = nn.ModuleDict({
            modality: nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
            for modality in context_modalities
        })
        self.norm = nn.LayerNorm(len(context_modalities) * feature_dim)
        self.fc = nn.Linear(len(context_modalities) * feature_dim, output_dim)
        
        if output_dim != feature_dim:
            self.query_res_proj = nn.Linear(feature_dim, output_dim)
        else:
            self.query_res_proj = nn.Identity()
        
    def forward(self, feature_dict):
        query = feature_dict[self.query_modality]
        attended_features = []
        for modality in self.context_modalities:
            context = feature_dict[modality]
            attn_out, _ = self.attn_blocks[modality](query, context, context)
            attended_features.append(self.dropout(attn_out))

        concat_features = torch.cat(attended_features, dim=-1)
        fused = self.fc(self.norm(concat_features))
        
        if self.use_residual:
            fused = fused + self.query_res_proj(query)
            
        return fused

class SMFusionBlock(nn.Module):
    """
    보조 손실(Auxiliary Loss) 계산을 위한 Self-Multimodal fusion.
    학습 가능한 가중치(gamma)로 각 모달리티의 중요도를 계산하여 가중합합니다.
    """
    def __init__(self, modalities, hidden_dim):
        super().__init__()
        self.modalities = modalities
        self.gamma = nn.Parameter(torch.randn(len(modalities), hidden_dim))

    def forward(self, features): # features: dict modality -> (B, D)
        scores = torch.stack([
            (features[m] * self.gamma[i]).sum(-1) for i, m in enumerate(self.modalities)
        ], dim=1)
        alpha = torch.softmax(scores, dim=1)
        
        feat_stack = torch.stack([features[m] for m in self.modalities], dim=1)
        X_aux = (alpha.unsqueeze(-1) * feat_stack).sum(1)
        return X_aux

class GMFusionBlock(nn.Module):
    """
    Guided cross-attention. 
    SMFusion의 결과(X_aux)를 Query로 사용하여 모든 모달리티 특징을 요약합니다.
    """
    def __init__(self, modalities, hidden_dim, num_heads):
        super().__init__()
        self.modalities = modalities
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X_aux, features):  
        Q = X_aux.unsqueeze(1) # (B, D) -> (B, 1, D) as query
        KV = torch.stack([features[m] for m in self.modalities], dim=1) # (B, M, D)
        
        attn_out, _ = self.attn(Q, KV, KV) # (B, 1, D)
        return self.out(attn_out).squeeze(1) # (B, D)