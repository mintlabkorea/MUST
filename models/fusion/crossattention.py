import torch
import torch.nn as nn

class CrossAttentionFusionBlock(nn.Module):
    """
    Asymmetrical Cross-Attention Fusion Block.
    Emotion features (Query) attend to Motion features (Key, Value).
    """
    def __init__(self, feature_dim: int, num_heads: int, post_fusion_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Cross-Attention 레이어
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True  # (B, T, D) 입력을 바로 사용
        )
        
        # 트랜스포머 블록의 표준 구성 요소
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.2)
        
        # 융합 후 최종 특징을 만드는 MLP
        self.post_fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, post_fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, motion_feat, emotion_feat):
        """
        Args:
            motion_feat (Tensor): (B, T, D) - Key, Value 역할
            emotion_feat (Tensor): (B, T, D) - Query 역할
        Returns:
            Tensor of shape (B, T, post_fusion_dim)
        """
        # 1. 감정(Query)이 모션(Key, Value)을 참조하여 어텐션 수행
        # attn_output은 모션 컨텍스트가 반영된 새로운 감정 특징
        attn_output, _ = self.cross_attention(
            query=emotion_feat, 
            key=motion_feat, 
            value=motion_feat
        )
        
        # 2. 잔차 연결(Residual Connection) 및 Layer Normalization
        # Query(emotion_feat)에 어텐션 결과를 더해줌
        contextual_emotion_feat = self.layer_norm(emotion_feat + self.dropout(attn_output))
        
        # 3. 원본 모션 특징과 컨텍스트가 추가된 감정 특징을 결합
        concatenated_feat = torch.cat([motion_feat, contextual_emotion_feat], dim=-1)
        
        # 4. 최종 융합 특징 생성
        fused_feat = self.post_fusion_mlp(concatenated_feat)
        
        return fused_feat
    
class BidirectionalCrossAttentionBlock(nn.Module):
    """
    Bidirectional Cross-Attention Fusion.
    - Path 1: Emotion queries Motion.
    - Path 2: Motion queries Emotion.
    """
    def __init__(self, feature_dim: int, num_heads: int, post_fusion_dim: int):
        super().__init__()
        
        # 경로 1: Emotion -> Motion 어텐션
        self.attn_e_on_m = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.norm_e = nn.LayerNorm(feature_dim)

        # 경로 2: Motion -> Emotion 어텐션
        self.attn_m_on_e = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.norm_m = nn.LayerNorm(feature_dim)
        
        self.dropout = nn.Dropout(0.2)
        
        # 최종 융합 MLP
        self.post_fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, post_fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, motion_feat, emotion_feat):
        """
        Args:
            motion_feat (Tensor): (B, T, D)
            emotion_feat (Tensor): (B, T, D)
        Returns:
            Tensor of shape (B, T, post_fusion_dim)
        """
        # --- 경로 1: Emotion이 Motion을 참조 ---
        attn_out_e, _ = self.attn_e_on_m(query=emotion_feat, key=motion_feat, value=motion_feat)
        contextual_emotion = self.norm_e(emotion_feat + self.dropout(attn_out_e))
        
        # --- 경로 2: Motion이 Emotion을 참조 ---
        attn_out_m, _ = self.attn_m_on_e(query=motion_feat, key=emotion_feat, value=emotion_feat)
        contextual_motion = self.norm_m(motion_feat + self.dropout(attn_out_m))
        
        # --- 결합 ---
        concatenated_feat = torch.cat([contextual_motion, contextual_emotion], dim=-1)
        fused_feat = self.post_fusion_mlp(concatenated_feat)
        
        return fused_feat