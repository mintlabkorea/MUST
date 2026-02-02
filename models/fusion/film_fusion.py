# mumu_fusion.py 또는 fusion_blocks.py 에 추가

import torch
import torch.nn as nn

class FiLMFusionBlock(nn.Module):
    """
    FiLM: Feature-wise Linear Modulation
    Motion features generate gamma and beta to modulate emotion features.
    """
    def __init__(self, feature_dim: int, post_fusion_dim: int):
        super().__init__()
        # Motion 특징을 입력받아 gamma와 beta를 생성하는 MLP
        # gamma와 beta가 각각 feature_dim 크기를 가지므로 출력은 feature_dim * 2
        self.film_generator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )
        
        # 융합 후 특징을 최종 차원으로 매핑하는 MLP
        # (원본 모션 특징 + 변조된 감정 특징)을 concat하므로 입력은 feature_dim * 2
        self.post_fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, post_fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, motion_feat, emotion_feat):
        """
        Args:
            motion_feat (Tensor): (B, T, D) - 컨트롤러 역할
            emotion_feat (Tensor): (B, T, D) - 조절될 신호
        Returns:
            Tensor of shape (B, T, post_fusion_dim)
        """
        # 1. Motion 특징으로 gamma, beta 생성
        film_params = self.film_generator(motion_feat)  # (B, T, D * 2)
        
        # 2. gamma와 beta 분리
        gamma, beta = torch.chunk(film_params, 2, dim=-1)  # 각각 (B, T, D)
        
        # 3. 감정 특징을 조절 (Modulation)
        modulated_emotion_feat = gamma * emotion_feat + beta
        
        # 4. 원본 모션 특징과 조절된 감정 특징을 결합
        concatenated_feat = torch.cat([motion_feat, modulated_emotion_feat], dim=-1)
        
        # 5. 최종 융합 특징 생성
        fused_feat = self.post_fusion_mlp(concatenated_feat)
        
        return fused_feat
    
class SymmetricalFiLMFusionBlock(nn.Module):
    """
    Symmetrical FiLM Fusion.
    - Motion modulates Emotion
    - Emotion modulates Motion
    The two modulated features are then concatenated and fused.
    """
    def __init__(self, feature_dim: int, post_fusion_dim: int):
        super().__init__()
        
        # 1. Motion이 Emotion을 조절하기 위한 FiLM 생성기
        self.film_gen_m_on_e = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )
        
        # 2. Emotion이 Motion을 조절하기 위한 FiLM 생성기
        self.film_gen_e_on_m = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )
        
        # 3. 양방향으로 조절된 두 특징을 최종 융합하기 위한 MLP
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
        # --- Path 1: Motion -> Emotion ---
        film_params_m = self.film_gen_m_on_e(motion_feat)
        gamma_m, beta_m = torch.chunk(film_params_m, 2, dim=-1)
        modulated_emotion = gamma_m * emotion_feat + beta_m
        
        # --- Path 2: Emotion -> Motion ---
        film_params_e = self.film_gen_e_on_m(emotion_feat)
        gamma_e, beta_e = torch.chunk(film_params_e, 2, dim=-1)
        modulated_motion = gamma_e * motion_feat + beta_e
        
        # --- Combine ---
        concatenated_feat = torch.cat([modulated_motion, modulated_emotion], dim=-1)
        fused_feat = self.post_fusion_mlp(concatenated_feat)
        
        return fused_feat