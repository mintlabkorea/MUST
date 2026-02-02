# models/head/emotion_head.py

import torch.nn as nn

class EmotionHead(nn.Module):
    """
    융합된 특징 벡터(fused embedding)를 입력받아
    Valence와 Arousal의 로짓(logits)을 각각 출력합니다.
    """
    def __init__(self, cfg):
        super().__init__()
        self.valence_classifier = nn.Linear(cfg.hidden, cfg.num_valence)
        self.arousal_classifier = nn.Linear(cfg.hidden, cfg.num_arousal)

    def forward(self, fused_embedding):
        """
        Args:
          fused_embedding (torch.Tensor): (B, H) 형태의 융합된 특징 벡터
        
        Returns:
          dict: 'valence_logits'와 'arousal_logits'를 포함하는 딕셔너리
        """
        valence_logits = self.valence_classifier(fused_embedding)
        arousal_logits = self.arousal_classifier(fused_embedding)
        
        return {
            'valence_logits': valence_logits,
            'arousal_logits': arousal_logits
        }