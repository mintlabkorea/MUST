#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# =============================================================================
# 1. 유틸 함수 및 기본 모듈
# =============================================================================

class Identity(nn.Module):
    """
    입력 x와 lengths를 그대로 반환.
    conv subsampling 없이 전체 시퀀스를 그대로 유지.
    """
    def forward(self, x, lengths):
        return x, lengths

def xavier_linear(in_features, out_features, bias=True):
    layer = nn.Linear(in_features, out_features, bias=bias)
    nn.init.xavier_uniform_(layer.weight)
    if bias:
        nn.init.zeros_(layer.bias)
    return layer

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class GLU(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        a, b = x.chunk(2, dim=self.dim)
        return a * torch.sigmoid(b)

class ResidualConnectionModule(nn.Module):
    """
    Residual Connection.
    출력 = module(x) * module_factor + x * input_factor
    """
    def __init__(self, module, module_factor=1.0, input_factor=1.0):
        super().__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, x):
        return (self.module(x) * self.module_factor) + (x * self.input_factor)

# =============================================================================
# 2. Conformer 모듈 
# =============================================================================

class ConformerConvModule(nn.Module):
    """
    Conformer Convolution Module.
    입력: (B, T, in_channels) → 1D Conv → 출력: (B, T, in_channels)
    """
    def __init__(self, in_channels, kernel_size, expansion_factor, dropout_p):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "Kernel size must be odd."
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * expansion_factor, kernel_size=1),
            GLU(dim=1),
            nn.Conv1d(in_channels, in_channels, kernel_size,
                      padding=(kernel_size - 1)//2,
                      groups=in_channels),
            nn.BatchNorm1d(in_channels),
            Swish(),
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.Dropout(p=dropout_p)
        )
    def forward(self, x):
        # x: (B, T, in_channels)
        x = x.transpose(1,2)   # (B, in_channels, T)
        x = self.sequential(x)
        x = x.transpose(1,2)   # (B, T, in_channels)
        return x

class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, d_model, num_heads, dropout_p):
        super().__init__()
        self.scaling_factor = (d_model // num_heads) ** -0.5
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_p,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        residual = x
        attn_output, _ = self.self_attn(x, x, x, need_weights=False)
        attn_output = self.dropout(attn_output)
        return self.layer_norm(residual + attn_output * self.scaling_factor)

class FeedForwardModule(nn.Module):
    """
    Feed-Forward 모듈.
    LayerNorm → 선형 계층 → Swish → Dropout → 선형 계층 → Dropout
    """
    def __init__(self, encoder_dim, expansion_factor, dropout_p):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            xavier_linear(encoder_dim, encoder_dim * expansion_factor),
            Swish(),
            nn.Dropout(p=dropout_p),
            xavier_linear(encoder_dim * expansion_factor, encoder_dim),
            nn.Dropout(p=dropout_p)
        )
    def forward(self, x):
        return self.sequential(x)

class ConformerBlock(nn.Module):
    """
    하나의 Conformer Block.
    구성: FeedForward → MultiHeaded Self-Attention → Convolution Module → FeedForward
    각 모듈은 Residual Connection과 함께 적용되고, 마지막에 LayerNorm 수행.
    Gradient checkpointing 적용.
    """
    def __init__(
        self,
        encoder_dim,
        num_attention_heads,
        feed_forward_expansion_factor,
        conv_expansion_factor,
        feed_forward_dropout_p,
        attention_dropout_p,
        conv_dropout_p,
        conv_kernel_size,
        half_step_residual
    ):
        super().__init__()
        self.ff_module1 = ResidualConnectionModule(
            FeedForwardModule(
                encoder_dim,
                feed_forward_expansion_factor,
                feed_forward_dropout_p
            ),
            module_factor=0.5 if half_step_residual else 1.0
        )
        self.mhsa_module = ResidualConnectionModule(
            MultiHeadedSelfAttentionModule(
                d_model=encoder_dim,
                num_heads=num_attention_heads,
                dropout_p=attention_dropout_p
            )
        )
        self.conv_module = ResidualConnectionModule(
            ConformerConvModule(
                in_channels=encoder_dim,
                kernel_size=conv_kernel_size,
                expansion_factor=conv_expansion_factor,
                dropout_p=conv_dropout_p
            )
        )
        self.ff_module2 = ResidualConnectionModule(
            FeedForwardModule(
                encoder_dim,
                feed_forward_expansion_factor,
                feed_forward_dropout_p
            ),
            module_factor=0.5 if half_step_residual else 1.0
        )
        self.layer_norm = nn.LayerNorm(encoder_dim)

    def forward(self, x):
        x = checkpoint(self.ff_module1, x)
        x = checkpoint(self.mhsa_module, x)
        x = checkpoint(self.conv_module, x)
        x = checkpoint(self.ff_module2, x)
        x = self.layer_norm(x)
        return x

class ConformerEncoderNoSubsampling(nn.Module):
    """
    Conformer Encoder (subsampling 없이).
    입력: (B, T, input_dim) → 출력: (B, T, encoder_dim)
    """
    def __init__(
        self,
        input_dim,
        encoder_dim,
        num_layers,
        num_attention_heads,
        feed_forward_expansion_factor,
        conv_expansion_factor,
        input_dropout_p,
        feed_forward_dropout_p,
        attention_dropout_p,
        conv_dropout_p,
        conv_kernel_size,
        half_step_residual
    ):
        super().__init__()
        self.conv_subsample = Identity()
        self.input_projection = nn.Sequential(
            xavier_linear(input_dim, encoder_dim),
            nn.Dropout(p=input_dropout_p)
        )
        self.layers = nn.ModuleList([
            ConformerBlock(
                encoder_dim=encoder_dim,
                num_attention_heads=num_attention_heads,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
                conv_expansion_factor=conv_expansion_factor,
                feed_forward_dropout_p=feed_forward_dropout_p,
                attention_dropout_p=attention_dropout_p,
                conv_dropout_p=conv_dropout_p,
                conv_kernel_size=conv_kernel_size,
                half_step_residual=half_step_residual
            ) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(encoder_dim)

    def forward(self, inputs, input_lengths):
        outputs, lengths = self.conv_subsample(inputs, input_lengths)
        outputs = self.input_projection(outputs)
        for layer in self.layers:
            outputs = layer(outputs)
        outputs = self.layer_norm(outputs)
        return outputs, lengths

# =============================================================================
# 3. IMU Feature-only Encoder: 그룹별 스케일 파라미터 추가 (in‐place 제거)
# =============================================================================

class IMUFeatureEncoder(nn.Module):
    """
    IMU 데이터용 Feature-only 인코더.
    14개 채널을 5개 그룹으로 묶어, 그룹별 학습 가능한 스케일을 곱한 뒤
    ConformerEncoderNoSubsampling을 통해 (B, T, encoder_dim) 형식의 feature를 반환.
    """
    def __init__(self, params):
        super().__init__()
        # ——— Config 전체가 넘어왔을 때도 처리해주기 ———
        if not isinstance(params, dict):
            # Config.Encoders.imu 에 정의된 파라미터 딕셔너리를 꺼냅니다.
            params = params.Encoders.imu
        # ── (1) 그룹별 스케일 파라미터 (5개) ─────────────────────────────
        #   초기값: 모두 1.0
        self.group_weight = nn.Parameter(torch.ones(5, dtype=torch.float32))

        # ── (2) 실제 Conformer 기반 인코더 ────────────────────────────────
        self.encoder = ConformerEncoderNoSubsampling(
            input_dim=params['input_dim'],                   # 14
            encoder_dim=params['encoder_dim'],               # ex: 256
            num_layers=params['num_layers'],                 # ex: 12
            num_attention_heads=params['num_heads'],         # ex: 4
            feed_forward_expansion_factor=params['ff_expansion'],  # ex: 4
            conv_expansion_factor=params['conv_expansion'],        # ex: 2
            input_dropout_p=params['input_dropout'],         # ex: 0.1
            feed_forward_dropout_p=params['ff_dropout'],     # ex: 0.1
            attention_dropout_p=params['attn_dropout'],      # ex: 0.1
            conv_dropout_p=params['conv_dropout'],           # ex: 0.1
            conv_kernel_size=params['conv_kernel'],          # ex: 7
            half_step_residual=params['half_step_residual']  # True/False
        )

    def forward(self, inputs, input_lengths):
        """
        Args:
            inputs: Tensor of shape (B, T, 14)
            input_lengths: 1D tensor of shape (B,) – 각 시퀀스의 실제 길이
        Returns:
            features: Tensor of shape (B, T, encoder_dim)
        """

        # (1) 소프트맥스로 게이팅된 그룹 중요도
        gates = F.softmax(self.group_weight, dim=0)  # shape (5,)

        part0 = inputs[:, :, 0:3]   * gates[0]
        part1 = inputs[:, :, 3:6]   * gates[1]
        part2 = inputs[:, :, 6:9]   * gates[2]
        part3 = inputs[:, :, 9:12]  * gates[3]
        part4 = inputs[:, :, 12:14] * gates[4]

        # 다시 순서대로 채널 축(dim=2)에서 concat → (B, T, 14)
        x = torch.cat([part0, part1, part2, part3, part4], dim=2)

        # (2) Conformer 인코더에 전달
        features, _ = self.encoder(x, input_lengths)
        return features  # (B, T, encoder_dim)


# =============================================================================
# 4. 간단한 테스트 코드 (행렬 곱이 잘 되는지 확인)
# =============================================================================

if __name__ == "__main__":
    # 배치 크기, 시퀀스 길이, 입력 feature 차원 설정
    B, T = 4, 50
    input_dim = 14
    encoder_dim = 256
    num_layers = 6  # 예시: 6층 사용

    # Config mock-up (필요한 부분만)
    class DummyCfg:
        pass
    cfg = DummyCfg()
    cfg.imu_params = {
        'input_dim': input_dim,
        'encoder_dim': encoder_dim,
        'num_layers': num_layers,
        'num_heads': 4,
        'ff_expansion': 4,
        'conv_expansion': 2,
        'input_dropout': 0.1,
        'ff_dropout': 0.1,
        'attn_dropout': 0.1,
        'conv_dropout': 0.1,
        'conv_kernel': 7,
        'half_step_residual': True
    }

    # 더미 입력 데이터 생성: (B, T, input_dim)
    dummy_input = torch.randn(B, T, input_dim)
    dummy_lengths = torch.full((B,), T, dtype=torch.long)

    model = IMUFeatureEncoder(cfg)
    out = model(dummy_input, dummy_lengths)
    print("Extracted features shape:", out.shape)  # 기대: (B, T, encoder_dim)

    # group_weight 파라미터가 잘 학습 가능한지 확인
    print("Group weights:", model.group_weight)
