import torch
from imu_encoder import ConformerNoSubsampling
from ppg_lstm_encoder import PPGEncoder
from sc_encoder import ScenarioEmbedding
from survey_encoder import PreSurveyEncoder
from veh_encoder import VehicleTCNEncoder

B, T = 4, 60  # batch_size, 시계열 길이

# 1. IMU 인코더
imu_enc = ConformerNoSubsampling(num_classes=None, input_dim=15, encoder_dim=256)
imu_out, _ = imu_enc(torch.randn(B, T, 15), torch.tensor([T]*B))
print("IMU shape:", imu_out.shape)  # (B, T, 256)

# 2. PPG 인코더
ppg_enc = PPGEncoder()
ppg_out = ppg_enc(
    torch.randn(B, 2000),
    torch.randn(B, 30),
    torch.randn(B),
    torch.randn(B)
)
print("PPG shape:", ppg_out.shape)  # (B, 64)

# 3. Scenario 인코더
sc_enc = ScenarioEmbedding()
sc_out = sc_enc(
    torch.randint(0, 21, (B,)),
    torch.randint(0, 6, (B,)),
    torch.randint(0, 4, (B,)),
    torch.randn(B)
)
print("Scenario shape:", sc_out.shape)  # (B, 16)

# 4. Survey 인코더
survey_enc = PreSurveyEncoder(input_dim=10)
survey_out = survey_enc(torch.randn(B, 10))
print("Survey shape:", survey_out.shape)  # (B, 8)

# 5. Vehicle 인코더
veh_enc = VehicleTCNEncoder(input_dim=15, embed_dim=32)
veh_out = veh_enc(torch.randn(B, 15, T))  # (B, embed_dim, T)
veh_out = veh_out.permute(0, 2, 1)        # (B, T, embed_dim)
print("Vehicle shape:", veh_out.shape)
