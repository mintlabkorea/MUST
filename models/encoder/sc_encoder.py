import torch
import torch.nn as nn
import torch.nn.functional as F

# class ScenarioEmbedding(nn.Module): #pooling
#     def __init__(self, params):
#         super().__init__()
#         # ——— Config 전체가 넘어왔을 때도 처리해주기 ———
#         if not isinstance(params, dict):
#             # Config.Encoders.imu 에 정의된 파라미터 딕셔너리를 꺼냅니다.
#             params = params.Encoders.sc
#         p=params
#         E = params['embed_dim']           # 16
#         vocab_size = params['vocab_size'] 

#         self.scenario_embed = nn.Embedding(vocab_size,       E, padding_idx=0)  # (B,16)
#         self.type_embed     = nn.Embedding(p['max_scenario_type'] + 1, E)       # (B,16)
#         self.phase_embed    = nn.Embedding(p['max_phase_id'] + 1,      E)       # (B,16)

#         # Timestamp mapping (1→16)
#         self.ts_linear = nn.Sequential(
#             nn.Linear(1, E),
#             nn.LayerNorm(E)
#         )

#         # Fusion MLP: (16*4 = 64) → (16*2 = 32) → 16
#         self.mlp = nn.Sequential(
#             nn.Linear(E * 4, E * 2),   # (64→32)
#             nn.ReLU(),
#             nn.Dropout(p['dropout']),
#             nn.Linear(E * 2, E)       # (32→16)
#         )
#         self.p = p

#     def forward(self, scenario_ids, scenario_types, phase_ids, timestamps):
#         device = self.scenario_embed.weight.device
#         scenario_ids = scenario_ids.to(device)
#         scenario_types = scenario_types.to(device)
#         phase_ids = phase_ids.to(device)
#         ts = timestamps.to(device)

#         if ts.dim() == 1:
#             ts = ts.unsqueeze(-1)

#         # ─── 모든 범주형 입력에 대해 안전하게 클램프로 OOB 처리 ─────────
#         scenario_ids = scenario_ids % self.p['vocab_size']
#         scenario_types = scenario_types % (self.p.get('max_scenario_type', 5) + 1)
#         phase_ids = phase_ids % (self.p.get('max_phase_id', 3) + 1)

#         s   = self.scenario_embed(scenario_ids)    # (B,16)
#         t_e = self.type_embed(scenario_types)      # (B,16)
#         p_e = self.phase_embed(phase_ids)          # (B,16)
#         ts_e = self.ts_linear(ts)                  # (B,16)


#         fused = torch.cat([s, t_e, p_e, ts_e], dim=-1)  # (B, 16*4 = 64)
#         return self.mlp(fused)                           # (B,16)

class ScenarioEmbedding(nn.Module): # 시계열

    def __init__(self, params):
        super().__init__()
        # ——— Config 전체가 넘어왔을 때도 처리해주기 ———
        if not isinstance(params, dict):
            # Config.Encoders.imu 에 정의된 파라미터 딕셔너리를 꺼냅니다.
            params = params.Encoders.sc
        p=params
        E = params['embed_dim']           # 16
        vocab_size = params['vocab_size'] 

        self.scenario_embed = nn.Embedding(vocab_size,       E, padding_idx=0)  # (B,16)
        self.type_embed     = nn.Embedding(p['max_scenario_type'] + 1, E)       # (B,16)
        self.phase_embed    = nn.Embedding(p['max_phase_id'] + 1,      E)       # (B,16)

        # Timestamp mapping (1→16)
        self.ts_linear = nn.Sequential(
            nn.Linear(1, E),
            nn.LayerNorm(E)
        )

        # Fusion MLP: (16*4 = 64) → (16*2 = 32) → 16
        self.mlp = nn.Sequential(
            nn.Linear(E * 4, E * 2),   # (64→32)
            nn.ReLU(),
            nn.Dropout(p['dropout']),
            nn.Linear(E * 2, E)       # (32→16)
        )
        self.p = p

    def forward(self, scenario_ids, scenario_types, phase_ids, timestamps, T=None):
        device = self.scenario_embed.weight.device

        scenario_ids = scenario_ids.to(device)
        scenario_types = scenario_types.to(device)
        phase_ids = phase_ids.to(device)
        ts = timestamps.to(device)

        if ts.dim() == 1:
            ts = ts.unsqueeze(-1)

        # ─── 모든 범주형 입력에 대해 안전하게 클램프로 OOB 처리 ─────────
        scenario_ids = scenario_ids % self.p['vocab_size']
        scenario_types = scenario_types % (self.p.get('max_scenario_type', 5) + 1)
        phase_ids = phase_ids % (self.p.get('max_phase_id', 3) + 1)

        # 1) original embedding (각각 (B,1,D))
        s   = self.scenario_embed(scenario_ids)      # (B,1,16)
        t_e = self.type_embed   (scenario_types)     # (B,1,16)
        p_e = self.phase_embed  (phase_ids)          # (B,1,16)
        ts  = timestamps.to(device)
        if ts.dim()==1: ts = ts.unsqueeze(-1)        # (B,1)
        ts_e = self.ts_linear(ts)                    # (B,16)

        # 2) squeeze out the length-1 차원 -> (B,D)
        s   = s.squeeze(1)      # (B,16)
        t_e = t_e.squeeze(1)    # (B,16)
        p_e = p_e.squeeze(1)    # (B,16)
        # ts_e 이미 (B,16)

        # 3) T가 주어지면 시퀀스 길이로 expand
        if T is not None:
            s   = s .unsqueeze(1).expand(-1, T, -1)  # (B, T, 16)
            t_e = t_e.unsqueeze(1).expand(-1, T, -1)  
            p_e = p_e.unsqueeze(1).expand(-1, T, -1)
            ts_e= ts_e.unsqueeze(1).expand(-1, T, -1)

            fused = torch.cat([s, t_e, p_e, ts_e], dim=-1)  # (B, T, 64)
            out = self.mlp(fused)                           # (B, T, E)
            return out

        # T가 없으면 그냥 (B, 64) -> (B, E)
        fused = torch.cat([s, t_e, p_e, ts_e], dim=-1)      # (B,64)
        out   = self.mlp(fused)                             # (B, E)
        return out