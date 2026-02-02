"""
v27 + v24: The Ultimate Single Model with 2-Stage Training
v27(Bi-directional FiLM, Asymmetric Alignment) 아키텍처 기반,
v24(2-Stage Warm-up & Fine-tuning)의 안정적인 훈련 전략을 결합한 최종 버전.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from sklearn.metrics import accuracy_score
from config.config import Config
from copy import deepcopy

from trainers.base_trainer import dataProcessor
from data.loader import make_multitask_loader
from models.encoder.imu_encoder import IMUFeatureEncoder
from models.encoder.ppg_TCN_encoder import PPGEncoder
from models.encoder.veh_encoder import VehicleTCNEncoder
from models.encoder.sc_encoder import ScenarioEmbedding
from models.encoder.survey_encoder import PreSurveyEncoder

from models.fusion.predictors import EmotionPredictor
from models.head.motion_head import MotionHead

class FusionTrainer(nn.Module, dataProcessor):
    def __init__(self, cfg: Config, train_keys, val_keys, test_keys):
        super().__init__()
        dataProcessor.__init__(self, cfg)
        self.prepare()

        self.train_keys, self.val_keys, self.test_keys = train_keys, val_keys, test_keys
        self.cfg = cfg
        self.device = cfg.Project.device
        
        self._build_model()
        self._load_pretrained_weights()
        mode = self.cfg.FusionModel.interaction_mode
        # 감정→모션 그라드, 배치에 감정 라벨 있을 때만 허용
        self.gate_emo2mot_by_label = getattr(self.cfg.FusionModel, "gate_emo2mot_by_label", True)
        # 모션→감정 그라드 허용(기본 True). False면 예전처럼 detach 유지
        self.allow_mot_grad_from_emotion_loss = getattr(self.cfg.FusionModel, "allow_mot_grad_from_emotion_loss", True)
        # 퓨전 때 로더 모드
        self.loader_mode = getattr(self.cfg.FusionModel, "loader_mode", "emotion")  # "motion"으로 두면 모션 윈도우 사용

        print(f"Fine-tuning with {mode}")

    def _build_model(self):
        print("\n--- Building V8 model architecture with 2-Stage Training logic ---")
        hidden_dim = self.cfg.FusionModel.hidden_dim
        dropout_p = getattr(self.cfg.FusionModel, 'dropout', 0.2)
        num_heads = getattr(self.cfg.FusionModel, 'num_heads', 4)

        # 비대칭 정렬 손실을 위한 헤드
        self.alignment_head = nn.Linear(hidden_dim, hidden_dim)

        self.emo_delay_steps = getattr(self.cfg.FusionModel, "emo_delay_steps", 50)
        self.emo_ctx_window  = getattr(self.cfg.FusionModel, "emo_ctx_window", 200)
        self.emo_use_mot_logits = getattr(self.cfg.FusionModel, "emo_use_mot_logits", True)
        C_mot = self.cfg.PretrainMotion.num_motion if self.emo_use_mot_logits else 0

        # 모달리티 및 인코더/프로젝터 준비
        mot_mods = set(getattr(self.cfg.PretrainMotion, 'modalities_to_use', []))
        emo_mods = set(getattr(self.cfg.PretrainEmotion, 'modalities_to_use', []))
        all_unique_modalities = mot_mods.union(emo_mods)
        print(f"Required modalities: {list(all_unique_modalities)}")

        self.nets = nn.ModuleDict()
        self.projs = nn.ModuleDict()
        if 'imu' in all_unique_modalities:
            self.nets['imu'] = IMUFeatureEncoder(self.cfg.Encoders.imu)
            self.projs['imu'] = nn.Linear(self.cfg.Encoders.imu['encoder_dim'], hidden_dim)
        if 'veh' in all_unique_modalities:
            self.nets['veh'] = VehicleTCNEncoder(self.cfg.Encoders.veh)
            self.projs['veh'] = nn.Linear(self.cfg.Encoders.veh['embed_dim'], hidden_dim)
        if 'ppg' in all_unique_modalities:
            self.nets['ppg'] = PPGEncoder(self.cfg.Encoders.ppg)
            self.projs['ppg'] = nn.Linear(self.cfg.Encoders.ppg['embed_dim'] + 6, hidden_dim)
        if 'sc' in all_unique_modalities:
            self.nets['sc'] = ScenarioEmbedding(self.cfg.Encoders.sc)
            self.projs['sc'] = nn.Linear(self.cfg.Encoders.sc['embed_dim'], hidden_dim)
        if 'survey' in all_unique_modalities:
            self.nets['survey'] = PreSurveyEncoder(self.cfg.Encoders.survey)
            self.projs['survey'] = nn.Linear(self.cfg.Encoders.survey['embed_dim'], hidden_dim)

        # 태스크별 fusion/헤드
        if mot_mods:
            self.motion_modalities = sorted(list(mot_mods))
            self.motion_feature_fusion = nn.Conv1d(hidden_dim * len(self.motion_modalities), hidden_dim, 1)
            self.motion_head = MotionHead(hidden_dim, self.cfg.PretrainMotion.num_motion)

        if emo_mods:
            self.emotion_modalities = sorted([m for m in emo_mods if m != 'survey'])
            emo_fusion_in = hidden_dim * len(self.emotion_modalities)
            predictor_in  = emo_fusion_in + C_mot
            self.emotion_feature_fusion = nn.Sequential(
                nn.Linear(emo_fusion_in, hidden_dim), 
                nn.ReLU(), 
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim), 
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            )
            self.emotion_valence_predictor = EmotionPredictor(predictor_in, hidden_dim, self.cfg.PretrainEmotion.num_valence)
            self.emotion_arousal_predictor = EmotionPredictor(predictor_in, hidden_dim, self.cfg.PretrainEmotion.num_arousal)

        # Bi-directional cross-attention & FiLM
        self.cross_attn_m2e = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_ln_m     = nn.LayerNorm(hidden_dim)
        self.cross_drop_m   = nn.Dropout(dropout_p)
        self.cross_attn_e2m = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_ln_e     = nn.LayerNorm(hidden_dim)
        self.cross_drop_e   = nn.Dropout(dropout_p)
        self.emo_film = nn.Sequential(nn.Linear(hidden_dim, 2*hidden_dim), nn.ReLU(), nn.Linear(2*hidden_dim, 2*hidden_dim))
        self.mot_film = nn.Sequential(nn.Linear(hidden_dim, 2*hidden_dim), nn.ReLU(), nn.Linear(2*hidden_dim, 2*hidden_dim))

        # 로스/유틸
        init_log_vars = torch.tensor([0.0, 0.0, 0.0, 0.0])
        self.log_vars = nn.Parameter(init_log_vars)
        self.scaler = torch.cuda.amp.GradScaler()
        self.to(self.cfg.Project.device)

    def _load_pretrained_weights(self):
        device = self.cfg.Project.device
        def _get_enc(ckpt): return ckpt.get('encoder', ckpt)
        def _safe_load(module, state_dict, prefix, tag):
            if state_dict is None or not state_dict: return
            sub_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)} if prefix else state_dict
            if not sub_dict: return
            tgt_sd = module.state_dict()
            filtered = {k: v for k, v in sub_dict.items() if k in tgt_sd and tgt_sd[k].shape == v.shape}
            module.load_state_dict(filtered, strict=False)

        emo_cfg = self.cfg.PretrainEmotion
        if getattr(emo_cfg, 'modalities_to_use', []):
            emo_ckpt_path = getattr(emo_cfg, 'ckpt_path', 'weights/best_pretrain_emotion_ppg_sc_survey.pt')
            emo_ckpt = torch.load(emo_ckpt_path, map_location=device)
            emo_enc_states = _get_enc(emo_ckpt)
            emo_map = { 'ppg': ('ppg', 'nets.ppg.', 'projs.ppg.'), 'sc': ('sc', 'nets.sc.', 'projs.sc.'), 'veh': ('veh_e', 'nets.veh.', 'projs.veh.'), 'survey': ('survey', 'nets.survey.', 'projs.survey.') }
            for m in emo_cfg.modalities_to_use:
                if m in emo_map:
                    attr, enc_prefix, proj_prefix = emo_map[m]
                    if hasattr(self.nets, attr): _safe_load(getattr(self.nets, attr), emo_enc_states, enc_prefix, f"Emotion Encoder[{m}]")
                    if attr in self.projs: _safe_load(self.projs[attr], emo_enc_states, proj_prefix, f"Emotion Projection[{m}]")
            if hasattr(self, 'emotion_valence_predictor'): _safe_load(self.emotion_valence_predictor, emo_ckpt.get('valence_predictor'), None, "Emotion Valence Predictor")
            if hasattr(self, 'emotion_arousal_predictor'): _safe_load(self.emotion_arousal_predictor, emo_ckpt.get('arousal_predictor'), None, "Emotion Arousal Predictor")

        mot_cfg = self.cfg.PretrainMotion
        if getattr(mot_cfg, 'modalities_to_use', []):
            mot_ckpt_path = getattr(mot_cfg, 'ckpt_path', 'weights/best_pretrain_motion_imu_veh.pt')
            mot_ckpt = torch.load(mot_ckpt_path, map_location=device)
            mot_enc_states = _get_enc(mot_ckpt)
            mot_map = { 'imu': ('imu', 'imu.', 'p_imu.'), 'veh': ('veh', 'veh.', 'p_veh.'), 'sc': ('sc', 'sc.', 'p_sc.') }

            for m in mot_cfg.modalities_to_use:
                if m in mot_map:
                    attr, enc_prefix, proj_prefix = mot_map[m]
                    if hasattr(self.nets, attr): _safe_load(getattr(self.nets, attr), mot_enc_states, enc_prefix, f"Motion Encoder[{m}]")
                    if attr in self.projs: _safe_load(self.projs[attr], mot_enc_states, proj_prefix, f"Motion Projection[{m}]")
            if getattr(self.cfg, 'load_motion_head', True) and hasattr(self, 'motion_head'):
                _safe_load(self.motion_head, mot_ckpt.get('head'), None, 'Motion Head')
    
    # 인코더의 동결/해제 상태를 제어하는 헬퍼 함수
    def _set_encoder_grad(self, requires_grad: bool):
        status = "Unfrozen" if requires_grad else "Frozen"
        print(f"\n--- Setting all encoders to {status} ---")
        for param in itertools.chain(self.nets.parameters(), self.projs.parameters()):
            param.requires_grad = requires_grad

    # 1단계 워밍업을 위한 옵티마이저
    def _create_optimizer_stage1(self):
        print("\n--- Creating optimizer for Stage 1: Warm-up (Heads Only) ---")
        base_lr = self.cfg.MainTask.lr
        base_wd = self.cfg.MainTask.weight_decay
        
        fusion_head_params = itertools.chain(
            self.motion_feature_fusion.parameters(), self.emotion_feature_fusion.parameters(),
            self.motion_head.parameters(), self.emotion_valence_predictor.parameters(),
            self.emotion_arousal_predictor.parameters(), self.cross_attn_m2e.parameters(),
            self.cross_ln_m.parameters(), self.cross_attn_e2m.parameters(),
            self.cross_ln_e.parameters(), self.emo_film.parameters(),
            self.mot_film.parameters(), self.alignment_head.parameters()
        )
        
        optim_groups = [{'params': fusion_head_params, 'lr': base_lr, 'weight_decay': base_wd, 'name': 'Fusion/Heads'}]
        if self.cfg.MainTask.use_uncertainty_loss:
            optim_groups.append({'params': self.log_vars, 'lr': base_lr, 'weight_decay': 0.0, 'name': 'LogVars (No WD)'})
        
        self.optim = torch.optim.Adam(optim_groups)
        for group in self.optim.param_groups:
            group_params = sum(p.numel() for p in group['params'])
            print(f" - Group '{group['name']}': {group_params} params, lr={group['lr']:.1e}, wd={group['weight_decay']:.1e}")
            
    # 2단계 미세 조정을 위한 옵티마이저 (차등 학습률)
    def _create_optimizer_stage2(self):
        print("\n--- Creating optimizer for Stage 2: Fine-tuning (Differential LR) ---")
        base_lr = self.cfg.MainTask.lr
        base_wd = self.cfg.MainTask.weight_decay
        encoder_lr = base_lr / 10

        encoder_params = itertools.chain(self.nets.parameters(), self.projs.parameters())
        fusion_head_params = itertools.chain(
            self.motion_feature_fusion.parameters(), self.emotion_feature_fusion.parameters(),
            self.motion_head.parameters(), self.emotion_valence_predictor.parameters(),
            self.emotion_arousal_predictor.parameters(), self.cross_attn_m2e.parameters(),
            self.cross_ln_m.parameters(), self.cross_attn_e2m.parameters(),
            self.cross_ln_e.parameters(), self.emo_film.parameters(),
            self.mot_film.parameters(), self.alignment_head.parameters()
        )

        optim_groups = [
            {'params': encoder_params, 'lr': encoder_lr, 'weight_decay': base_wd, 'name': 'Encoders (Low LR)'},
            {'params': fusion_head_params, 'lr': base_lr, 'weight_decay': base_wd, 'name': 'Fusion/Heads (Base LR)'}
        ]
        if self.cfg.MainTask.use_uncertainty_loss:
            optim_groups.append({'params': self.log_vars, 'lr': base_lr, 'weight_decay': 0.0, 'name': 'LogVars (No WD)'})

        self.optim = torch.optim.Adam(optim_groups)
        for group in self.optim.param_groups:
            group_params = sum(p.numel() for p in group['params'])
            print(f" - Group '{group['name']}': {group_params} params, lr={group['lr']:.1e}, wd={group['weight_decay']:.1e}")
        
    def _process_hrv(self, batch, device):
        ppg_rr = batch['ppg_rr_emotion'].to(device)
        hrv_list = []
        if ppg_rr.dim() > 1 and ppg_rr.shape[1] > 0:
            hrv_list.append(torch.nan_to_num(ppg_rr.mean(dim=1, keepdim=True), nan=0.0))
            std_dev = torch.std(ppg_rr, dim=1, keepdim=True)
            hrv_list.append(torch.nan_to_num(std_dev, nan=0.0))
            hrv_list.append(torch.min(ppg_rr, dim=1, keepdim=True).values)
            hrv_list.append(torch.max(ppg_rr, dim=1, keepdim=True).values)
        else:
            hrv_list.extend([torch.zeros(ppg_rr.shape[0], 1, device=device)] * 4)

        for key in ['ppg_rmssd_emotion', 'ppg_sdnn_emotion']:
            value_tensor = batch[key].to(device).unsqueeze(1)
            hrv_list.append(torch.nan_to_num(value_tensor, nan=0.0))
        
        return torch.cat(hrv_list, dim=1)

    def _fit_channels(x, conv):
        """
        x: (B, C, T)   conv: nn.Conv1d (in_channels=?, kernel_size=1 등)
        conv가 기대하는 in_channels에 맞게 C 차원을 pad/slice.
        """
        need = conv.weight.shape[1]   # in_channels
        have = x.shape[1]
        if have == need:
            return x
        if have < need:
            pad = x.new_zeros(x.size(0), need - have, x.size(2))
            return torch.cat([x, pad], dim=1)
        else:
            return x[:, :need, :]
    
    
    def forward(self, batch):
        device = self.cfg.Project.device
        features = {'motion': {}, 'emotion': {}}
        
        # --- (A) 유틸: 모듈이 기대하는 입력 채널로 x를 보정 ---
        def _fit_to_module(x, module):
            """
            x: (B,C,T) 또는 (B,C)
            module: nn.Sequential 등 (Conv1d/BN1d/Linear 중 첫 레이어의 입력 크기에 맞춤)
            부족하면 0-pad, 많으면 slice.
            """
            need = None
            # 1) 시계열 입력 (B, C, T) → Conv1d/BN1d
            if x.dim() == 3:
                for m in module.modules():
                    if isinstance(m, torch.nn.BatchNorm1d):
                        need = m.num_features
                        break
                    if isinstance(m, torch.nn.Conv1d):
                        need = m.in_channels
                        break
                if need is not None:
                    have = x.size(1)
                    if have < need:
                        pad = x.new_zeros(x.size(0), need - have, x.size(2))
                        x = torch.cat([x, pad], dim=1)
                    elif have > need:
                        x = x[:, :need, :]
                return x

            # 2) 벡터 입력 (B, C) → Linear
            if x.dim() == 2:
                for m in module.modules():
                    if isinstance(m, torch.nn.Linear):
                        need = m.in_features
                        break
                if need is not None:
                    have = x.size(1)
                    if have < need:
                        pad = x.new_zeros(x.size(0), need - have)
                        x = torch.cat([x, pad], dim=1)
                    elif have > need:
                        x = x[:, :need]
                return x

            # 그외: 손대지 않음
            return x
        
        def _expected_in_channels_1d(module):
            # Conv1d 또는 BatchNorm1d의 입력 채널 수 추론
            for m in module.modules():
                if isinstance(m, torch.nn.Conv1d):
                    return m.in_channels
                if isinstance(m, torch.nn.BatchNorm1d):
                    return m.num_features
            return None

        def _expected_in_features_linear(module):
            for m in module.modules():
                if isinstance(m, torch.nn.Linear):
                    return m.in_features
            return None

        # -------------------------------------------------------

        T_mot = 1
        if 'label_motion' in batch and batch['label_motion'].dim() == 2:
            T_mot = batch['label_motion'].shape[1]
        elif 'imu_motion' in batch:
            T_mot = batch['imu_motion'].shape[1]

        # --- 1. Encoder & Projection ---
        if hasattr(self, 'motion_modalities'):
            for mod in self.motion_modalities:
                if mod == 'imu' and 'imu_motion' in batch: 
                    features['motion']['imu'] = self.projs['imu'](
                        self.nets['imu'](
                            batch['imu_motion'].to(device),
                            (batch['imu_motion'].abs().sum(-1) > 0).sum(1)
                        )
                    )
                elif mod == 'veh' and 'veh_motion' in batch: 
                    features['motion']['veh'] = self.projs['veh'](
                        self.nets['veh'](batch['veh_motion'].to(device).permute(0, 2, 1), return_pooled=True)
                    ).unsqueeze(1).expand(-1, T_mot, -1)
                elif mod == 'sc':
                    sc_pooled = self.nets['sc'](
                        batch['sc_motion_evt'].to(device),
                        batch['sc_motion_type'].to(device),
                        batch['sc_motion_phase'].to(device),
                        batch['sc_motion_time'].to(device)
                    )
                    features['motion']['sc'] = self.projs['sc'](sc_pooled.unsqueeze(1).expand(-1, T_mot, -1))
                elif mod == 'survey' and 'survey_e' in batch: 
                    survey_pooled = self.projs['survey'](self.nets['survey'](batch['survey_e'].to(device)))
                    features['motion']['survey'] = survey_pooled.unsqueeze(1).expand(-1, T_mot, -1)
                elif mod == 'ppg' and 'ppg_emotion' in batch: 
                    ppg_tcn = self.nets['ppg'](batch['ppg_emotion'].to(device).permute(0, 2, 1))
                    combined = torch.cat([ppg_tcn, self._process_hrv(batch, device)], dim=1)
                    ppg_pooled = self.projs['ppg'](combined)
                    features['motion']['ppg'] = ppg_pooled.unsqueeze(1).expand(-1, T_mot, -1)

        if hasattr(self, 'emotion_modalities'):
            for mod in self.emotion_modalities:
                if mod == 'ppg' and 'ppg_emotion' in batch: 
                    ppg_tcn = self.nets['ppg'](batch['ppg_emotion'].to(device).permute(0, 2, 1))
                    combined = torch.cat([ppg_tcn, self._process_hrv(batch, device)], dim=1)
                    features['emotion']['ppg'] = self.projs['ppg'](combined)
                elif mod == 'veh' and 'veh_emotion' in batch: 
                    features['emotion']['veh'] = self.projs['veh'](
                        self.nets['veh'](batch['veh_emotion'].to(device).permute(0, 2, 1), return_pooled=True)
                    )
                elif mod == 'sc':
                    features['emotion']['sc'] = self.projs['sc'](
                        self.nets['sc'](
                            batch['scenario_evt_e'].to(device), 
                            batch['scenario_type_e'].to(device), 
                            batch['phase_evt_e'].to(device), 
                            batch['scenario_time_e'].to(device)
                        )
                    )
                elif mod == 'imu' and 'imu_motion' in batch: 
                    imu_seq_out = self.nets['imu'](
                        batch['imu_motion'].to(device), 
                        (batch['imu_motion'].abs().sum(-1) > 0).sum(1)
                    )
                    imu_pooled = imu_seq_out.mean(dim=1)
                    features['emotion']['imu'] = self.projs['imu'](imu_pooled)
                elif mod == 'survey' and 'survey_e' in batch: 
                    features['emotion']['survey'] = self.projs['survey'](
                        self.nets['survey'](batch['survey_e'].to(device))
                    )

        # --- 2. 초기 Fusion ---
        H = int(self.cfg.FusionModel.hidden_dim)
        device = self.cfg.Project.device

        mot_feats = []
        for mod in getattr(self, "motion_modalities", []):
            if mod in features["motion"]:
                x = features["motion"][mod]          # (B, T, H)
            else:
                # 결측이면 0 텐서 (B, T_mot, H)
                # B 추론
                if features["emotion"]:
                    B = next(iter(features["emotion"].values())).shape[0]
                elif features["motion"]:
                    B = next(iter(features["motion"].values())).shape[0]
                else:
                    # 배치 임의 텐서 하나에서 B 추론 (fallback)
                    B = next(v for v in batch.values() if torch.is_tensor(v)).shape[0]
                x = torch.zeros(B, T_mot, H, device=device)
            mot_feats.append(x)

        if len(mot_feats) == 0:
            # ★ 리스트가 비면, fusion 블록이 기대하는 채널 크기로 0 텐서 만들어 바로 통과
            if features["emotion"]:
                B = next(iter(features["emotion"].values())).shape[0]
            else:
                B = next(v for v in batch.values() if torch.is_tensor(v)).shape[0]
            expC = _expected_in_channels_1d(self.motion_feature_fusion) or H
            motion_cat_full = torch.zeros(B, expC, max(1, T_mot), device=device)
            fused_motion_seq = self.motion_feature_fusion(motion_cat_full)       # (B, H, T)
        else:
            motion_cat_full = torch.cat(mot_feats, dim=2).permute(0, 2, 1)       # (B, H*M, T)
            motion_cat_full = _fit_to_module(motion_cat_full, self.motion_feature_fusion)
            fused_motion_seq = self.motion_feature_fusion(motion_cat_full)       # (B, H, T)

        # emotion: 항상 len(self.emotion_modalities)개를 쌓는다 (없으면 0)
        emo_feats = []
        for mod in getattr(self, "emotion_modalities", []):
            if mod in features["emotion"]:
                v = features["emotion"][mod]             # (B,H)
            else:
                B = next(iter(features["motion"].values())).shape[0] if features["motion"] else \
                    next(iter(features["emotion"].values())).shape[0]
                v = torch.zeros(B, H, device=device)
            emo_feats.append(v)

        # (B, H*E)
        emo_feats = []

        if len(emo_feats) == 0:
            # ★ 리스트가 비면, fusion 블록이 기대하는 in_features 크기로 0 벡터 만들어 바로 통과
            B = next(v for v in batch.values() if torch.is_tensor(v)).shape[0]
            need = _expected_in_features_linear(self.emotion_feature_fusion) or H
            emotion_cat = torch.zeros(B, need, device=device)
            fused_emotion_vector = self.emotion_feature_fusion(emotion_cat)      # (B, H)
        else:
            emotion_cat = torch.cat(emo_feats, dim=1)                            # (B, H*E)
            emotion_cat = _fit_to_module(emotion_cat, self.emotion_feature_fusion)
            fused_emotion_vector = self.emotion_feature_fusion(emotion_cat)      # (B, H)

        print(f"[DBG] mot_feats={len(mot_feats)}, emo_feats={len(emo_feats)}, "
        f"motion_expC={_expected_in_channels_1d(self.motion_feature_fusion)}, "
        f"emotion_need={_expected_in_features_linear(self.emotion_feature_fusion)}")

        # --- 3. Bi-directional Cross-Interaction & FiLM ---
        motion_tokens = fused_motion_seq.permute(0, 2, 1)
        emotion_token = fused_emotion_vector.unsqueeze(1)
        B, T, H = motion_tokens.shape
        delta  = min(self.emo_delay_steps, T-1)
        t_star = max(0, T - 1 - delta)
        left   = max(0, t_star - self.emo_ctx_window + 1)
        right  = t_star + 1

        causal_motion_ctx = motion_tokens[:, left:right, :]
        if not self.allow_mot_grad_from_emotion_loss:
            causal_motion_ctx = causal_motion_ctx.detach()

        if 'valence_reg_emotion' in batch:
            _v = batch['valence_reg_emotion'][:, -1].to(device); has_v = (_v >= 1) & (_v < 10)
        else:
            has_v = torch.zeros(B, dtype=torch.bool, device=device)
        if 'arousal_reg_emotion' in batch:
            _a = batch['arousal_reg_emotion'][:, -1].to(device); has_a = (_a >= 1) & (_a < 10)
        else:
            has_a = torch.zeros(B, dtype=torch.bool, device=device)
        has_emo_any = (has_v | has_a).any()

        def emo_kv_for_m2e():
            if self.gate_emo2mot_by_label:
                return emotion_token if has_emo_any else emotion_token.detach()
            else:
                return emotion_token

        # --- 교차 상호작용을 배치 조건에 따라 선택적으로 수행 ---
        has_emo_inputs = any(m in features["emotion"] for m in getattr(self, "emotion_modalities", []))
        fusion_enabled = bool(has_emo_inputs or has_emo_any.item())
        mode = self.cfg.FusionModel.interaction_mode
        if fusion_enabled:
            if mode == 'cross_attention_film':
                attn_e, _ = self.cross_attn_e2m(emotion_token, causal_motion_ctx, causal_motion_ctx)
                refined_emotion_token = self.cross_ln_e(emotion_token + self.cross_drop_e(attn_e))
                kv = emo_kv_for_m2e()
                attn_m, _ = self.cross_attn_m2e(motion_tokens, kv, kv)
                refined_motion_tokens = self.cross_ln_m(motion_tokens + self.cross_drop_m(attn_m))
                motion_context_for_emo = refined_emotion_token.squeeze(1)
                gamma_beta_emo = self.emo_film(motion_context_for_emo)
                gamma_emo, beta_emo = gamma_beta_emo.chunk(2, dim=-1)
                final_emotion_vector = fused_emotion_vector * (1 + torch.tanh(gamma_emo)) + beta_emo
                final_motion_tokens = refined_motion_tokens
            elif mode == 'cross_attention_only':
                attn_e, _ = self.cross_attn_e2m(emotion_token, causal_motion_ctx, causal_motion_ctx)
                refined_emotion_token = self.cross_ln_e(emotion_token + self.cross_drop_e(attn_e))
                kv = emo_kv_for_m2e()
                attn_m, _ = self.cross_attn_m2e(motion_tokens, kv, kv)
                refined_motion_tokens = self.cross_ln_m(motion_tokens + self.cross_drop_m(attn_m))
                final_emotion_vector = refined_emotion_token.squeeze(1)
                motion_context_for_emo = final_emotion_vector
                final_motion_tokens = refined_motion_tokens
            elif mode == 'film_only':
                motion_context_for_emo = causal_motion_ctx.mean(dim=1)
                gamma_beta_emo = self.emo_film(motion_context_for_emo)
                gamma_emo, beta_emo = gamma_beta_emo.chunk(2, dim=-1)
                final_emotion_vector = fused_emotion_vector * (1 + torch.tanh(gamma_emo)) + beta_emo
                final_motion_tokens = motion_tokens
            else:
                final_emotion_vector = fused_emotion_vector
                final_motion_tokens = motion_tokens
                motion_context_for_emo = final_emotion_vector  # 아래 predictor context 필요 시 대비
        else:
            # 감정 입력/라벨이 전혀 없는 배치 → 상호작용 스킵 (base만 사용)
            final_emotion_vector = fused_emotion_vector.detach()
            final_motion_tokens = motion_tokens
            motion_context_for_emo = final_emotion_vector

        # --- 4. Prediction Heads ---
        # 4-1) 먼저 모션 로짓부터 만든다
        motion_input_for_head = final_motion_tokens.permute(0, 2, 1)  # (B, H, T)
        mot_logits = self.motion_head(motion_input_for_head)          # (B, T, C_mot)

        # 4-2) emotion_cat_plus 구성 (모션 로짓을 붙일 옵션이면 지금 붙인다)
        emotion_cat_plus = emotion_cat
        if getattr(self, "emo_use_mot_logits", False):
            # (B, C_mot) — 시간 평균, grad 차단
            p_mot = F.softmax(mot_logits, dim=-1).mean(dim=1).detach()
            emotion_cat_plus = torch.cat([emotion_cat, p_mot], dim=1)

        # 4-3) emotion predictors 입력 준비
        emotion_input_dict = {'fused': emotion_cat_plus.unsqueeze(1)}
        if ('survey' in self.cfg.PretrainEmotion.modalities_to_use) and ('survey' in features['emotion']):
            # 필요하면 static 키 채워주기 (네 코드와 동일)
            emotion_input_dict['static'] = features['emotion']['survey']

        # 4-4) 감정 로짓 계산
        valence_logits = self.emotion_valence_predictor(emotion_input_dict, context=motion_context_for_emo)
        arousal_logits = self.emotion_arousal_predictor(emotion_input_dict, context=motion_context_for_emo)

        fusion_enabled = (len(emo_feats) > 0) 
        
        return {
            'motion_logits': mot_logits,
            'valence_logits': valence_logits,
            'arousal_logits': arousal_logits,
            'fused_motion': final_motion_tokens.mean(dim=1),
            'fused_emotion': final_emotion_vector,
            'fusion_enabled': fusion_enabled,
        }
    
    def run_epoch(self, loader, train: bool):
        self.train(train)
        total_loss, num_batches = 0.0, 0
        device = self.cfg.Project.device

        for batch in loader:
            with torch.set_grad_enabled(train):
                out = self.forward(batch) 

                mot_logits, mot_labels = out['motion_logits'], batch['label_motion'].to(device)
                valid_mask = (mot_labels.reshape(-1) > 0) & (mot_labels.reshape(-1) != 4)
                loss_mot = F.cross_entropy(mot_logits.reshape(-1, mot_logits.shape[-1])[valid_mask], (mot_labels.reshape(-1)[valid_mask] - 1).long()) if valid_mask.any() else torch.tensor(0., device=device)

                val_logits, raw_v_seq = out['valence_logits'], batch['valence_reg_emotion'].to(device)
                raw_v = raw_v_seq[:, -1] # (B, T) -> (B,)
                mask_v = (raw_v >= 1) & (raw_v < 10)
                tgt_v = torch.full_like(raw_v, -100, dtype=torch.long)
                tgt_v[mask_v] = raw_v[mask_v].long() - 1
                loss_v = F.cross_entropy(val_logits, tgt_v, ignore_index=-100)

                aro_logits, raw_a_seq = out['arousal_logits'], batch['arousal_reg_emotion'].to(device)
                raw_a = raw_a_seq[:, -1] # (B, T) -> (B,)
                mask_a = (raw_a >= 1) & (raw_a < 10)
                tgt_a = torch.full_like(raw_a, -100, dtype=torch.long)
                tgt_a[mask_a] = raw_a[mask_a].long() - 1
                loss_a = F.cross_entropy(aro_logits, tgt_a, ignore_index=-100)
                                
                fused_motion_detached = out['fused_motion'].detach()
                predicted_emotion = self.alignment_head(fused_motion_detached)
                loss_align = F.mse_loss(predicted_emotion, out['fused_emotion'])

                emo_loss_weight = self.cfg.MainTask.lambda_emotion
                
                if self.cfg.MainTask.use_uncertainty_loss:
                    # [수정] Uncertainty Loss에도 가중치를 적용할 수 있으나, 더 간단한 방식은 
                    # 각 손실 항에 직접 곱해주는 것입니다. 여기서는 두 손실에 동일한 가중치를 적용합니다.
                    loss_v_weighted = emo_loss_weight * loss_v
                    loss_a_weighted = emo_loss_weight * loss_a
                    
                    loss = (torch.exp(-self.log_vars[0]) * loss_mot + 0.5 * self.log_vars[0]) + \
                        (torch.exp(-self.log_vars[1]) * loss_v_weighted + 0.5 * self.log_vars[1]) + \
                        (torch.exp(-self.log_vars[2]) * loss_a_weighted + 0.5 * self.log_vars[2]) + \
                        (torch.exp(-self.log_vars[3]) * loss_align * self.cfg.MainTask.cross_modal_lambda + 0.5 * self.log_vars[3])
                else:
                    # [수정] Valence와 Arousal 손실에 lambda_emotion 가중치를 곱해줍니다.
                    loss = loss_mot + emo_loss_weight * (loss_v + loss_a) + (loss_align * self.cfg.MainTask.cross_modal_lambda)

            if train:
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.scaler.step(self.optim)
                self.scaler.update()

            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    @torch.no_grad()
    def evaluate(self, loader, return_preds=False):
        self.eval()
        all_preds = {'motion_preds': [], 'motion_trues': [], 'valence_preds': [], 'valence_trues': [], 'arousal_preds': [], 'arousal_trues': []}
        for batch in loader:
            out = self.forward(batch)
            p_mot = out['motion_logits'].argmax(-1).cpu(); t_mot_raw = batch['label_motion'].cpu()
            mask_mot = (t_mot_raw > 0) & (t_mot_raw != 4)
            all_preds['motion_preds'].append(p_mot[mask_mot]); all_preds['motion_trues'].append(t_mot_raw[mask_mot] - 1)
            
            # Valence
            p_v_multiclass = out['valence_logits'].argmax(-1).cpu()
            raw_v_seq = batch['valence_reg_emotion'].cpu()
            raw_v = raw_v_seq[:, -1] # (B, T) -> (B,)
            mask_v = (raw_v >= 1) & (raw_v < 10)
            t_v_multiclass = torch.full_like(raw_v, -1, dtype=torch.long)
            t_v_multiclass[mask_v] = raw_v[mask_v].long() - 1

            # 3진으로 변환하여 평가
            p_v_ternary = torch.full_like(p_v_multiclass, -1)
            p_v_ternary[(p_v_multiclass >= 0) & (p_v_multiclass <= 2)] = 0  # Low: 1,2,3점
            p_v_ternary[(p_v_multiclass >= 3) & (p_v_multiclass <= 5)] = 1  # Medium: 4,5,6점
            p_v_ternary[(p_v_multiclass >= 6) & (p_v_multiclass <= 8)] = 2  # High: 7,8,9점
            
            t_v_ternary = torch.full_like(t_v_multiclass, -1)
            t_v_ternary[(t_v_multiclass >= 0) & (t_v_multiclass <= 2)] = 0
            t_v_ternary[(t_v_multiclass >= 3) & (t_v_multiclass <= 5)] = 1
            t_v_ternary[(t_v_multiclass >= 6) & (t_v_multiclass <= 8)] = 2

            valid_mask_v = t_v_ternary != -1
            all_preds['valence_preds'].append(p_v_ternary[valid_mask_v])
            all_preds['valence_trues'].append(t_v_ternary[valid_mask_v])

            # Arousal
            p_a_multiclass = out['arousal_logits'].argmax(-1).cpu()
            raw_a_seq = batch['arousal_reg_emotion'].cpu()
            raw_a = raw_a_seq[:, -1] # (B, T) -> (B,)

            mask_a = (raw_a >= 1) & (raw_a < 10)
            t_a_multiclass = torch.full_like(raw_a, -1, dtype=torch.long)
            t_a_multiclass[mask_a] = raw_a[mask_a].long() - 1

            # 3진으로 변환하여 평가
            p_a_ternary = torch.full_like(p_a_multiclass, -1)
            p_a_ternary[(p_a_multiclass >= 0) & (p_a_multiclass <= 2)] = 0
            p_a_ternary[(p_a_multiclass >= 3) & (p_a_multiclass <= 5)] = 1
            p_a_ternary[(p_a_multiclass >= 6) & (p_a_multiclass <= 8)] = 2
            
            t_a_ternary = torch.full_like(t_a_multiclass, -1)
            t_a_ternary[(t_a_multiclass >= 0) & (t_a_multiclass <= 2)] = 0
            t_a_ternary[(t_a_multiclass >= 3) & (t_a_multiclass <= 5)] = 1
            t_a_ternary[(t_a_multiclass >= 6) & (t_a_multiclass <= 8)] = 2

            valid_mask_a = t_a_ternary != -1
            all_preds['arousal_preds'].append(p_a_ternary[valid_mask_a])
            all_preds['arousal_trues'].append(t_a_ternary[valid_mask_a])
            
        for key in all_preds: all_preds[key] = torch.cat(all_preds[key])
        if return_preds: return all_preds
        
        acc_mot = accuracy_score(all_preds['motion_trues'], all_preds['motion_preds'])
        acc_v = accuracy_score(all_preds['valence_trues'], all_preds['valence_preds'])
        acc_a = accuracy_score(all_preds['arousal_trues'], all_preds['arousal_preds'])
        return acc_mot, acc_v, acc_a
            
    # 2단계 학습을 총괄하는 메인 훈련 함수
    def fusion_train(self, save_path="weights/best_fusion_v28.pt"):
        tr_loader = make_multitask_loader(self.cfg, self.train_keys, shuffle=True, dp=self,
                                          include_indices=True, mode=self.loader_mode)
        va_loader = make_multitask_loader(self.cfg, self.val_keys, shuffle=False, dp=self,
                                          include_indices=True, mode=self.loader_mode)
        
        warmup_epochs = getattr(self.cfg.MainTask, 'warmup_epochs', 10)
        total_epochs = self.cfg.MainTask.epochs
        
        # 어떤 지표를 기준으로 최고 모델을 저장할지 설정 ('accuracy' 또는 'loss')
        metric_to_monitor = 'loss'
        
        # 선택된 지표에 따라 best_score 초기값을 설정합니다.
        if metric_to_monitor == 'loss':
            best_score = float('inf')
            print(f"\n✅ Monitoring Validation Loss to find the best model. Lower is better.")
        else:
            best_score = 0.0
            print(f"\n✅ Monitoring Validation Accuracy to find the best model. Higher is better.")
            
        patience_counter = 0
        best_performance = {}

        # --- 1단계: 워밍업 (Warm-up) ---
        if warmup_epochs > 0:
            print("\n" + "="*50)
            print(f"🚀 STARTING STAGE 1: WARM-UP FOR {warmup_epochs} EPOCHS")
            print("="*50)
            self._set_encoder_grad(requires_grad=False)
            self._create_optimizer_stage1()

            for epoch in range(1, warmup_epochs + 1):
                self.current_epoch = epoch
                tr_loss = self.run_epoch(tr_loader, train=True)
                va_loss = self.run_epoch(va_loader, train=False)
                va_acc_mot, va_acc_v, va_acc_a = self.evaluate(va_loader)
                print(f"[WARM-UP] Epoch {epoch:02d}/{warmup_epochs} | Tr Loss {tr_loss:.4f} | Val Loss {va_loss:.4f} | Acc(M/V/A): {va_acc_mot:.3f}/{va_acc_v:.3f}/{va_acc_a:.3f}")

                # 설정된 지표에 따라 현재 점수를 결정합니다.
                current_score = va_loss if metric_to_monitor == 'loss' else (va_acc_v + va_acc_a) / 2

                # 지표에 따라 최고 성능인지 판단합니다.
                is_best = (metric_to_monitor == 'loss' and current_score < best_score) or \
                        (metric_to_monitor == 'accuracy' and current_score > best_score)

                if is_best:
                    best_score = current_score
                    patience_counter = 0
                    best_performance = {'mot_acc': va_acc_mot, 'val_acc': va_acc_v, 'aro_acc': va_acc_a, 'best_val_loss': va_loss}
                    torch.save({'model_state_dict': self.state_dict()}, save_path)
                    print(f"  -> Best warm-up model saved to {save_path} (Best {metric_to_monitor.capitalize()}: {best_score:.4f})")
                else:
                    patience_counter += 1

        # --- 2단계: 미세 조정 (Fine-tuning) ---
        print("\n" + "="*50)
        print(f"🚀 STARTING STAGE 2: FINE-TUNING FOR {total_epochs - warmup_epochs} EPOCHS")
        print("="*50)
        self._set_encoder_grad(requires_grad=True)

        if warmup_epochs > 0 and os.path.exists(save_path):
            print(f"Loading best model from warm-up stage: {save_path}")
            checkpoint = torch.load(save_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            
        print("\n--- Adding Encoder group to the optimizer for Fine-tuning ---")
        encoder_lr = self.cfg.MainTask.lr / 10
        encoder_params = itertools.chain(self.nets.parameters(), self.projs.parameters())
        self.optim.add_param_group({
            'params': encoder_params, 
            'lr': encoder_lr, 
            'weight_decay': self.cfg.MainTask.weight_decay, 
            'name': 'Encoders (Low LR)'
        })

        print("Optimizer groups after update:")
        for group in self.optim.param_groups:
            group_params = sum(p.numel() for p in group['params'])
            print(f" - Group '{group['name']}': {group_params} params, lr={group['lr']:.1e}, wd={group['weight_decay']:.1e}")
        
        patience_counter = 0

        for epoch in range(warmup_epochs + 1, total_epochs + 1):
            self.current_epoch = epoch
            tr_loss = self.run_epoch(tr_loader, train=True)
            va_loss = self.run_epoch(va_loader, train=False)
            va_acc_mot, va_acc_v, va_acc_a = self.evaluate(va_loader)
            print(f"[FINE-TUNE] Epoch {epoch:02d}/{total_epochs} | Tr Loss {tr_loss:.4f} | Val Loss {va_loss:.4f} | Acc(M/V/A): {va_acc_mot:.3f}/{va_acc_v:.3f}/{va_acc_a:.3f}")

            # 설정된 지표에 따라 현재 점수를 결정합니다.
            current_score = va_loss if metric_to_monitor == 'loss' else (va_acc_v + va_acc_a) / 2

            # 지표에 따라 최고 성능인지 판단합니다.
            is_best = (metric_to_monitor == 'loss' and current_score < best_score) or \
                    (metric_to_monitor == 'accuracy' and current_score > best_score)
            
            if is_best:
                best_score = current_score
                patience_counter = 0
                best_performance = {'mot_acc': va_acc_mot, 'val_acc': va_acc_v, 'aro_acc': va_acc_a, 'best_val_loss': va_loss}
                torch.save({
                    'model_state_dict': self.state_dict(), 
                    'optimizer_state_dict': self.optim.state_dict(),
                }, save_path)
                print(f"  -> Best fine-tuned model saved to {save_path} (Best {metric_to_monitor.capitalize()}: {best_score:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.MainTask.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        print(f"\n--- Best Validation Performance (Staged Training) ---")
        # 최종 결과 출력문을 개선했습니다.
        print(f"Monitored Metric: {metric_to_monitor.capitalize()} | Best Score: {best_score:.4f}")
        print(f"Acc(M/V/A): {best_performance.get('mot_acc', 0):.3f}/{best_performance.get('val_acc', 0):.3f}/{best_performance.get('aro_acc', 0):.3f} | Loss: {best_performance.get('best_val_loss', 0):.4f}")

        return best_performance