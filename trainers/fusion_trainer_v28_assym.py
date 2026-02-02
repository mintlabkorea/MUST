"""
v28 assymetric: 감정/행동이 각 데이터로더를 사용, 감정 레이블이 존재하는 경우에만 cross fusion
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from sklearn.metrics import accuracy_score
from config.config import Config
from copy import deepcopy
from tqdm import tqdm

from trainers.base_trainer import dataProcessor
from data.loader import make_motion_loader, make_emotion_loader, make_multitask_loader
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
        print(f"Fine-tuning with {mode}")

    def _build_model(self):
        print("\n--- Building V28-assym model architecture---")
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
            emo_ckpt_path = getattr(emo_cfg, 'ckpt_path', 'weights/best_emotion_ppg_sc_survey.pt')
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
            mot_ckpt_path = getattr(mot_cfg, 'ckpt_path', 'weights/best_motion_imu_veh.pt')
            mot_ckpt = torch.load(mot_ckpt_path, map_location=device)
            mot_enc_states = _get_enc(mot_ckpt)
            mot_map = { 'imu': ('imu', 'imu.', 'p_imu.'), 'veh_m': ('veh_m', 'veh.', 'p_veh_m.'), 'sc': ('sc', 'sc.', 'p_sc.') }
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

    def forward(self, batch, task_type: str):
        device = self.cfg.Project.device
        features = {'motion': {}, 'emotion': {}}
        
        # motion과 emotion 배치에 따라 시퀀스 길이(T_mot) 결정
        if task_type == 'motion':
            T_mot = batch['label_motion'].shape[1]
        else: # emotion
            T_mot = batch['imu_emotion'].shape[1]

        # --- 1. Encoder & Projection ---
        imu_key = 'imu_motion' if task_type == 'motion' else 'imu_emotion'
        veh_key = 'veh_motion' if task_type == 'motion' else 'veh_emotion'
        
        # Motion 스트림 특징 추출
        if hasattr(self, 'motion_modalities'):
            for mod in self.motion_modalities:
                if mod == 'imu' and imu_key in batch and batch[imu_key].numel() > 0:
                    features['motion']['imu'] = self.projs['imu'](self.nets['imu'](batch[imu_key].to(device), (batch[imu_key].abs().sum(-1) > 0).sum(1)))
                
                elif mod == 'veh' and veh_key in batch and batch[veh_key].numel() > 0:
                    veh_out = self.nets['veh'](batch[veh_key].to(device).permute(0, 2, 1), return_pooled=False)
                    veh_out_permuted = veh_out.permute(0, 2, 1)
                    if veh_out_permuted.shape[1] != T_mot:
                        veh_out_permuted = F.interpolate(veh_out_permuted.transpose(1, 2), size=T_mot, mode='linear', align_corners=False).transpose(1, 2)
                    features['motion']['veh'] = self.projs['veh'](veh_out_permuted)
        
                # [버그 수정] 누락되었던 sc, survey, ppg 모달리티 처리 로직 추가
                elif mod == 'sc':
                    sc_pooled = self.nets['sc'](batch.get('sc_motion_evt', batch.get('scenario_evt_e')).to(device), ...)
                    features['motion']['sc'] = self.projs['sc'](sc_pooled).unsqueeze(1).expand(-1, T_mot, -1)
                
                elif mod == 'survey' and 'survey_e' in batch: 
                    survey_pooled = self.projs['survey'](self.nets['survey'](batch['survey_e'].to(device)))
                    features['motion']['survey'] = survey_pooled.unsqueeze(1).expand(-1, T_mot, -1)
                
                elif mod == 'ppg' and 'ppg_emotion' in batch: 
                    ppg_tcn = self.nets['ppg'](batch['ppg_emotion'].to(device).permute(0, 2, 1))
                    combined = torch.cat([ppg_tcn, self._process_hrv(batch, device)], dim=1)
                    ppg_pooled = self.projs['ppg'](combined)
                    features['motion']['ppg'] = ppg_pooled.unsqueeze(1).expand(-1, T_mot, -1)

        # 감정(Emotion) 배치의 경우에만 감정 스트림 특징 추출
        if task_type == 'emotion':
            if hasattr(self, 'emotion_modalities'):
                for mod in self.emotion_modalities:
                    if mod == 'ppg' and 'ppg_emotion' in batch and batch['ppg_emotion'].numel() > 0:
                        ppg_tcn = self.nets['ppg'](batch['ppg_emotion'].to(device).permute(0, 2, 1))
                        combined = torch.cat([ppg_tcn, self._process_hrv(batch, device)], dim=1)
                        features['emotion']['ppg'] = self.projs['ppg'](combined)

                    elif mod == 'veh' and 'veh_emotion' in batch and batch['veh_emotion'].numel() > 0:
                        features['emotion']['veh'] = self.projs['veh'](self.nets['veh'](batch['veh_emotion'].
                                                            to(device).permute(0, 2, 1), return_pooled=True))

                    elif mod == 'sc': features['emotion']['sc'] = self.projs['sc'](self.nets['sc']
                                                            (batch['scenario_evt_e'].to(device), 
                                                            batch['scenario_type_e'].to(device), 
                                                            batch['phase_evt_e'].to(device), 
                                                            batch['scenario_time_e'].to(device)))
                        
                    elif mod == 'imu' and 'imu_motion' in batch: 
                        imu_seq_out = self.nets['imu'](batch['imu_motion'].to(device), 
                                                    (batch['imu_motion'].abs().sum(-1) > 0).sum(1))
                        imu_pooled = imu_seq_out.mean(dim=1)
                        features['emotion']['imu'] = self.projs['imu'](imu_pooled)

                    elif mod == 'survey' and 'survey_e' in batch: 
                        features['emotion']['survey'] = self.projs['survey'](self.nets['survey']
                                                                            (batch['survey_e'].to(device)))

        # --- 2. 초기 Fusion ---
        motion_cat_list = [features['motion'][m] for m in self.motion_modalities if m in features['motion']]
        if not motion_cat_list: 
            return {}
        motion_cat = torch.cat(motion_cat_list, dim=2).permute(0, 2, 1)
        fused_motion_seq = self.motion_feature_fusion(motion_cat)
        motion_tokens = fused_motion_seq.permute(0, 2, 1)

        # --- 3. 분기 처리 ---
        if task_type == 'emotion':
            emotion_cat = torch.cat([features['emotion'][m] for m in self.emotion_modalities if m in features['emotion']], dim=1)
            fused_emotion_vector = self.emotion_feature_fusion(emotion_cat)
            emotion_token = fused_emotion_vector.unsqueeze(1)

            B, T, H = motion_tokens.shape
            delta  = min(self.emo_delay_steps, T-1)
            t_star = max(0, T - 1 - delta)
            left   = max(0, t_star - self.emo_ctx_window + 1)
            right  = t_star + 1
            causal_motion_ctx = motion_tokens[:, left:right, :].detach()

            # --- Bi-directional Cross-Interaction & FiLM ---
            mode = self.cfg.FusionModel.interaction_mode
            # 모든 분기에서 변수가 할당되도록 구조 변경
            if mode == 'cross_attention_film':
                attn_e, _ = self.cross_attn_e2m(emotion_token, causal_motion_ctx, causal_motion_ctx)
                refined_emotion_token = self.cross_ln_e(emotion_token + self.cross_drop_e(attn_e))
                attn_m, _ = self.cross_attn_m2e(motion_tokens, emotion_token.detach(), emotion_token.detach())
                refined_motion_tokens = self.cross_ln_m(motion_tokens + self.cross_drop_m(attn_m))
                
                motion_context_for_emo = refined_emotion_token.squeeze(1)
                gamma_beta_emo = self.emo_film(motion_context_for_emo)
                gamma_emo, beta_emo = gamma_beta_emo.chunk(2, dim=-1)
                
                final_emotion_vector = fused_emotion_vector * (1 + torch.tanh(gamma_emo)) + beta_emo
                final_motion_tokens = refined_motion_tokens

            elif mode == 'cross_attention_only':
                attn_e, _ = self.cross_attn_e2m(emotion_token, causal_motion_ctx, causal_motion_ctx)
                refined_emotion_token = self.cross_ln_e(emotion_token + self.cross_drop_e(attn_e))
                attn_m, _ = self.cross_attn_m2e(motion_tokens, emotion_token.detach(), emotion_token.detach())
                refined_motion_tokens = self.cross_ln_m(motion_tokens + self.cross_drop_m(attn_m))
                
                # Cross-Attention의 결과를 final_emotion_vector와 context로 모두 사용
                final_emotion_vector = refined_emotion_token.squeeze(1)
                motion_context_for_emo = final_emotion_vector
                final_motion_tokens = refined_motion_tokens

            elif mode == 'film_only':
                motion_context_for_emo = causal_motion_ctx.mean(dim=1)
                gamma_beta_emo = self.emo_film(motion_context_for_emo)
                gamma_emo, beta_emo = gamma_beta_emo.chunk(2, dim=-1)
                final_emotion_vector = fused_emotion_vector * (1 + torch.tanh(gamma_emo)) + beta_emo

                # gamma_beta_mot = self.mot_film(fused_emotion_vector.detach())
                # gamma_mot, beta_mot = gamma_beta_mot.unsqueeze(1).chunk(2, dim=-1)
                # final_motion_tokens = motion_tokens * (1 + torch.tanh(gamma_mot)) + beta_mot
                final_motion_tokens = motion_tokens
                
            else:
                # 기본값 또는 모드가 잘못 지정된 경우 (상호작용 없음)
                final_emotion_vector = fused_emotion_vector
                final_motion_tokens = motion_tokens

        
            # --- 4. Prediction Heads ---
            mot_logits = self.motion_head(final_motion_tokens.permute(0, 2, 1))

            if self.emo_use_mot_logits:
                p_mot = F.softmax(mot_logits, dim=-1).mean(dim=1).detach()
                emotion_cat_plus = torch.cat([emotion_cat, p_mot], dim=1)
            else:
                emotion_cat_plus = emotion_cat

            emotion_input_dict = {'fused': emotion_cat_plus.unsqueeze(1)}
            if 'survey' in self.cfg.PretrainEmotion.modalities_to_use and 'survey' in features['emotion']:
                emotion_input_dict['static'] = features['emotion']['survey']

            valence_logits = self.emotion_valence_predictor(emotion_input_dict, context=motion_context_for_emo)
            arousal_logits = self.emotion_arousal_predictor(emotion_input_dict, context=motion_context_for_emo)

            return {
                'motion_logits': mot_logits, 'valence_logits': valence_logits, 'arousal_logits': arousal_logits,
                'fused_motion': final_motion_tokens.mean(dim=1), 'fused_emotion': final_emotion_vector
            }
        
        else: # task_type == motion
            mot_logits = self.motion_head(motion_tokens.permute(0, 2, 1))
            return {'motion_logits': mot_logits}

    
    # run_epoch 함수를 아래 코드로 전체 교체하세요.

    def run_epoch(self, motion_loader, emotion_loader, train: bool):
        self.train(train)
        total_loss, num_batches = 0.0, 0
        device = self.cfg.Project.device

        pbar = tqdm(zip(motion_loader, emotion_loader), 
                    total=min(len(motion_loader), len(emotion_loader)),
                    desc=f"[{'Train' if train else 'Valid'}] Asymmetric Fusion")
        
        for motion_batch, emotion_batch in pbar:
            # --- 1. 행동(Motion) 배치 독립 학습 ---
            with torch.set_grad_enabled(train):
                out_mot = self.forward(motion_batch, task_type='motion')
                loss_mot = torch.tensor(0., device=device)
                if out_mot and 'motion_logits' in out_mot and out_mot['motion_logits'].numel() > 0:
                    mot_logits, mot_labels = out_mot['motion_logits'], motion_batch['label_motion'].to(device)
                    valid_mask_mot = (mot_labels.reshape(-1) > 0) & (mot_labels.reshape(-1) != 4)
                    if valid_mask_mot.any():
                        loss_mot = F.cross_entropy(mot_logits.reshape(-1, mot_logits.shape[-1])[valid_mask_mot], (mot_labels.reshape(-1)[valid_mask_mot] - 1).long())

            if train and loss_mot.requires_grad:
                # [핵심 수정] 행동(Motion) 손실에 대해 즉시 역전파 및 업데이트
                self.optim.zero_grad()
                self.scaler.scale(loss_mot).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.scaler.step(self.optim)
                self.scaler.update()

            # --- 2. 감정(Emotion) 배치 상호 학습 ---
            with torch.set_grad_enabled(train):
                out_emo = self.forward(emotion_batch, task_type='emotion')
                loss_emo_group = torch.tensor(0., device=device)
                loss_mot_from_emo = torch.tensor(0., device=device)

                if out_emo and 'valence_logits' in out_emo:
                    # ========================= [핵심 수정 부분 START] =========================
                    # 모델은 시퀀스당 하나의 값을 예측하므로, 라벨도 시퀀스에서 하나의 값만 선택합니다.
                    # 여기서는 라벨 시퀀스의 마지막 값([-1])을 사용합니다.
                    val_logits = out_emo['valence_logits']
                    raw_v = emotion_batch['valence_reg_emotion'].to(device)[:, -1] # .view(-1) 대신 [:, -1] 사용

                    # Arousal도 동일하게 수정합니다.
                    aro_logits = out_emo['arousal_logits']
                    raw_a = emotion_batch['arousal_reg_emotion'].to(device)[:, -1] # .view(-1) 대신 [:, -1] 사용
                    # ========================== [핵심 수정 부분 END] ==========================

                    # 감정 손실 계산
                    mask_v = (raw_v >= 1) & (raw_v < 10)
                    tgt_v = torch.full_like(raw_v, -100, dtype=torch.long)
                    if mask_v.any(): tgt_v[mask_v] = raw_v[mask_v].long() - 1
                    loss_v = F.cross_entropy(val_logits, tgt_v, ignore_index=-100)

                    mask_a = (raw_a >= 1) & (raw_a < 10)
                    tgt_a = torch.full_like(raw_a, -100, dtype=torch.long)
                    if mask_a.any(): tgt_a[mask_a] = raw_a[mask_a].long() - 1
                    loss_a = F.cross_entropy(aro_logits, tgt_a, ignore_index=-100)
                    
                    loss_align = F.mse_loss(self.alignment_head(out_emo['fused_motion'].detach()), out_emo['fused_emotion'])
                    loss_emo_group = (loss_v + loss_a) * self.cfg.MainTask.lambda_emotion + loss_align * self.cfg.MainTask.cross_modal_lambda
                    
                    # 감정 배치로부터 나온 행동 손실 계산
                    mot_logits_from_emo, mot_labels_from_emo = out_emo['motion_logits'], emotion_batch['label_motion'].to(device)
                    valid_mask_emo_mot = (mot_labels_from_emo.reshape(-1) > 0) & (mot_labels_from_emo.reshape(-1) != 4)
                    if valid_mask_emo_mot.any():
                        loss_mot_from_emo = F.cross_entropy(mot_logits_from_emo.reshape(-1, mot_logits_from_emo.shape[-1])[valid_mask_emo_mot], (mot_labels_from_emo.reshape(-1)[valid_mask_emo_mot] - 1).long())
            
            # 감정 배치에서 발생한 모든 손실을 합산
            total_emo_loss = loss_emo_group + loss_mot_from_emo
            if train and total_emo_loss.requires_grad:
                self.optim.zero_grad()
                self.scaler.scale(total_emo_loss).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.scaler.step(self.optim)
                self.scaler.update()

            # 전체 손실은 리포팅 용도로만 합산
            total_loss += loss_mot.item() + total_emo_loss.item()
            num_batches += 1
        
        return total_loss / (num_batches * 2) if num_batches > 0 else 0.0 # 스텝이 2번이므로 *2
    
    def run_epoch_for_validation(self, loader):
        self.eval()
        total_loss, num_batches = 0.0, 0
        device = self.cfg.Project.device
        for batch in tqdm(loader, desc="[Valid] Full Fusion"):
            with torch.no_grad():
                out = self.forward(batch, task_type='emotion') # 검증은 항상 풀-퓨전 모드
                if not out: continue

                # ========================= [핵심 수정 부분 START] =========================
                # 예상치 못한 라벨 값에 의한 CUDA 오류를 방지하기 위해 안전한 라벨 처리 로직으로 변경
                
                # Valence 처리
                raw_v = batch['valence_reg_emotion'].to(device)[:, -1]
                mask_v = (raw_v >= 1) & (raw_v < 10)
                valence_target = torch.full_like(raw_v, -100, dtype=torch.long) # 기본값을 ignore_index로 설정
                valence_target[mask_v] = raw_v[mask_v].long() - 1 # 유효한 라벨만 0~8로 변환
                
                loss_v = F.cross_entropy(out['valence_logits'], valence_target, ignore_index=-100)

                # Arousal 처리
                raw_a = batch['arousal_reg_emotion'].to(device)[:, -1]
                mask_a = (raw_a >= 1) & (raw_a < 10)
                arousal_target = torch.full_like(raw_a, -100, dtype=torch.long) # 기본값을 ignore_index로 설정
                arousal_target[mask_a] = raw_a[mask_a].long() - 1 # 유효한 라벨만 0~8로 변환

                loss_a = F.cross_entropy(out['arousal_logits'], arousal_target, ignore_index=-100)
                # ========================== [핵심 수정 부분 END] ==========================

                loss_align = F.mse_loss(self.alignment_head(out['fused_motion']), out['fused_emotion'])
                loss_emo_group = (loss_v + loss_a) * self.cfg.MainTask.lambda_emotion + loss_align * self.cfg.MainTask.cross_modal_lambda

                mot_logits, mot_labels = out['motion_logits'], batch['label_motion'].to(device)
                valid_mask_mot = (mot_labels.reshape(-1) > 0) & (mot_labels.reshape(-1) != 4)
                loss_mot = F.cross_entropy(mot_logits.reshape(-1, mot_logits.shape[-1])[valid_mask_mot], (mot_labels.reshape(-1)[valid_mask_mot] - 1).long()) if valid_mask_mot.any() else torch.tensor(0., device=device)
                
                loss = loss_mot + loss_emo_group
                
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    @torch.no_grad()
    def evaluate(self, loader, return_preds=False):
        self.eval()
        all_preds = {'motion_preds': [], 'motion_trues': [], 'valence_preds': [], 'valence_trues': [], 'arousal_preds': [], 'arousal_trues': []}
        for batch in loader:
            out = self.forward(batch, task_type='emotion') 
            p_mot = out['motion_logits'].argmax(-1).cpu(); t_mot_raw = batch['label_motion'].cpu()
            mask_mot = (t_mot_raw > 0) & (t_mot_raw != 4)
            all_preds['motion_preds'].append(p_mot[mask_mot]); all_preds['motion_trues'].append(t_mot_raw[mask_mot] - 1)
            
            # ========================= [Valence 핵심 수정 부분 START] =========================
            # 모델 예측 (크기: B)
            p_v_multiclass = out['valence_logits'].argmax(-1).cpu()

            # 정답 라벨도 시퀀스 전체(.view(-1))가 아닌 마지막 값([:, -1])만 사용 (크기: B)
            raw_v = batch['valence_reg_emotion'][:, -1].cpu() 
            
            mask_v = (raw_v >= 1) & (raw_v < 10)
            t_v_multiclass = torch.full_like(raw_v, -1, dtype=torch.long)
            if mask_v.any():
                t_v_multiclass[mask_v] = raw_v[mask_v].long() - 1

            # 3진으로 변환하여 평가 (이제 모든 텐서의 크기가 B로 동일)
            p_v_ternary = torch.full_like(p_v_multiclass, -1)
            p_v_ternary[(p_v_multiclass >= 0) & (p_v_multiclass <= 2)] = 0
            p_v_ternary[(p_v_multiclass >= 3) & (p_v_multiclass <= 5)] = 1
            p_v_ternary[(p_v_multiclass >= 6) & (p_v_multiclass <= 8)] = 2
            
            t_v_ternary = torch.full_like(t_v_multiclass, -1)
            t_v_ternary[(t_v_multiclass >= 0) & (t_v_multiclass <= 2)] = 0
            t_v_ternary[(t_v_multiclass >= 3) & (t_v_multiclass <= 5)] = 1
            t_v_ternary[(t_v_multiclass >= 6) & (t_v_multiclass <= 8)] = 2

            valid_mask_v = t_v_ternary != -1
            all_preds['valence_preds'].append(p_v_ternary[valid_mask_v])
            all_preds['valence_trues'].append(t_v_ternary[valid_mask_v])
            # ========================== [Valence 핵심 수정 부분 END] ==========================


            # ========================= [Arousal 핵심 수정 부분 START] =========================
            # 모델 예측 (크기: B)
            p_a_multiclass = out['arousal_logits'].argmax(-1).cpu()

            # 정답 라벨도 시퀀스 전체(.view(-1))가 아닌 마지막 값([:, -1])만 사용 (크기: B)
            raw_a = batch['arousal_reg_emotion'][:, -1].cpu()

            mask_a = (raw_a >= 1) & (raw_a < 10)
            t_a_multiclass = torch.full_like(raw_a, -1, dtype=torch.long)
            if mask_a.any():
                t_a_multiclass[mask_a] = raw_a[mask_a].long() - 1

            # 3진으로 변환하여 평가 (이제 모든 텐서의 크기가 B로 동일)
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
            # ========================== [Arousal 핵심 수정 부분 END] ==========================
            
        for key in all_preds: 
            if all_preds[key]: # 리스트가 비어있지 않은 경우에만 cat 수행
                all_preds[key] = torch.cat(all_preds[key])
                
        if return_preds: return all_preds
        
        # 리스트가 비어있을 경우를 대비한 안전장치
        acc_mot = accuracy_score(all_preds['motion_trues'], all_preds['motion_preds']) if len(all_preds['motion_trues']) > 0 else 0.0
        acc_v = accuracy_score(all_preds['valence_trues'], all_preds['valence_preds']) if len(all_preds['valence_trues']) > 0 else 0.0
        acc_a = accuracy_score(all_preds['arousal_trues'], all_preds['arousal_preds']) if len(all_preds['arousal_trues']) > 0 else 0.0
        
        return acc_mot, acc_v, acc_a
            
    # 2단계 학습을 총괄하는 메인 훈련 함수
    def fusion_train(self, save_path="weights/best_fusion_v28_assym.pt"):
        tr_motion_loader = make_motion_loader(self.cfg, self.train_keys, shuffle=True, dp=self)
        tr_emotion_loader = make_emotion_loader(self.cfg, self.train_keys, shuffle=True, dp=self)
        va_loader = make_multitask_loader(self.cfg, self.val_keys, shuffle=False, dp=self)       
        
        warmup_epochs = getattr(self.cfg.MainTask, 'warmup_epochs', 10)
        total_epochs = self.cfg.MainTask.epochs
        
        # 어떤 지표를 기준으로 최고 모델을 저장할지 설정 ('accuracy' 또는 'loss')
        metric_to_monitor = 'loss'
        
        # 선택된 지표에 따라 best_score 초기값을 설정합니다.
        if metric_to_monitor == 'loss':
            best_score = float('inf')
            print(f"\n Monitoring Validation Loss to find the best model. Lower is better.")
        else:
            best_score = 0.0
            print(f"\n Monitoring Validation Accuracy to find the best model. Higher is better.")
            
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
                tr_loss = self.run_epoch(tr_motion_loader, tr_emotion_loader, train=True)
                va_loss = self.run_epoch_for_validation(va_loader)
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

        for epoch in range(1, total_epochs + 1):
            # 훈련/검증 루프 호출 방식 변경
            tr_loss = self.run_epoch(tr_motion_loader, tr_emotion_loader, train=True)
            va_loss = self.run_epoch_for_validation(va_loader) # 검증 전용 함수 사용
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