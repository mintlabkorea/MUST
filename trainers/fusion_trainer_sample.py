"""
v28+tot/act (e2e version)
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

        self.tot_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.act_loss_fn = nn.MSELoss(reduction='none')
        
        mode = self.cfg.FusionModel.interaction_mode
        print(f"Fine-tuning with {mode}")

    def _build_model(self):
        print("\n--- Building V30 model architecture with 2-Stage Training logic ---")
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

        final_fusion_dim = self.cfg.FusionModel.hidden_dim * 2
        
        # TOT Head (분류)
        num_tot_classes = 3  # 0: 빠름, 1: 보통, 2: 느림
        self.tot_head = nn.Sequential(
            nn.Linear(final_fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_tot_classes)
        )
        
        # ACT Head (회귀)
        self.act_head = nn.Sequential(
            nn.Linear(final_fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 1개의 연속적인 값 예측
        )
        # 감정/행동 fusion/헤드
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
        init_log_vars = torch.tensor([0.0] * 6)
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
            mot_ckpt_path = getattr(mot_cfg, 'ckpt_path', 'weights/best_motion.pt')
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

    def forward(self, batch):
        device = self.cfg.Project.device
        features = {'motion': {}, 'emotion': {}}
        
        if 'imu_motion' in batch and batch['imu_motion'] is not None:
            T_mot = batch['imu_motion'].shape[1]
        else:
            T_mot = 1000

         # --- 1. Encoder & Projection ---
        if hasattr(self, 'motion_modalities'):
            for mod in self.motion_modalities:
                if mod == 'imu' and 'imu_motion' in batch: 
                    features['motion']['imu'] = self.projs['imu'](self.nets['imu'](batch['imu_motion'].
                                                to(device), (batch['imu_motion'].abs().sum(-1) > 0).sum(1)))
                
                elif mod == 'veh' and 'veh_motion' in batch: 
                    features['motion']['veh'] = self.projs['veh'](self.nets['veh'](batch['veh_motion'].
                    to(device).permute(0, 2, 1), return_pooled=True)).unsqueeze(1).expand(-1, T_mot, -1)
                
                elif mod == 'sc': sc_pooled = self.nets['sc'](batch['sc_motion_evt'].
                        to(device), batch['sc_motion_type'].to(device), batch['sc_motion_phase'].to(device), 
                        batch['sc_motion_time'].to(device)); features['motion']['sc'] = self.projs['sc'](sc_pooled.unsqueeze(1).expand(-1, T_mot, -1))
                
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
        motion_cat = torch.cat([features['motion'][m] for m in self.motion_modalities], dim=2).permute(0, 2, 1)
        fused_motion_seq = self.motion_feature_fusion(motion_cat)
        
        emotion_cat = torch.cat([features['emotion'][m] for m in self.emotion_modalities], dim=1)
        fused_emotion_vector = self.emotion_feature_fusion(emotion_cat)

        B, H, T = fused_motion_seq.shape

        # --- 3. Bi-directional Cross-Interaction & FiLM ---
        motion_tokens = fused_motion_seq.permute(0, 2, 1) # (B, T, H)
        emotion_token = fused_emotion_vector.unsqueeze(1) # (B, 1, H)

        delta  = min(self.emo_delay_steps, T-1)
        t_star = max(0, T - 1 - delta)
        left   = max(0, t_star - self.emo_ctx_window + 1)
        right  = t_star + 1
        causal_motion_ctx = motion_tokens[:, left:right, :].detach()

        # Config 값에 따라 다른 상호작용 로직 실행
        mode = self.cfg.FusionModel.interaction_mode
        # [수정] 모든 분기에서 변수가 할당되도록 구조 변경
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
        motion_input_for_head = final_motion_tokens.permute(0, 2, 1)
        mot_logits = self.motion_head(motion_input_for_head)

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

        # --- 5. TOT / ACT Prediction Heads ---
        # [수정] 시퀀스 길이를 다시 계산할 필요 없이, 확정된 T를 사용합니다.
        expanded_emotion = final_emotion_vector.unsqueeze(1).expand(-1, T, -1)
        final_fused_features = torch.cat([final_motion_tokens, expanded_emotion], dim=-1)

        act_pred = self.act_head(final_fused_features).squeeze(-1)
        last_step_features = final_fused_features[:, -1, :]
        tot_logits = self.tot_head(last_step_features)
        
        # 최종 반환 딕셔너리
        return {
            'motion_logits': mot_logits, 'valence_logits': valence_logits, 'arousal_logits': arousal_logits,
            'fused_motion': final_motion_tokens.mean(dim=1),
            'fused_emotion': final_emotion_vector,
            'tot_logits': tot_logits, 
            'act_pred': act_pred    
        }

    def _calculate_act_loss(self, pred, true):
        # The shapes of 'pred' and 'true' should now both be (B, L)
        mask = true != -100.0
        
        if not mask.any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Now boolean indexing will work correctly
        loss_per_sample = self.act_loss_fn(pred[mask], true[mask])
        return loss_per_sample.mean()
    
    def run_epoch(self, loader, train: bool):
        self.train(train)
        total_loss, num_batches = 0.0, 0
        device = self.cfg.Project.device

        for batch in loader:
            # batch 내 텐서들을 device로 이동
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            with torch.set_grad_enabled(train):
                out = self.forward(batch) 
               
                # NDRT (Motion) 손실
                loss_mot = torch.tensor(0., device=device)
                if batch.get('label_motion') is not None:
                    mot_logits, mot_labels = out['motion_logits'], batch['label_motion']
                    valid_mask = (mot_labels.reshape(-1) > 0) & (mot_labels.reshape(-1) != 4)
                    if valid_mask.any():
                        loss_mot = F.cross_entropy(mot_logits.reshape(-1, mot_logits.shape[-1])[valid_mask], (mot_labels.reshape(-1)[valid_mask] - 1).long())

                # Valence 손실
                loss_v = torch.tensor(0., device=device)
                if batch.get('valence_reg_emotion') is not None:
                    loss_v = F.cross_entropy(out['valence_logits'], (batch['valence_reg_emotion'].view(-1) - 1).long(), ignore_index=-101)

                # Arousal 손실
                loss_a = torch.tensor(0., device=device)
                if batch.get('arousal_reg_emotion') is not None:
                    loss_a = F.cross_entropy(out['arousal_logits'], (batch['arousal_reg_emotion'].view(-1) - 1).long(), ignore_index=-101)
                                
                # Alignment (상호 정렬) 손실
                loss_align = F.mse_loss(self.alignment_head(out['fused_motion'].detach()), out['fused_emotion'])

                # TOT 손실
                loss_tot = torch.tensor(0., device=device)
                if batch.get('label_tot') is not None:
                    loss_tot = self.tot_loss_fn(out['tot_logits'], batch['label_tot'])

                # ACT 손실
                if batch.get('label_act') is not None:
                    # Squeeze the label tensor from (B, L, 1) to (B, L)
                    act_true_labels = batch['label_act'].squeeze(-1) 
                    loss_act = self._calculate_act_loss(out['act_pred'], act_true_labels)
                else:
                    loss_act = torch.tensor(0., device=device)

                # --- 최종 손실 취합 ---
                emo_loss_weight = self.cfg.MainTask.lambda_emotion
                loss_v_weighted = emo_loss_weight * loss_v
                loss_a_weighted = emo_loss_weight * loss_a
                
                if self.cfg.MainTask.use_uncertainty_loss:
                    loss = (torch.exp(-self.log_vars[0]) * loss_mot + 0.5 * self.log_vars[0]) + \
                           (torch.exp(-self.log_vars[1]) * loss_v_weighted + 0.5 * self.log_vars[1]) + \
                           (torch.exp(-self.log_vars[2]) * loss_a_weighted + 0.5 * self.log_vars[2]) + \
                           (torch.exp(-self.log_vars[3]) * loss_align * self.cfg.MainTask.cross_modal_lambda + 0.5 * self.log_vars[3]) + \
                           (torch.exp(-self.log_vars[4]) * loss_tot * self.cfg.MainTask.lambda_tot + 0.5 * self.log_vars[4]) + \
                           (torch.exp(-self.log_vars[5]) * loss_act * self.cfg.MainTask.lambda_act + 0.5 * self.log_vars[5])
                else:
                    loss = (loss_mot + 
                            emo_loss_weight * (loss_v + loss_a) + 
                            loss_align * self.cfg.MainTask.cross_modal_lambda +
                            loss_tot * self.cfg.MainTask.lambda_tot +
                            loss_act * self.cfg.MainTask.lambda_act) 

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
        all_preds = {
            'motion_preds': [], 'motion_trues': [], 'valence_preds': [], 'valence_trues': [],
            'arousal_preds': [], 'arousal_trues': [], 'tot_preds': [], 'tot_trues': []
        }
        total_act_mse, act_samples = 0.0, 0

        for batch in loader:
            # 1. Move data to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            # 2. Forward pass
            out = self.forward(batch)

            # 3. Process each task's results
            # Motion (NDRT) --- Includes special debugging for this issue
            if batch.get('label_motion') is not None:
                p_mot = out['motion_logits'].argmax(-1).cpu()
                t_mot_raw = batch['label_motion'].cpu()

                # --- [CRUCIAL DEBUGGING STEP] Check the motion labels ---
                print("\n[Debug] Motion Task:")
                unique_labels, counts = torch.unique(t_mot_raw, return_counts=True)
                label_dist = {label.item(): count.item() for label, count in zip(unique_labels, counts)}
                print(f"  - Label Distribution in Batch: {label_dist}")

                mask_mot = (t_mot_raw > 0) & (t_mot_raw != 4)
                print(f"  - Number of valid labels (not 0 or 4): {mask_mot.sum().item()}")

                if mask_mot.any():
                    print("  -> SUCCESS: Found valid motion labels. Appending to list.")
                    all_preds['motion_preds'].append(p_mot[mask_mot])
                    all_preds['motion_trues'].append(t_mot_raw[mask_mot] - 1)
                else:
                    print("  -> INFO: No valid motion labels found in this batch. Skipping.")
            else:
                 print("\n[Debug] Motion Task: - SKIP: 'label_motion' key is None.")

            # Valence
            if batch.get('valence_reg_emotion') is not None:
                p_v_multiclass = out['valence_logits'].argmax(-1).cpu()
                raw_v = batch['valence_reg_emotion'].view(-1).cpu()
                mask_v = (raw_v >= 1) & (raw_v < 10)

                t_v_multiclass = torch.full_like(raw_v, -1, dtype=torch.long)
                t_v_multiclass[mask_v] = raw_v[mask_v].long() - 1

                p_v_ternary = torch.full_like(p_v_multiclass, -1)
                p_v_ternary[(p_v_multiclass >= 0) & (p_v_multiclass <= 2)] = 0
                p_v_ternary[(p_v_multiclass >= 3) & (p_v_multiclass <= 5)] = 1
                p_v_ternary[(p_v_multiclass >= 6) & (p_v_multiclass <= 8)] = 2

                t_v_ternary = torch.full_like(t_v_multiclass, -1)
                t_v_ternary[(t_v_multiclass >= 0) & (t_v_multiclass <= 2)] = 0
                t_v_ternary[(t_v_multiclass >= 3) & (t_v_multiclass <= 5)] = 1
                t_v_ternary[(t_v_multiclass >= 6) & (t_v_multiclass <= 8)] = 2

                valid_mask_v = t_v_ternary != -1
                if valid_mask_v.any():
                    all_preds['valence_preds'].append(p_v_ternary[valid_mask_v])
                    all_preds['valence_trues'].append(t_v_ternary[valid_mask_v])

            # Arousal
            if batch.get('arousal_reg_emotion') is not None:
                p_a_multiclass = out['arousal_logits'].argmax(-1).cpu()
                raw_a = batch['arousal_reg_emotion'].view(-1).cpu()
                mask_a = (raw_a >= 1) & (raw_a < 10)

                t_a_multiclass = torch.full_like(raw_a, -1, dtype=torch.long)
                t_a_multiclass[mask_a] = raw_a[mask_a].long() - 1

                p_a_ternary = torch.full_like(p_a_multiclass, -1)
                p_a_ternary[(p_a_multiclass >= 0) & (p_a_multiclass <= 2)] = 0
                p_a_ternary[(p_a_multiclass >= 3) & (p_a_multiclass <= 5)] = 1
                p_a_ternary[(p_a_multiclass >= 6) & (p_a_multiclass <= 8)] = 2

                t_a_ternary = torch.full_like(t_a_multiclass, -1)
                t_a_ternary[(t_a_multiclass >= 0) & (t_a_multiclass <= 2)] = 0
                t_a_ternary[(t_a_multiclass >= 3) & (t_a_multiclass <= 5)] = 1
                t_a_ternary[(t_a_multiclass >= 6) & (t_a_multiclass <= 8)] = 2

                valid_mask_a = t_a_ternary != -1
                if valid_mask_a.any():
                    all_preds['arousal_preds'].append(p_a_ternary[valid_mask_a])
                    all_preds['arousal_trues'].append(t_a_ternary[valid_mask_a])

            # TOT (Time-On-Task)
            if batch.get('label_tot') is not None:
                mask_tot = batch['label_tot'].cpu() != -100
                if mask_tot.any():
                    all_preds['tot_preds'].append(out['tot_logits'].argmax(-1).cpu()[mask_tot])
                    all_preds['tot_trues'].append(batch['label_tot'].cpu()[mask_tot])

            # ACT (Action)
            if batch.get('label_act') is not None:
                act_pred = out['act_pred'].cpu()
                act_true = batch['label_act'].cpu().squeeze(-1)

                mask_act = act_true != -100.0
                if mask_act.any():
                    mse = F.mse_loss(act_pred[mask_act], act_true[mask_act])
                    total_act_mse += mse.item() * mask_act.sum().item()
                    act_samples += mask_act.sum().item()

        # 4. Concatenate results from all batches
        for key in all_preds:
            if all_preds[key]:
                all_preds[key] = torch.cat(all_preds[key])

        if return_preds:
            return all_preds

        # 5. Calculate final metrics
        acc_mot = accuracy_score(all_preds['motion_trues'], all_preds['motion_preds']) if len(all_preds['motion_trues']) > 0 else 0.0
        acc_v = accuracy_score(all_preds['valence_trues'], all_preds['valence_preds']) if len(all_preds['valence_trues']) > 0 else 0.0
        acc_a = accuracy_score(all_preds['arousal_trues'], all_preds['arousal_preds']) if len(all_preds['arousal_trues']) > 0 else 0.0
        acc_tot = accuracy_score(all_preds['tot_trues'], all_preds['tot_preds']) if len(all_preds['tot_trues']) > 0 else 0.0
        mse_act = total_act_mse / act_samples if act_samples > 0 else 0.0

        return acc_mot, acc_v, acc_a, acc_tot, mse_act

    
    @torch.no_grad()
    def evaluate(self, loader, return_preds=False):
        self.eval()
        all_preds = {
            'motion_preds': [], 'motion_trues': [], 'valence_preds': [], 'valence_trues': [], 
            'arousal_preds': [], 'arousal_trues': [], 'tot_preds': [], 'tot_trues': []
        }
        total_act_mse, act_samples = 0.0, 0
        
        # [디버깅] 배치 번호 추적
        batch_idx = 0

        for batch in loader:
            print(f"\n--- [Debug] Evaluating Validation Batch #{batch_idx} ---")
            
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            out = self.forward(batch)

            # --- [Debug] Valence 라벨 처리 과정 추적 ---
            print("[Debug] Valence Task:")
            if batch.get('valence_reg_emotion') is not None:
                raw_v = batch['valence_reg_emotion'].view(-1).cpu()
                mask_v = (raw_v >= 1) & (raw_v < 10)
                print(f"  - Raw Valence Labels (first 10): {raw_v[:10].tolist()}")
                print(f"  - Valid Labels (1~9) found in batch: {mask_v.sum().item()}")

                # ... (Valence 3진 분류 변환 로직은 동일) ...
                t_v_multiclass = torch.full_like(raw_v, -1, dtype=torch.long)
                t_v_multiclass[mask_v] = raw_v[mask_v].long() - 1
                t_v_ternary = torch.full_like(t_v_multiclass, -1)
                t_v_ternary[(t_v_multiclass >= 0) & (t_v_multiclass <= 2)] = 0
                t_v_ternary[(t_v_multiclass >= 3) & (t_v_multiclass <= 5)] = 1
                t_v_ternary[(t_v_multiclass >= 6) & (t_v_multiclass <= 8)] = 2
                valid_mask_v = t_v_ternary != -1

                if valid_mask_v.sum().item() > 0:
                    print(f"  -> SUCCESS: Found {valid_mask_v.sum().item()} valid ternary labels. Appending to list.")
                    # ... (p_v_ternary 계산 로직) ...
                    p_v_multiclass = out['valence_logits'].argmax(-1).cpu()
                    p_v_ternary = torch.full_like(p_v_multiclass, -1)
                    p_v_ternary[(p_v_multiclass >= 0) & (p_v_multiclass <= 2)] = 0
                    p_v_ternary[(p_v_multiclass >= 3) & (p_v_multiclass <= 5)] = 1
                    p_v_ternary[(p_v_multiclass >= 6) & (p_v_multiclass <= 8)] = 2
                    all_preds['valence_preds'].append(p_v_ternary[valid_mask_v])
                    all_preds['valence_trues'].append(t_v_ternary[valid_mask_v])
                else:
                    print("  -> INFO: No valid ternary labels to append in this batch.")
            else:
                print("  - SKIP: 'valence_reg_emotion' key is None.")
            
            # --- [Debug] TOT 라벨 처리 과정 추적 ---
            print("[Debug] TOT Task:")
            if batch.get('label_tot') is not None:
                tot_true_labels = batch['label_tot'].cpu()
                mask_tot = tot_true_labels != -100
                print(f"  - Raw TOT Labels (first 10): {tot_true_labels[:10].tolist()}")
                print(f"  - Valid Labels (not -100) found in batch: {mask_tot.sum().item()}")

                if mask_tot.any():
                    print(f"  -> SUCCESS: Found {mask_tot.sum().item()} valid labels. Appending to list.")
                    all_preds['tot_preds'].append(out['tot_logits'].argmax(-1).cpu()[mask_tot])
                    all_preds['tot_trues'].append(tot_true_labels[mask_tot])
                else:
                    print("  -> INFO: No valid labels to append in this batch.")
            else:
                print("  - SKIP: 'label_tot' key is None.")

            # (Arousal, Motion, ACT는 일단 생략)
            batch_idx += 1

            # Arousal
            if batch.get('arousal_reg_emotion') is not None:
                p_a_multiclass = out['arousal_logits'].argmax(-1).cpu()
                raw_a = batch['arousal_reg_emotion'].view(-1).cpu()
                mask_a = (raw_a >= 1) & (raw_a < 10)
                t_a_multiclass = torch.full_like(raw_a, -1, dtype=torch.long)
                t_a_multiclass[mask_a] = raw_a[mask_a].long() - 1

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

            # TOT
            if batch.get('label_tot') is not None:
                mask_tot = batch['label_tot'].cpu() != -100
                if mask_tot.any():
                    all_preds['tot_preds'].append(out['tot_logits'].argmax(-1).cpu()[mask_tot])
                    all_preds['tot_trues'].append(batch['label_tot'].cpu()[mask_tot])

            # ACT
            if batch.get('label_act') is not None:
                act_pred = out['act_pred'].cpu()
                act_true_raw = batch['label_act'].cpu()
                
                # [FIX] Squeeze the label tensor from (B, L, 1) to (B, L)
                act_true = act_true_raw.squeeze(-1)
                
                mask_act = act_true != -100.0
                if mask_act.any():
                    mse = F.mse_loss(act_pred[mask_act], act_true[mask_act])
                    total_act_mse += mse.item() * mask_act.sum().item()
                    act_samples += mask_act.sum().item()
            
        # [디버깅] 최종 리스트 길이 출력
        print("\n--- [Debug] Final list lengths before accuracy calculation ---")
        for key, value in all_preds.items():
            print(f"  - Length of all_preds['{key}']: {len(value)}")

        for key in all_preds: 
            if all_preds[key]: all_preds[key] = torch.cat(all_preds[key])

        if return_preds: return all_preds
        
        # [최종 수정] accuracy_score의 인자를 (실제값, 예측값)으로 모두 수정합니다.
        acc_mot = accuracy_score(all_preds['motion_trues'], all_preds['motion_preds']) if len(all_preds['motion_trues']) > 0 else 0.0
        acc_v = accuracy_score(all_preds['valence_trues'], all_preds['valence_preds']) if len(all_preds['valence_trues']) > 0 else 0.0
        acc_a = accuracy_score(all_preds['arousal_trues'], all_preds['arousal_preds']) if len(all_preds['arousal_trues']) > 0 else 0.0
        acc_tot = accuracy_score(all_preds['tot_trues'], all_preds['tot_preds']) if len(all_preds['tot_trues']) > 0 else 0.0
        mse_act = total_act_mse / act_samples if act_samples > 0 else 0.0
        
        return acc_mot, acc_v, acc_a, acc_tot, mse_act
            
    # 2단계 학습을 총괄하는 메인 훈련 함수
    def fusion_train(self, save_path="weights/best_fusion_v30.pt"): 
        tr_loader = make_multitask_loader(self.cfg, self.train_keys, shuffle=True, dp=self)
        va_loader = make_multitask_loader(self.cfg, self.val_keys, shuffle=False, dp=self)
        
        warmup_epochs = getattr(self.cfg.MainTask, 'warmup_epochs', 10)
        total_epochs = self.cfg.MainTask.epochs
        
        # best_loss, patience_counter, best_performance = float('inf'), 0, {}
        best_val_loss, patience_counter, best_performance = float('inf'), 0, {}

        # --- 1단계: 워밍업 (Warm-up) ---
        if warmup_epochs > 0:
            print("\n" + "="*50)
            print(f"🚀 STARTING STAGE 1: WARM-UP FOR {warmup_epochs} EPOCHS")
            print("="*50)
            self._set_encoder_grad(requires_grad=False)
            
            # 옵티마이저를 여기서 단 한 번만 생성합니다.
            self._create_optimizer_stage1()

            for epoch in range(1, warmup_epochs + 1):
                self.current_epoch = epoch
                tr_loss = self.run_epoch(tr_loader, train=True)
                va_loss = self.run_epoch(va_loader, train=False)
                va_acc_mot, va_acc_v, va_acc_a, va_acc_tot, va_mse_act = self.evaluate(va_loader)
                print(f"[WARM-UP] Epoch {epoch:02d}/{warmup_epochs} | Tr Loss {tr_loss:.4f} | Val Loss {va_loss:.4f}")
                print(f"  -> Acc(M/V/A/T): {va_acc_mot:.3f}/{va_acc_v:.3f}/{va_acc_a:.3f}/{va_acc_tot:.3f} | ACT_MSE: {va_mse_act:.4f}")

                # [수정] 검증 손실(va_loss)이 개선되었는지 확인
                if va_loss < best_val_loss:
                    best_val_loss, patience_counter = va_loss, 0 # [수정] best_val_loss 업데이트
                    best_performance = {'mot_acc': va_acc_mot, 'val_acc': va_acc_v, 'aro_acc': va_acc_a, 'best_val_loss': best_val_loss}
                    torch.save({'model_state_dict': self.state_dict()}, save_path)
                    print(f"  -> Best warm-up model saved to {save_path} (Val Loss: {best_val_loss:.4f})") # [수정] 출력 메시지 변경
                else:
                    patience_counter += 1


        # --- 2단계: 미세 조정 (Fine-tuning) ---
        print("\n" + "="*50)
        print(f"🚀 STARTING STAGE 2: FINE-TUNING FOR {total_epochs - warmup_epochs} EPOCHS")
        print("="*50)

        # 인코더를 학습에 포함시킵니다.
        self._set_encoder_grad(requires_grad=True)

        # 저장된 최고 성능 모델의 가중치를 불러옵니다.
        if warmup_epochs > 0 and os.path.exists(save_path):
            print(f"Loading best model from warm-up stage: {save_path}")
            checkpoint = torch.load(save_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            # 옵티마이저 상태는 불러오지 않습니다. 대신 새 그룹을 추가합니다.
            
        print("\n--- Adding Encoder group to the optimizer for Fine-tuning ---")
        encoder_lr = self.cfg.MainTask.lr / 10
        encoder_params = itertools.chain(self.nets.parameters(), self.projs.parameters())
        # 기존 옵티마이저에 인코더 파라미터 그룹을 추가합니다.
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
            va_acc_mot, va_acc_v, va_acc_a, va_acc_tot, va_mse_act = self.evaluate(va_loader)
            print(f"[FINE-TUNE] Epoch {epoch:02d}/{total_epochs} | Tr Loss {tr_loss:.4f} | Val Loss {va_loss:.4f}")
            print(f"  -> Acc(M/V/A/T): {va_acc_mot:.3f}/{va_acc_v:.3f}/{va_acc_a:.3f}/{va_acc_tot:.3f} | ACT_MSE: {va_mse_act:.4f}")

            # 검증 손실(va_loss)이 개선되었는지 확인
            if va_loss < best_val_loss:
                best_val_loss, patience_counter = va_loss, 0 # [수정] best_val_loss 업데이트
                best_performance = {'mot_acc': va_acc_mot, 'val_acc': va_acc_v, 'aro_acc': va_acc_a, 'best_val_loss': best_val_loss}
                torch.save({
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optim.state_dict(),
                }, save_path)
                print(f"  -> Best fine-tuned model saved to {save_path} (Val Loss: {best_val_loss:.4f})") # [수정] 출력 메시지 변경
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.MainTask.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        print(f"\n--- Best Validation Performance (Staged Training) ---")
        # 최종 성능 출력 메시지 변경
        print(f"Best Val Loss: {best_performance.get('best_val_loss', 0):.4f} | Acc(M/V/A): {best_performance.get('mot_acc', 0):.3f}/{best_performance.get('val_acc', 0):.3f}/{best_performance.get('aro_acc', 0):.3f}")

        return best_performance