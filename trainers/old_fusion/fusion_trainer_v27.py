"""
v22: The Ultimate Single Model
v19(Bi-directional FiLM) 아키텍처를 기반, 
v20(Asymmetric LR/WD)의 옵티마이저 전략과 
v21(Asymmetric Alignment Loss)의 손실 함수를 모두 결합한 최종 버전.
"""
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

        # Model build -> weight load -> encoder freeze -> optimizer creation
        self._build_model()
        self._load_pretrained_weights()
        self._freeze_encoders()
        self._create_optimizer()

    def _build_model(self):
        print("\n--- Building V27 model architecture: Bi-directional FiLM + Causal Context ---")
        hidden_dim = self.cfg.FusionModel.hidden_dim
        dropout_p = getattr(self.cfg.FusionModel, 'dropout', 0.2) # 안정화된 dropout
        num_heads = getattr(self.cfg.FusionModel, 'num_heads', 4)

        # 비대칭 정렬 손실을 위한 헤드 
        self.alignment_head = nn.Linear(hidden_dim, hidden_dim)

        self.emo_delay_steps = getattr(self.cfg.FusionModel, "emo_delay_steps", 50)
        self.emo_ctx_window  = getattr(self.cfg.FusionModel, "emo_ctx_window", 200)
        self.emo_use_mot_logits = getattr(self.cfg.FusionModel, "emo_use_mot_logits", True)
        C_mot = self.cfg.PretrainMotion.num_motion if self.emo_use_mot_logits else 0

        # --- 1. 모달리티 및 인코더/프로젝터 준비 (v18과 동일) ---
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

        # --- 2. 태스크별 fusion/헤드 (v18과 동일) ---
        if mot_mods:
            self.motion_modalities = sorted(list(mot_mods))
            self.motion_feature_fusion = nn.Conv1d(hidden_dim * len(self.motion_modalities), hidden_dim, 1)
            self.motion_head = MotionHead(hidden_dim, self.cfg.PretrainMotion.num_motion)

        if emo_mods:
            self.emotion_modalities = sorted([m for m in emo_mods if m != 'survey'])
            emo_fusion_in = hidden_dim * len(self.emotion_modalities)
            predictor_in  = emo_fusion_in + C_mot
            self.emotion_feature_fusion = nn.Sequential(nn.Linear(emo_fusion_in, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
            self.emotion_valence_predictor = EmotionPredictor(predictor_in, hidden_dim, self.cfg.PretrainEmotion.num_valence)
            self.emotion_arousal_predictor = EmotionPredictor(predictor_in, hidden_dim, self.cfg.PretrainEmotion.num_arousal)
 
        # === 3. Bi-directional cross-attention (v18과 동일) ===
        self.cross_attn_m2e = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_ln_m     = nn.LayerNorm(hidden_dim)
        self.cross_drop_m   = nn.Dropout(dropout_p)

        self.cross_attn_e2m = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_ln_e     = nn.LayerNorm(hidden_dim)
        self.cross_drop_e   = nn.Dropout(dropout_p)

        # === 4. Bi-directional FiLM 게이팅 모듈 (제안 1 적용) ===
        # (Motion → Emotion) 기존 모듈
        self.emo_film = nn.Sequential(nn.Linear(hidden_dim, 2*hidden_dim), nn.ReLU(), nn.Linear(2*hidden_dim, 2*hidden_dim))
        
        # (Emotion → Motion) 신규 모듈 ★
        self.mot_film = nn.Sequential(nn.Linear(hidden_dim, 2*hidden_dim), nn.ReLU(), nn.Linear(2*hidden_dim, 2*hidden_dim))

        # --- 5. 로스/유틸 ---
        init_log_vars = torch.tensor([0.0, 0.0, 0.0, 0.0])
        self.log_vars = nn.Parameter(init_log_vars)
        self.scaler = torch.cuda.amp.GradScaler()
        self.to(self.cfg.Project.device)

    def _load_pretrained_weights(self):
        # v18과 동일 (내용 생략)
        device = self.cfg.Project.device
        def _get_enc(ckpt): return ckpt.get('encoder', ckpt)
        def _safe_load(module, state_dict, prefix, tag):
            if state_dict is None or not state_dict: return
            sub_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)} if prefix else state_dict
            if not sub_dict: return
            tgt_sd = module.state_dict()
            filtered = {k: v for k, v in sub_dict.items() if k in tgt_sd and tgt_sd[k].shape == v.shape}
            module.load_state_dict(filtered, strict=False)
            print(f"[Load OK/Partial] {tag}: {len(filtered)}/{len(sub_dict)} keys loaded.")

        # 감정/행동 모델 가중치 로드... (v18과 동일)
        emo_cfg = self.cfg.PretrainEmotion
        if getattr(emo_cfg, 'modalities_to_use', []):
            emo_ckpt = torch.load(getattr(emo_cfg, 'ckpt_path', 'weights/best_emotion_imu_ppg_sc_survey_veh.pt'), map_location=device)
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
            mot_ckpt = torch.load(getattr(mot_cfg, 'ckpt_path', 'weights/best_motion.pt'), map_location=device)
            mot_enc_states = _get_enc(mot_ckpt)
            mot_map = { 'imu': ('imu', 'imu.', 'p_imu.'), 'veh_m': ('veh_m', 'veh.', 'p_veh_m.'), 'sc': ('sc', 'sc.', 'p_sc.') }
            for m in mot_cfg.modalities_to_use:
                if m in mot_map:
                    attr, enc_prefix, proj_prefix = mot_map[m]
                    if hasattr(self.nets, attr): _safe_load(getattr(self.nets, attr), mot_enc_states, enc_prefix, f"Motion Encoder[{m}]")
                    if attr in self.projs: _safe_load(self.projs[attr], mot_enc_states, proj_prefix, f"Motion Projection[{m}]")
            if getattr(self.cfg, 'load_motion_head', True) and hasattr(self, 'motion_head'):
                _safe_load(self.motion_head, mot_ckpt.get('head'), None, 'Motion Head')

    def _freeze_encoders(self):
        print("\n--- Freezing all pre-trained encoders and projections ---")
        for param in self.nets.parameters(): param.requires_grad = False
        for param in self.projs.parameters(): param.requires_grad = False
            
    def _create_optimizer(self):
        # 비대칭적 학습률/정규화 전략을 적용
        base_lr = self.cfg.MainTask.lr
        base_wd = self.cfg.MainTask.weight_decay
        
        cautious_lr = base_lr * self.cfg.MainTask.lr_asym_ratio
        cautious_wd = base_wd * self.cfg.MainTask.wd_asym_ratio

        # Standard Parameters (안정적인 경로 및 예측 헤드)
        params_standard = []
        if hasattr(self, 'motion_feature_fusion'): params_standard.append(self.motion_feature_fusion.parameters())
        if hasattr(self, 'emotion_feature_fusion'): params_standard.append(self.emotion_feature_fusion.parameters())
        if hasattr(self, 'motion_head'): params_standard.append(self.motion_head.parameters())
        if hasattr(self, 'emotion_valence_predictor'): params_standard.append(self.emotion_valence_predictor.parameters())
        if hasattr(self, 'emotion_arousal_predictor'): params_standard.append(self.emotion_arousal_predictor.parameters())
        
        # Motion -> Emotion 경로
        params_standard.append(self.cross_attn_e2m.parameters())
        params_standard.append(self.cross_ln_e.parameters())
        params_standard.append(self.emo_film.parameters())
        
        # alignment_head도 Standard 그룹에 추가
        params_standard.append(self.alignment_head.parameters())

        # Cautious Parameters (복잡성을 더하는 Emotion -> Motion 경로)
        params_cautious = []
        params_cautious.append(self.cross_attn_m2e.parameters())
        params_cautious.append(self.cross_ln_m.parameters())
        params_cautious.append(self.mot_film.parameters())

        # 최종 옵티마이저 그룹 구성
        optim_groups = [
            {'params': itertools.chain(*params_standard), 'lr': base_lr, 'weight_decay': base_wd, 'name': 'Standard Group'},
            {'params': itertools.chain(*params_cautious), 'lr': cautious_lr, 'weight_decay': cautious_wd, 'name': 'Cautious Group'},
        ]
        
        # 불확실성 손실 사용 여부에 따라 log_vars를 옵티마이저에 추가
        if self.cfg.MainTask.use_uncertainty_loss:
            optim_groups.append(
                {'params': self.log_vars, 'lr': base_lr, 'weight_decay': 0.0, 'name': 'LogVars (No WD)'}
            )
        
        self.optim = torch.optim.Adam(optim_groups)

        print("\n--- Optimizer created with Asymmetric Hyperparameters ---")
        for group in self.optim.param_groups:
            group_params = sum(p.numel() for p in group['params'])
            print(f" - Group '{group.get('name', 'Unnamed')}': {group_params} params, lr={group['lr']:.1e}, wd={group['weight_decay']:.1e}")
        
    def _process_hrv(self, batch, device):
        """
        [수정됨] HRV 관련 특징들을 하나의 텐서로 결합하는 헬퍼 함수.
        NaN 값을 0.0으로 안전하게 변환하는 로직이 추가되었습니다.
        """
        ppg_rr = batch['ppg_rr_emotion'].to(device)
        
        hrv_list = []
        if ppg_rr.dim() > 1 and ppg_rr.shape[1] > 0:
            # mean, std 계산 시 발생할 수 있는 NaN도 방지
            hrv_list.append(torch.nan_to_num(ppg_rr.mean(dim=1, keepdim=True), nan=0.0))
            std_dev = torch.std(ppg_rr, dim=1, keepdim=True)
            hrv_list.append(torch.nan_to_num(std_dev, nan=0.0))
            hrv_list.append(torch.min(ppg_rr, dim=1, keepdim=True).values)
            hrv_list.append(torch.max(ppg_rr, dim=1, keepdim=True).values)
        else: # 데이터가 없는 경우 0으로 채움
            hrv_list.extend([torch.zeros(ppg_rr.shape[0], 1, device=device)] * 4)

        # Dataloader에서 생성된 ppg_rmssd와 ppg_sdnn에 포함될 수 있는 NaN을
        # 0.0으로 안전하게 변환해줍니다.
        for key in ['ppg_rmssd_emotion', 'ppg_sdnn_emotion']:
            value_tensor = batch[key].to(device).unsqueeze(1)
            hrv_list.append(torch.nan_to_num(value_tensor, nan=0.0))
        
        return torch.cat(hrv_list, dim=1)


    def forward(self, batch):
        device = self.cfg.Project.device
        features = {'motion': {}, 'emotion': {}}
        
        T_mot = batch['label_motion'].shape[1] if 'label_motion' in batch and batch['label_motion'].dim() == 2 else 1000
   
        # --- 1. Encoder & Projection (v18과 동일) ---
        # Motion Path (Sequential Features)
        if hasattr(self, 'motion_modalities'):
            # ... (코드 생략, v18과 동일)
            for mod in self.motion_modalities:
                if mod == 'imu' and 'imu_motion' in batch: features['motion']['imu'] = self.projs['imu'](self.nets['imu'](batch['imu_motion'].to(device), (batch['imu_motion'].abs().sum(-1) > 0).sum(1)))
                elif mod == 'veh' and 'veh_motion' in batch: features['motion']['veh'] = self.projs['veh'](self.nets['veh'](batch['veh_motion'].to(device).permute(0, 2, 1), return_pooled=True)).unsqueeze(1).expand(-1, T_mot, -1)
                elif mod == 'sc': sc_pooled = self.nets['sc'](batch['sc_motion_evt'].to(device), batch['sc_motion_type'].to(device), batch['sc_motion_phase'].to(device), batch['sc_motion_time'].to(device)); features['motion']['sc'] = self.projs['sc'](sc_pooled.unsqueeze(1).expand(-1, T_mot, -1))
                elif mod == 'survey' and 'survey_e' in batch: survey_pooled = self.projs['survey'](self.nets['survey'](batch['survey_e'].to(device))); features['motion']['survey'] = survey_pooled.unsqueeze(1).expand(-1, T_mot, -1)
                elif mod == 'ppg' and 'ppg_emotion' in batch: ppg_tcn = self.nets['ppg'](batch['ppg_emotion'].to(device).permute(0, 2, 1)); combined = torch.cat([ppg_tcn, self._process_hrv(batch, device)], dim=1); ppg_pooled = self.projs['ppg'](combined); features['motion']['ppg'] = ppg_pooled.unsqueeze(1).expand(-1, T_mot, -1)

        # Emotion Path (Pooled Features)
        if hasattr(self, 'emotion_modalities'):
            # ... (코드 생략, v18과 동일)
            for mod in self.emotion_modalities:
                if mod == 'ppg' and 'ppg_emotion' in batch: ppg_tcn = self.nets['ppg'](batch['ppg_emotion'].to(device).permute(0, 2, 1)); combined = torch.cat([ppg_tcn, self._process_hrv(batch, device)], dim=1); features['emotion']['ppg'] = self.projs['ppg'](combined)
                elif mod == 'veh' and 'veh_emotion' in batch: features['emotion']['veh'] = self.projs['veh'](self.nets['veh'](batch['veh_emotion'].to(device).permute(0, 2, 1), return_pooled=True))
                elif mod == 'sc': features['emotion']['sc'] = self.projs['sc'](self.nets['sc'](batch['scenario_evt_e'].to(device), batch['scenario_type_e'].to(device), batch['phase_evt_e'].to(device), batch['scenario_time_e'].to(device)))
                elif mod == 'imu' and 'imu_motion' in batch: imu_seq_out = self.nets['imu'](batch['imu_motion'].to(device), (batch['imu_motion'].abs().sum(-1) > 0).sum(1)); imu_pooled = imu_seq_out.mean(dim=1); features['emotion']['imu'] = self.projs['imu'](imu_pooled)
                elif mod == 'survey' and 'survey_e' in batch: features['emotion']['survey'] = self.projs['survey'](self.nets['survey'](batch['survey_e'].to(device)))

        # --- 2. 초기 Fusion (v18과 동일) ---
        motion_cat = torch.cat([features['motion'][m] for m in self.motion_modalities], dim=2).permute(0, 2, 1)
        fused_motion_seq = self.motion_feature_fusion(motion_cat)
        
        emotion_cat = torch.cat([features['emotion'][m] for m in self.emotion_modalities], dim=1)
        fused_emotion_vector = self.emotion_feature_fusion(emotion_cat)

        # --- 3. Bi-directional Cross-Interaction & FiLM ---
        motion_tokens = fused_motion_seq.permute(0, 2, 1)   # (B,T,H)
        emotion_token = fused_emotion_vector.unsqueeze(1)   # (B,1,H)

        # 3-A. Emotion Path: Emotion attends to Causal Motion Context (제안 2 적용) ★
        # 인과성을 위해 과거의 모션 정보만 참조. .detach()로 그래디언트 흐름을 차단하여 모션 성능 저하 방지.
        B, T, H = motion_tokens.shape
        delta  = min(self.emo_delay_steps, T-1)
        t_star = max(0, T - 1 - delta)
        left   = max(0, t_star - self.emo_ctx_window + 1)
        right  = t_star + 1
        causal_motion_ctx = motion_tokens[:, left:right, :].detach()

        attn_e, _ = self.cross_attn_e2m(emotion_token, causal_motion_ctx, causal_motion_ctx)
        refined_emotion_token = self.cross_ln_e(emotion_token + self.cross_drop_e(attn_e))
        motion_context_for_emo = refined_emotion_token.squeeze(1) # (B,H)

        # 3-B. Motion Path: Motion attends to Emotion Context
        attn_m, _ = self.cross_attn_m2e(motion_tokens, emotion_token.detach(), emotion_token.detach())
        refined_motion_tokens = self.cross_ln_m(motion_tokens + self.cross_drop_m(attn_m)) # (B,T,H)

        # 3-C. Bi-directional FiLM Modulation (제안 1 적용) ★
        # Motion → Emotion 변조 (기존)
        gamma_beta_emo = self.emo_film(motion_context_for_emo)
        gamma_emo, beta_emo = gamma_beta_emo.chunk(2, dim=-1)
        final_emotion_vector = fused_emotion_vector * (1 + torch.tanh(gamma_emo)) + beta_emo

        # Emotion → Motion 변조 (신규)
        gamma_beta_mot = self.mot_film(fused_emotion_vector.detach())
        gamma_mot, beta_mot = gamma_beta_mot.unsqueeze(1).chunk(2, dim=-1)
        final_motion_tokens = refined_motion_tokens * (1 + torch.tanh(gamma_mot)) + beta_mot

        # --- 4. Prediction Heads ---
        # Motion Head는 최종 변조된 모션 토큰을 사용
        motion_input_for_head = final_motion_tokens.permute(0, 2, 1)
        mot_logits = self.motion_head(motion_input_for_head)

        # Emotion Head는 최종 변조된 감정 벡터와 추가 정보를 사용
        if self.emo_use_mot_logits:
            p_mot = F.softmax(mot_logits, dim=-1).mean(dim=1).detach() # 감정 예측에 사용 시 그래디언트 차단
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
            'fused_motion': final_motion_tokens.mean(dim=1),
            'fused_emotion': final_emotion_vector
        }
    
    def run_epoch(self, loader, train: bool):
        self.train(train)
        total_loss, num_batches = 0.0, 0
        device = self.cfg.Project.device

        for batch in loader:
            with torch.set_grad_enabled(train):
                out = self.forward(batch) 

                # Main Task Losses
                mot_logits, mot_labels = out['motion_logits'], batch['label_motion'].to(device)
                valid_mask = (mot_labels.reshape(-1) > 0) & (mot_labels.reshape(-1) != 4)
                loss_mot = F.cross_entropy(mot_logits.reshape(-1, mot_logits.shape[-1])[valid_mask], (mot_labels.reshape(-1)[valid_mask] - 1).long()) if valid_mask.any() else torch.tensor(0., device=device)

                val_logits, raw_v = out['valence_logits'], batch['valence_reg_emotion'].to(device).view(-1)
                tgt_v = torch.full_like(raw_v, -100, dtype=torch.long); tgt_v[raw_v < 4]=0; tgt_v[(raw_v>=4)&(raw_v<7)]=1; tgt_v[raw_v>=7]=2
                loss_v = F.cross_entropy(val_logits, tgt_v, ignore_index=-100)
                
                aro_logits, raw_a = out['arousal_logits'], batch['arousal_reg_emotion'].to(device).view(-1)
                tgt_a = torch.full_like(raw_a, -100, dtype=torch.long); tgt_a[raw_a < 4]=0; tgt_a[(raw_a>=4)&(raw_a<7)]=1; tgt_a[raw_a>=7]=2
                loss_a = F.cross_entropy(aro_logits, tgt_a, ignore_index=-100)
                
                # Asymmetric Alignment Loss (from v21)
                fused_motion_detached = out['fused_motion'].detach()
                predicted_emotion = self.alignment_head(fused_motion_detached)
                loss_align = F.mse_loss(predicted_emotion, out['fused_emotion'])

                loss_weight_mot = 1.0
                loss_weight_val = 2.0 
                loss_weight_aro = 3.0  # Make arousal the highest priority
                loss_weight_align = 0.5

                # Total Loss
                if self.cfg.MainTask.use_uncertainty_loss:
                    loss = (loss_weight_mot * loss_mot) + \
                            (loss_weight_val * loss_v) + \
                            (loss_weight_aro * loss_a) + \
                            (loss_weight_align * loss_align * self.cfg.MainTask.cross_modal_lambda)
                else:
                    # 불확실성 가중치를 사용하지 않는 일반적인 가중합 손실
                    loss = loss_mot + loss_v + loss_a + (loss_align * self.cfg.MainTask.cross_modal_lambda)
            
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
        # v18과 동일 (내용 생략)
        self.eval()
        all_preds = {'motion_preds': [], 'motion_trues': [], 'valence_preds': [], 'valence_trues': [], 'arousal_preds': [], 'arousal_trues': []}
        for batch in loader:
            out = self.forward(batch)
            p_mot = out['motion_logits'].argmax(-1).cpu(); t_mot_raw = batch['label_motion'].cpu()
            mask_mot = (t_mot_raw > 0) & (t_mot_raw != 4)
            all_preds['motion_preds'].append(p_mot[mask_mot]); all_preds['motion_trues'].append(t_mot_raw[mask_mot] - 1)
            
            p_v = out['valence_logits'].argmax(-1).cpu(); raw_v = batch['valence_reg_emotion'].view(-1).cpu()
            t_v = torch.full_like(raw_v, -1, dtype=torch.long); t_v[raw_v<4]=0; t_v[(raw_v>=4)&(raw_v<7)]=1; t_v[raw_v>=7]=2
            all_preds['valence_preds'].append(p_v[t_v != -1]); all_preds['valence_trues'].append(t_v[t_v != -1])
            
            p_a = out['arousal_logits'].argmax(-1).cpu(); raw_a = batch['arousal_reg_emotion'].view(-1).cpu()
            t_a = torch.full_like(raw_a, -1, dtype=torch.long); t_a[raw_a<4]=0; t_a[(raw_a>=4)&(raw_a<7)]=1; t_a[raw_a>=7]=2
            all_preds['arousal_preds'].append(p_a[t_a != -1]); all_preds['arousal_trues'].append(t_a[t_a != -1])

        for key in all_preds: all_preds[key] = torch.cat(all_preds[key])
        if return_preds: return all_preds
        
        acc_mot = accuracy_score(all_preds['motion_trues'], all_preds['motion_preds'])
        acc_v = accuracy_score(all_preds['valence_trues'], all_preds['valence_preds'])
        acc_a = accuracy_score(all_preds['arousal_trues'], all_preds['arousal_preds'])
        return acc_mot, acc_v, acc_a
            
            
    def fusion_train(self, save_path="weights/best_fusion_v27.pt"): 
        tr_loader = make_multitask_loader(self.cfg, self.train_keys, shuffle=True, dp=self)
        va_loader = make_multitask_loader(self.cfg, self.val_keys, shuffle=False, dp=self)
        
        best_loss, patience_counter, best_performance = float('inf'), 0, {}

        for epoch in range(1, self.cfg.MainTask.epochs + 1):
            self.current_epoch = epoch
            tr_loss = self.run_epoch(tr_loader, train=True)
            va_loss = self.run_epoch(va_loader, train=False)
            
            va_acc_mot, va_acc_v, va_acc_a = self.evaluate(va_loader)
            print(f"Epoch {epoch:02d} | Tr Loss {tr_loss:.4f} | Val Loss {va_loss:.4f} | Acc(M/V/A): {va_acc_mot:.3f}/{va_acc_v:.3f}/{va_acc_a:.3f}")

            if va_loss < best_loss:
                best_loss, patience_counter = va_loss, 0
                best_performance = {'mot_acc': va_acc_mot, 'val_acc': va_acc_v, 'aro_acc': va_acc_a, 'best_loss': best_loss}
                
                torch.save({
                    'model_state_dict': self.state_dict(), 
                    'optimizer_state_dict': self.optim.state_dict(),
                }, save_path)
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.MainTask.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        print(f"\n--- Best Validation Performance (v27) ---")
        print(f"Loss: {best_performance.get('best_loss', 'N/A'):.4f} | Acc(M/V/A): {best_performance.get('mot_acc', 0):.3f}/{best_performance.get('val_acc', 0):.3f}/{best_performance.get('aro_acc', 0):.3f}")
        return best_performance