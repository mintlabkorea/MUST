"""
Hard freeze + cross-task(cross-attention)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from sklearn.metrics import accuracy_score
from config.config import Config
from copy import deepcopy

from trainers.base_trainer import dataProcessor
from data.loader import make_multitask_loader # 멀티태스크 로더를 그대로 사용
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
        print("\n--- Building SIMPLIFIED model architecture based on config ---")
        hidden_dim = self.cfg.FusionModel.hidden_dim
        dropout_p = getattr(self.cfg.FusionModel, 'dropout', 0.1)
        num_heads = getattr(self.cfg.FusionModel, 'fusion_num_heads', 1)
        self.cross_attn_mode = getattr(self.cfg.FusionModel, "cross_attn_mode", "motion_only")

        # --- 1. Prepare list of all required modalities ---
        mot_raw = set(getattr(self.cfg.PretrainMotion, 'modalities_to_use', []))
        emo_raw = set(getattr(self.cfg.PretrainEmotion, 'modalities_to_use', []))
        def _norm(mods):
            out = []
            for m in mods:
                if m.startswith('veh'): out.append('veh')
                else: out.append(m)
            return set(out)
        mot_mods = _norm(mot_raw)
        emo_mods = _norm(emo_raw)
        all_unique_modalities = mot_mods.union(emo_mods)
        print(f"Required modalities: {list(all_unique_modalities)}")

        # --- 2. Dynamically create required encoders and projectors ---
        self.nets = nn.ModuleDict()
        self.projs = nn.ModuleDict()

        if 'imu' in all_unique_modalities:
            self.nets['imu'] = IMUFeatureEncoder(self.cfg.Encoders.imu)
            self.projs['imu'] = nn.Linear(self.cfg.Encoders.imu['encoder_dim'], hidden_dim)
        if 'veh' in all_unique_modalities:
            # Create a single 'veh' encoder for both tasks if needed
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
            
        # --- 3. Dynamically create task-specific fusion and prediction modules ---
        if mot_mods:
            self.motion_modalities = sorted(list(mot_mods))
            self.motion_feature_fusion = nn.Conv1d(hidden_dim * len(self.motion_modalities), hidden_dim, 1)
            self.motion_head = MotionHead(hidden_dim, self.cfg.PretrainMotion.num_motion)

            # Motion attends to Emotion
            self.cross_attn_m2e = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
            self.cross_attn_ln_m = nn.LayerNorm(hidden_dim)
            self.cross_attn_dropout_m = nn.Dropout(dropout_p)

        if emo_mods:
            self.emotion_modalities = sorted([m for m in emo_mods if m != 'survey'])
            predictor_input_dim = hidden_dim * len(self.emotion_modalities)
            
            self.emotion_feature_fusion = nn.Sequential(
                nn.Linear(predictor_input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            )
            
            self.emotion_valence_predictor = EmotionPredictor(predictor_input_dim, hidden_dim, self.cfg.PretrainEmotion.num_valence)
            self.emotion_arousal_predictor = EmotionPredictor(predictor_input_dim, hidden_dim, self.cfg.PretrainEmotion.num_arousal)

            # Emotion attends to Motion
            self.cross_attn_e2m = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
            self.cross_attn_ln_e = nn.LayerNorm(hidden_dim)
            self.cross_attn_dropout_e = nn.Dropout(dropout_p)

        # --- 4. Loss and utility setup ---
        init_log_vars = torch.tensor([0.0, 0.0, 0.0, 0.0]) # Initialize uncertainty vars
        self.log_vars = nn.Parameter(init_log_vars)
        self.scaler = torch.cuda.amp.GradScaler()
        self.to(self.cfg.Project.device)

    def _load_pretrained_weights(self):
        device = self.cfg.Project.device

        def _get_enc(ckpt):
            return ckpt.get('encoder', ckpt)

        def _safe_load(module, state_dict, prefix, tag):
            # state_dict가 None이거나 비어있으면 로드 시도 안 함
            if state_dict is None or not state_dict:
                print(f"[Load Skip] {tag}: State dictionary is empty or None.")
                return

            # prefix가 있으면 해당 부분만 잘라내고, 없으면 전체 사용
            if prefix:
                sub_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            else:
                sub_dict = state_dict

            if not sub_dict:
                print(f"[Load Skip] {tag}: No keys found with prefix '{prefix if prefix else 'N/A'}'")
                return
                
            msg = module.load_state_dict(sub_dict, strict=False)
            missing = getattr(msg, 'missing_keys', [])
            unexpected = getattr(msg, 'unexpected_keys', [])
            print(f"[Load OK] {tag}: {len(sub_dict)} keys loaded. (missing: {len(missing)}, unexpected: {len(unexpected)})")

        # ===========================
        # 1. 감정 모델 가중치 로드
        # ===========================
        emo_cfg = self.cfg.PretrainEmotion
        if getattr(emo_cfg, 'modalities_to_use', []):
            print("\n--- Loading Emotion Pre-trained Weights ---")
            emo_ckpt = torch.load(getattr(emo_cfg, 'ckpt_path', 'weights/best_emotion_tri.pt'), map_location=device)
            emo_enc_states = _get_enc(emo_ckpt)

            emo_map = { 'ppg': ('ppg', 'nets.ppg.', 'projs.ppg.'),
                        'sc' : ('sc' , 'nets.sc.' , 'projs.sc.' ),
                        'veh': ('veh', 'nets.veh.', 'projs.veh.'),  # ← attr 통일
                        'survey': ('survey', 'nets.survey.', 'projs.survey.') }

            for m in emo_cfg.modalities_to_use:
                if m in emo_map:
                    attr, enc_prefix, proj_prefix = emo_map[m]
                    if hasattr(self.nets, attr): _safe_load(getattr(self.nets, attr), emo_enc_states, enc_prefix, f"Emotion Encoder[{m}]")
                    if attr in self.projs: _safe_load(self.projs[attr], emo_enc_states, proj_prefix, f"Emotion Projection[{m}]")
            
            # 감정 예측기(Valence/Arousal) 가중치 로드
            if hasattr(self, 'emotion_valence_predictor'):
                _safe_load(self.emotion_valence_predictor, emo_ckpt.get('valence_predictor'), None, "Emotion Valence Predictor")
            if hasattr(self, 'emotion_arousal_predictor'):
                _safe_load(self.emotion_arousal_predictor, emo_ckpt.get('arousal_predictor'), None, "Emotion Arousal Predictor")

        # ===========================
        # 2. 행동 모델 가중치 로드
        # ===========================
        mot_cfg = self.cfg.PretrainMotion
        if getattr(mot_cfg, 'modalities_to_use', []):
            print("\n--- Loading Motion Pre-trained Weights ---")
            mot_ckpt = torch.load(getattr(mot_cfg, 'ckpt_path', 'weights/best_motion.pt'), map_location=device)
            mot_enc_states = _get_enc(mot_ckpt)

            mot_map = { 'imu': ('imu', 'imu.', 'p_imu.'),
                        'veh': ('veh', 'veh.', 'p_veh_m.'),          # ← attr 통일
                        'sc' : ('sc' , 'sc.' , 'p_sc.' ) }            
            for m in mot_cfg.modalities_to_use:
                if m in mot_map:
                    attr, enc_prefix, proj_prefix = mot_map[m]
                    if hasattr(self.nets, attr): _safe_load(getattr(self.nets, attr), mot_enc_states, enc_prefix, f"Motion Encoder[{m}]")
                    if attr in self.projs: _safe_load(self.projs[attr], mot_enc_states, proj_prefix, f"Motion Projection[{m}]")

            # 행동 헤드 가중치 로드
            if getattr(self.cfg, 'load_motion_head', True) and hasattr(self, 'motion_head'):
                _safe_load(self.motion_head, mot_ckpt.get('head'), None, 'Motion Head')
    
    def _freeze_encoders(self):
        print("\n--- Freezing all pre-trained encoders and projections ---")
        for param in self.nets.parameters():
            param.requires_grad = False
        for param in self.projs.parameters():
            param.requires_grad = False
            
    def _create_optimizer(self):
        params_to_train = []
        if hasattr(self, 'motion_feature_fusion'):
            params_to_train.append(self.motion_feature_fusion.parameters())
        if hasattr(self, 'emotion_feature_fusion'):
            params_to_train.append(self.emotion_feature_fusion.parameters())
        if hasattr(self, 'motion_head'):
            params_to_train.append(self.motion_head.parameters())
        if hasattr(self, 'emotion_valence_predictor'):
            params_to_train.append(self.emotion_valence_predictor.parameters())
        if hasattr(self, 'emotion_arousal_predictor'):
            params_to_train.append(self.emotion_arousal_predictor.parameters())
        
        # cross-attention
        params_to_train += [self.cross_attn_m2e.parameters(), self.cross_attn_ln_m.parameters()]
        params_to_train += [self.cross_attn_e2m.parameters(), self.cross_attn_ln_e.parameters()]

        params_to_train.append([self.log_vars])
        
        self.optim = torch.optim.Adam(itertools.chain(*params_to_train), lr=self.cfg.MainTask.lr)
        print("Optimizer created for fine-tuning fusion layers, heads, and predictors.")
        
    def _process_hrv(self, batch, device):
        # Helper to process HRV features
        hrv_keys = ['ppg_rmssd_emotion', 'ppg_sdnn_emotion']
        rr = batch['ppg_rr_emotion'].to(device)
        hrv_list = [rr.mean(1, keepdim=True), rr.std(1, keepdim=True), rr.min(1, keepdim=True).values, rr.max(1, keepdim=True).values]
        hrv_list.extend([batch[k].to(device).unsqueeze(1) for k in hrv_keys])
        return torch.cat(hrv_list, dim=1)

    def forward(self, batch):
        device = self.cfg.Project.device
        features = {'motion': {}, 'emotion': {}}
        
        # Default sequence length for motion tasks
        if 'label_motion' in batch and batch['label_motion'].dim() == 2:
            T_mot = batch['label_motion'].shape[1]
        elif 'imu_motion' in batch:
            T_mot = batch['imu_motion'].shape[1]
        else:
            T_mot = 1000  

        # --- Motion Path (Sequential Features) ---
        if hasattr(self, 'motion_modalities'):
            for mod in self.motion_modalities:
                if mod == 'imu' and 'imu_motion' in batch:
                    imu_len = (batch['imu_motion'].abs().sum(-1) > 0).sum(1)
                    features['motion']['imu'] = self.projs['imu'](self.nets['imu'](batch['imu_motion'].to(device), imu_len))
                
                elif mod == 'veh' and 'veh_motion' in batch:
                    out = self.nets['veh'](batch['veh_motion'].to(device).permute(0, 2, 1), return_pooled=True)
                    features['motion']['veh'] = self.projs['veh'](out).unsqueeze(1).expand(-1, T_mot, -1)
                
                elif mod == 'sc':
                    sc_pooled = self.nets['sc'](
                        batch['sc_motion_evt'].to(device), 
                        batch['sc_motion_type'].to(device), 
                        batch['sc_motion_phase'].to(device),
                        batch['sc_motion_time'].to(device)
                    )
                    sc_expanded = sc_pooled.unsqueeze(1).expand(-1, T_mot, -1)
                    features['motion']['sc'] = self.projs['sc'](sc_expanded)

                elif mod == 'survey' and 'survey_e' in batch:
                    survey_pooled = self.projs['survey'](self.nets['survey'](batch['survey_e'].to(device)))
                    survey_expanded = survey_pooled.unsqueeze(1).expand(-1, T_mot, -1)
                    features['motion']['survey'] = survey_expanded

                # Add PPG to Motion Path (New)
                elif mod == 'ppg' and 'ppg_emotion' in batch:
                    ppg_tcn = self.nets['ppg'](batch['ppg_emotion'].to(device).permute(0, 2, 1))
                    combined = torch.cat([ppg_tcn, self._process_hrv(batch, device)], dim=1)
                    ppg_pooled = self.projs['ppg'](combined)
                    ppg_expanded = ppg_pooled.unsqueeze(1).expand(-1, T_mot, -1)
                    features['motion']['ppg'] = ppg_expanded

        # --- Emotion Path (Pooled Features) ---
        if hasattr(self, 'emotion_modalities'):
            for mod in self.emotion_modalities:
                if mod == 'ppg' and 'ppg_emotion' in batch:
                    ppg_tcn = self.nets['ppg'](batch['ppg_emotion'].to(device).permute(0, 2, 1))
                    combined = torch.cat([ppg_tcn, self._process_hrv(batch, device)], dim=1)
                    features['emotion']['ppg'] = self.projs['ppg'](combined)
                
                elif mod == 'veh' and 'veh_emotion' in batch:
                    out = self.nets['veh'](batch['veh_emotion'].to(device).permute(0, 2, 1), return_pooled=True)
                    features['emotion']['veh'] = self.projs['veh'](out)
                
                elif mod == 'sc':
                    sc_pooled = self.nets['sc'](
                        batch['scenario_evt_e'].to(device), 
                        batch['scenario_type_e'].to(device), 
                        batch['phase_evt_e'].to(device),
                        batch['scenario_time_e'].to(device)
                    )
                    features['emotion']['sc'] = self.projs['sc'](sc_pooled)
                
                # Add IMU to Emotion Path (New)
                elif mod == 'imu' and 'imu_motion' in batch:
                    imu_len = (batch['imu_motion'].abs().sum(-1) > 0).sum(1)
                    imu_seq_out = self.nets['imu'](batch['imu_motion'].to(device), imu_len)
                    imu_pooled = imu_seq_out.mean(dim=1)
                    features['emotion']['imu'] = self.projs['imu'](imu_pooled)

                # Add Survey to Emotion Path (New)
                elif mod == 'survey' and 'survey_e' in batch:
                    features['emotion']['survey'] = self.projs['survey'](self.nets['survey'](batch['survey_e'].to(device)))

        # --- 2. Task-Specific Fusion (zero-pad missing modalities) ---
        H = self.cfg.FusionModel.hidden_dim
        # Batch size 안전 추출
        if len(features['motion']):
            B = next(iter(features['motion'].values())).shape[0]
        elif len(features['emotion']):
            B = next(iter(features['emotion'].values())).shape[0]
        else:
            # 텐서 항목을 하나 찾아서 batch size 추출
            B = None
            for v in batch.values():
                if torch.is_tensor(v) and v.dim() > 0:
                    B = v.shape[0]; break
            if B is None:
                raise ValueError("Empty batch: cannot infer batch size")

        # Motion (B,T,H) 고정 순서 concat
        m_list = [features['motion'][m] if m in features['motion'] else torch.zeros(B, T_mot, H, device=device)
                  for m in getattr(self, 'motion_modalities', [])]
        motion_cat = torch.cat(m_list, dim=2).permute(0, 2, 1) if m_list else torch.zeros(B, H, T_mot, device=device)
        fused_motion_seq = self.motion_feature_fusion(motion_cat)      # (B,H,T)

        # Emotion (B,H) 고정 순서 concat
        e_list = [features['emotion'][m] if m in features['emotion'] else torch.zeros(B, H, device=device)
                  for m in getattr(self, 'emotion_modalities', [])]
        emotion_cat = torch.cat(e_list, dim=1) if e_list else torch.zeros(B, H, device=device)
        fused_emotion_vector = self.emotion_feature_fusion(emotion_cat) # (B,H)
        
        # --- 3. Cross-Task Prediction (Bi-directional Cross-Attention) ---
        # Motion attends to Emotion
        motion_tokens = fused_motion_seq.permute(0, 2, 1)              # (B,T,H)
        emotion_token = fused_emotion_vector.unsqueeze(1)              # (B,1,H)
        attn_m, _ = self.cross_attn_m2e(motion_tokens, emotion_token.detach(), emotion_token.detach())
        attn_m = self.cross_attn_dropout_m(attn_m)
        motion_tokens = self.cross_attn_ln_m(motion_tokens + attn_m)   # (B,T,H)
        motion_input_for_head = motion_tokens.permute(0, 2, 1)         # (B,H,T)
        mot_logits = self.motion_head(motion_input_for_head)           # (B,T,C)

        # Emotion attends to Motion
        attn_e, _ = self.cross_attn_e2m(emotion_token, motion_tokens.detach(), motion_tokens.detach())
        attn_e = self.cross_attn_dropout_e(attn_e)
        emotion_token = self.cross_attn_ln_e(emotion_token + attn_e)   # (B,1,H)
        motion_context_for_emo = emotion_token.squeeze(1)              # (B,H)

        # Predictors: full concatenated vector 사용 + survey 재사용
        emotion_input_dict = {'fused': emotion_cat.unsqueeze(1)}
        if 'survey' in self.cfg.PretrainEmotion.modalities_to_use and 'survey' in features['emotion']:
            emotion_input_dict['static'] = features['emotion']['survey']

        valence_logits = self.emotion_valence_predictor(emotion_input_dict, context=motion_context_for_emo)
        arousal_logits = self.emotion_arousal_predictor(emotion_input_dict, context=motion_context_for_emo)

        return {
            'motion_logits': mot_logits, 'valence_logits': valence_logits, 'arousal_logits': arousal_logits,
            'fused_motion': fused_motion_seq.mean(dim=2),
            'fused_emotion': fused_emotion_vector
        }

    def run_epoch(self, loader, train: bool):
        self.train(train)
        total_loss, num_batches = 0.0, 0
        device = self.cfg.Project.device

        for batch in loader:
            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    out = self.forward(batch) 

                    # --- 1) Main Task Losses ---
                    mot_logits, mot_labels = out['motion_logits'], batch['label_motion'].to(device)
                    
                    valid_mask = (mot_labels.reshape(-1) > 0) & (mot_labels.reshape(-1) != 4)
                    loss_mot = F.cross_entropy(mot_logits.reshape(-1, mot_logits.shape[-1])[valid_mask], (mot_labels.reshape(-1)[valid_mask] - 1).long()) if valid_mask.any() else torch.tensor(0., device=device)

                    val_logits, raw_v = out['valence_logits'], batch['valence_reg_emotion'].to(device).view(-1)
                    tgt_v = torch.full_like(raw_v, -100, dtype=torch.long); tgt_v[raw_v < 4]=0; tgt_v[(raw_v>=4)&(raw_v<7)]=1; tgt_v[raw_v>=7]=2
                    loss_v = F.cross_entropy(val_logits, tgt_v, ignore_index=-100)
                    
                    aro_logits, raw_a = out['arousal_logits'], batch['arousal_reg_emotion'].to(device).view(-1)
                    tgt_a = torch.full_like(raw_a, -100, dtype=torch.long); tgt_a[raw_a < 4]=0; tgt_a[(raw_a>=4)&(raw_a<7)]=1; tgt_a[raw_a>=7]=2
                    loss_a = F.cross_entropy(aro_logits, tgt_a, ignore_index=-100)
                    
                    cos_sim = F.cosine_similarity(out['fused_motion'], out['fused_emotion'], dim=1)
                    loss_cross = (1 - cos_sim).mean()

                    # Total Loss with Uncertainty Weighting
                    loss = (torch.exp(-self.log_vars[0]) * loss_mot + 0.5 * self.log_vars[0]) + \
                        (torch.exp(-self.log_vars[1]) * loss_v + 0.5 * self.log_vars[1]) + \
                        (torch.exp(-self.log_vars[2]) * loss_a + 0.5 * self.log_vars[2]) + \
                        (torch.exp(-self.log_vars[3]) * loss_cross * self.cfg.MainTask.cross_modal_lambda + 0.5 * self.log_vars[3])

            if train:
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()

            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    @torch.no_grad()
    def evaluate(self, loader, return_preds=False):
        # This method is simplified for brevity but logic remains the same
        self.eval()
        all_preds = {'motion_preds': [], 'motion_trues': [], 'valence_preds': [], 'valence_trues': [], 'arousal_preds': [], 'arousal_trues': []}

        for batch in loader:
            out = self.forward(batch)
            
            p_mot = out['motion_logits'].argmax(-1).cpu(); t_mot_raw = batch['label_motion'].cpu()
            mask_mot = (t_mot_raw > 0) & (t_mot_raw != 4)
            all_preds['motion_preds'].append(p_mot[mask_mot]); all_preds['motion_trues'].append(t_mot_raw[mask_mot] - 1)
            
            p_v = out['valence_logits'].argmax(-1).cpu(); raw_v = batch['valence_reg_emotion'].view(-1).cpu()
            t_v = torch.full_like(raw_v, -1, dtype=torch.long); t_v[raw_v<4]=0; t_v[(raw_v>=4)&(raw_v<7)]=1; t_v[raw_v>=7]=2
            m_v = t_v != -1
            all_preds['valence_preds'].append(p_v[m_v]); all_preds['valence_trues'].append(t_v[m_v])
            
            p_a = out['arousal_logits'].argmax(-1).cpu(); raw_a = batch['arousal_reg_emotion'].view(-1).cpu()
            t_a = torch.full_like(raw_a, -1, dtype=torch.long); t_a[raw_a<4]=0; t_a[(raw_a>=4)&(raw_a<7)]=1; t_a[raw_a>=7]=2
            m_a = t_a != -1
            all_preds['arousal_preds'].append(p_a[m_a]); all_preds['arousal_trues'].append(t_a[m_a])

        for key in all_preds: all_preds[key] = torch.cat(all_preds[key])
        
        if return_preds:
            return all_preds
        
        acc_mot = accuracy_score(all_preds['motion_trues'], all_preds['motion_preds'])
        acc_v = accuracy_score(all_preds['valence_trues'], all_preds['valence_preds'])
        acc_a = accuracy_score(all_preds['arousal_trues'], all_preds['arousal_preds'])
        return acc_mot, acc_v, acc_a
            
            
    def fusion_train(self):
        tr_loader = make_multitask_loader(self.cfg, self.train_keys, shuffle=True, dp=self)
        va_loader = make_multitask_loader(self.cfg, self.val_keys, shuffle=False, dp=self)
        
        best_loss, patience_counter, best_performance = float('inf'), 0, {}

        for epoch in range(1, self.cfg.MainTask.epochs + 1):
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
                }, "weights/best_fusion_v16.pt") # Updated save path
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.MainTask.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        print(f"\n--- Best Validation Performance ---")
        print(f"Loss: {best_performance.get('best_loss', 'N/A'):.4f} | Acc(M/V/A): {best_performance.get('mot_acc', 0):.3f}/{best_performance.get('val_acc', 0):.3f}/{best_performance.get('aro_acc', 0):.3f}")
        return best_performance