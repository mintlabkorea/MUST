"""
FiLM-Only Fusion Model
- Interaction: Bi-directional FiLM을 유일한 상호작용 메커니즘으로 사용. (v19 참고)
- Training: 안정적인 2단계 훈련(Warm-up + Fine-tuning) 방식을 채택. (v24 참고)
- Simplicity: Cross-Attention, Alignment Loss 등 다른 복잡한 모듈은 제거.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from sklearn.metrics import accuracy_score
from config.config import Config

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
        self._build_model()
        self._load_pretrained_weights()

    def _build_model(self):
        print("\n--- Building FiLM-Only Fusion Model (V31) ---")
        hidden_dim = self.cfg.FusionModel.hidden_dim
        
        # --- 1. 인코더 및 프로젝터 (기존 버전들과 동일) ---
        mot_mods = set(getattr(self.cfg.PretrainMotion, 'modalities_to_use', []))
        emo_mods = set(getattr(self.cfg.PretrainEmotion, 'modalities_to_use', []))
        all_unique_modalities = mot_mods.union(emo_mods)

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

        # --- 2. 태스크별 Fusion 및 예측 헤드 ---
        if mot_mods:
            self.motion_modalities = sorted(list(mot_mods))
            self.motion_feature_fusion = nn.Conv1d(hidden_dim * len(self.motion_modalities), hidden_dim, 1)
            self.motion_head = MotionHead(hidden_dim, self.cfg.PretrainMotion.num_motion)

        if emo_mods:
            self.emotion_modalities = sorted([m for m in emo_mods if m != 'survey'])
            emo_fusion_in = hidden_dim * len(self.emotion_modalities)
            self.emotion_feature_fusion = nn.Sequential(nn.Linear(emo_fusion_in, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
            # [단순화] predictor_in은 다른 정보 없이 오직 fused_emotion_vector의 차원만 사용
            self.emotion_valence_predictor = EmotionPredictor(hidden_dim, hidden_dim, self.cfg.PretrainEmotion.num_valence)
            self.emotion_arousal_predictor = EmotionPredictor(hidden_dim, hidden_dim, self.cfg.PretrainEmotion.num_arousal)

        # --- 3. [핵심] Bi-directional FiLM 모듈 ---
        # (Motion Context -> Emotion Feature) 변조기
        self.emo_film = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(), nn.Linear(hidden_dim * 2, hidden_dim * 2))
        
        # (Emotion Context -> Motion Feature) 변조기
        self.mot_film = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(), nn.Linear(hidden_dim * 2, hidden_dim * 2))

        # --- 4. 로스 및 유틸 ---
        self.log_vars = nn.Parameter(torch.tensor([0.0, 0.0, 0.0])) # 3개 태스크(mot, val, aro)만 사용
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
            print(f"[Load OK/Partial] {tag}: {len(filtered)}/{len(sub_dict)} keys loaded.")

        emo_cfg = self.cfg.PretrainEmotion
        if getattr(emo_cfg, 'modalities_to_use', []):
            emo_ckpt = torch.load(getattr(emo_cfg, 'ckpt_path', 'weights/best_emotion_tri.pt'), map_location=device)
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

    def _set_encoder_grad(self, requires_grad: bool):
        # v24의 _set_encoder_grad 함수를 여기에 복사하여 사용하시면 됩니다.
        status = "Unfrozen" if requires_grad else "Frozen"
        print(f"\n--- Setting all encoders to {status} ---")
        for param in itertools.chain(self.nets.parameters(), self.projs.parameters()):
            param.requires_grad = requires_grad

    def _create_optimizer_stage1(self):
        print("\n--- Creating optimizer for Stage 1: Warm-up (Fusion, FiLM, Heads) ---")
        base_lr, base_wd = self.cfg.MainTask.lr, self.cfg.MainTask.weight_decay
        
        trainable_params = [
            self.motion_feature_fusion.parameters(), self.emotion_feature_fusion.parameters(),
            self.motion_head.parameters(), self.emotion_valence_predictor.parameters(),
            self.emotion_arousal_predictor.parameters(),
            self.emo_film.parameters(), self.mot_film.parameters()
        ]
        
        optim_groups = [{'params': itertools.chain(*trainable_params), 'lr': base_lr, 'weight_decay': base_wd, 'name': 'Fusion & Heads'}]
        optim_groups.append({'params': self.log_vars, 'lr': base_lr, 'weight_decay': 0.0, 'name': 'LogVars'})
        self.optim = torch.optim.Adam(optim_groups)

    def _create_optimizer_stage2(self):
        print("\n--- Creating optimizer for Stage 2: Fine-tuning ---")
        base_lr, base_wd = self.cfg.MainTask.lr, self.cfg.MainTask.weight_decay
        encoder_lr = base_lr / 10

        encoder_params = itertools.chain(self.nets.parameters(), self.projs.parameters())
        fusion_params = [
            self.motion_feature_fusion.parameters(), self.emotion_feature_fusion.parameters(),
            self.motion_head.parameters(), self.emotion_valence_predictor.parameters(),
            self.emotion_arousal_predictor.parameters(),
            self.emo_film.parameters(), self.mot_film.parameters()
        ]
            
        optim_groups = [
            {'params': encoder_params, 'lr': encoder_lr, 'weight_decay': base_wd, 'name': 'Encoders (Low LR)'},
            {'params': itertools.chain(*fusion_params), 'lr': base_lr, 'weight_decay': base_wd, 'name': 'Fusion & Heads (Base LR)'}
        ]
        optim_groups.append({'params': self.log_vars, 'lr': base_lr, 'weight_decay': 0.0, 'name': 'LogVars'})
        self.optim = torch.optim.Adam(optim_groups)

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

        for key in ['ppg_rmssd_emotion', 'ppg_sdnn_emotion']:
            value_tensor = batch[key].to(device).unsqueeze(1)
            hrv_list.append(torch.nan_to_num(value_tensor, nan=0.0))
        
        return torch.cat(hrv_list, dim=1)


    def forward(self, batch):
        # --- 1. 특징 추출 (기존과 동일) ---
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

        # --- 2. 초기 모달리티 Fusion ---
        motion_cat = torch.cat([features['motion'][m] for m in self.motion_modalities], dim=2).permute(0, 2, 1)
        fused_motion_seq = self.motion_feature_fusion(motion_cat) # (B, H, T)
        
        emotion_cat = torch.cat([features['emotion'][m] for m in self.emotion_modalities], dim=1)
        fused_emotion_vector = self.emotion_feature_fusion(emotion_cat) # (B, H)

        # --- 3. [핵심] Bi-directional FiLM 상호작용 ---
        
        # 3-A. Motion -> Emotion 경로
        # Motion 시퀀스를 pooling하여 감정에 영향을 줄 context vector 생성
        motion_context = fused_motion_seq.permute(0, 2, 1).mean(dim=1).detach() # (B, H)
        
        # FiLM 파라미터(gamma, beta) 생성
        gamma_beta_emo = self.emo_film(motion_context)
        gamma_emo, beta_emo = gamma_beta_emo.chunk(2, dim=-1)
        
        # FiLM 적용하여 최종 감정 특징 생성
        final_emotion_vector = fused_emotion_vector * (1 + torch.tanh(gamma_emo)) + beta_emo

        # 3-B. Emotion -> Motion 경로
        # Emotion 벡터 자체가 행동에 영향을 줄 context vector
        emotion_context = fused_emotion_vector.detach()
        
        # FiLM 파라미터(gamma, beta) 생성
        gamma_beta_mot = self.mot_film(emotion_context)
        gamma_mot, beta_mot = gamma_beta_mot.unsqueeze(1).chunk(2, dim=-1) # (B, 1, H) 형태로 변환

        # FiLM 적용하여 최종 행동 특징 시퀀스 생성
        final_motion_seq = fused_motion_seq * (1 + torch.tanh(gamma_mot.permute(0, 2, 1))) + beta_mot.permute(0, 2, 1)

        # --- 4. 최종 예측 ---
        mot_logits = self.motion_head(final_motion_seq)

        static_feat = self.projs['survey'](self.nets['survey'](batch['survey_e'].to(device))) if 'survey' in self.emotion_modalities else None
        
        # [단순화] Predictor는 최종 변조된 벡터만 입력으로 받음
        valence_logits = self.emotion_valence_predictor({'fused': final_emotion_vector.unsqueeze(1)}, context=static_feat)
        arousal_logits = self.emotion_arousal_predictor({'fused': final_emotion_vector.unsqueeze(1)}, context=static_feat)

        return {
            'motion_logits': mot_logits,
            'valence_logits': valence_logits,
            'arousal_logits': arousal_logits,
            'fused_motion': final_motion_seq.mean(dim=2),    
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
                
                if getattr(self.cfg.MainTask, 'use_uncertainty_loss', True): # Use uncertainty loss by default
                    loss = (torch.exp(-self.log_vars[0]) * loss_mot + 0.5 * self.log_vars[0]) + \
                        (torch.exp(-self.log_vars[1]) * loss_v + 0.5 * self.log_vars[1]) + \
                        (torch.exp(-self.log_vars[2]) * loss_a + 0.5 * self.log_vars[2])
                else:
                    # Fallback to simple weighted sum if uncertainty loss is turned off
                    loss = loss_mot + loss_v + loss_a
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
            
            
    def fusion_train(self, save_path="weights/best_fusion_v31.pt"): 
        tr_loader = make_multitask_loader(self.cfg, self.train_keys, shuffle=True, dp=self)
        va_loader = make_multitask_loader(self.cfg, self.val_keys, shuffle=False, dp=self)
        
        warmup_epochs = getattr(self.cfg.MainTask, 'warmup_epochs', 10)
        total_epochs = self.cfg.MainTask.epochs
        
        best_loss, patience_counter, best_performance = float('inf'), 0, {}

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

                if va_loss < best_loss:
                    best_loss, patience_counter = va_loss, 0
                    torch.save({'model_state_dict': self.state_dict()}, save_path)
                    print(f"  -> Best warm-up model saved to {save_path}")

        # --- 2단계: 미세 조정 (Fine-tuning) ---
        print("\n" + "="*50)
        print(f"🚀 STARTING STAGE 2: FINE-TUNING FOR {total_epochs - warmup_epochs} EPOCHS")
        print("="*50)
        
        # 1단계에서 최적 모델 로드 (warmup을 안했으면 사전학습 모델 그대로 사용)
        if warmup_epochs > 0:
            print(f"Loading best model from warm-up stage: {save_path}")
            self.load_state_dict(torch.load(save_path)['model_state_dict'])

        self._set_encoder_grad(requires_grad=True)
        self._create_optimizer_stage2()
        
        # best_loss와 patience는 2단계에서 이어서 사용하거나 초기화할 수 있음 (여기서는 이어서 사용)
        patience_counter = 0 

        for epoch in range(warmup_epochs + 1, total_epochs + 1):
            self.current_epoch = epoch
            tr_loss = self.run_epoch(tr_loader, train=True)
            va_loss = self.run_epoch(va_loader, train=False)
            va_acc_mot, va_acc_v, va_acc_a = self.evaluate(va_loader)
            print(f"[FINE-TUNE] Epoch {epoch:02d}/{total_epochs} | Tr Loss {tr_loss:.4f} | Val Loss {va_loss:.4f} | Acc(M/V/A): {va_acc_mot:.3f}/{va_acc_v:.3f}/{va_acc_a:.3f}")

            if va_loss < best_loss:
                best_loss, patience_counter = va_loss, 0
                best_performance = {'mot_acc': va_acc_mot, 'val_acc': va_acc_v, 'aro_acc': va_acc_a, 'best_loss': best_loss}
                torch.save({
                    'model_state_dict': self.state_dict(), 
                    'optimizer_state_dict': self.optim.state_dict(),
                }, save_path)
                print(f"  -> Best fine-tuned model saved to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.MainTask.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        print(f"\n--- Best Validation Performance (v24) ---")
        print(f"Loss: {best_performance.get('best_loss', 'N/A'):.4f} | Acc(M/V/A): {best_performance.get('mot_acc', 0):.3f}/{best_performance.get('val_acc', 0):.3f}/{best_performance.get('aro_acc', 0):.3f}")
        return best_performance