"""
v30.2: Gated Selective Attention Fusion (Final, Corrected)
- Base: 가장 강력한 사전학습 감정 모델을 기본骨格으로 사용.
- Context: 감정 특징 시퀀스가 모션 특징 시퀀스 전체를 Cross-Attention으로 참조.
- Gating: 학습 가능한 게이트를 통해 모션 컨텍스트를 참고할지, 기존 감정 특징을 사용할지 동적으로 결정.
- Training: 안정적인 2단계 훈련(Warm-up + Fine-tuning) 방식을 유지.
- Fix: 각 Encoder를 생성할 때, 해당 사전학습 단계에서 사용된 config를 전달하여 모델 구조의 완전한 일치를 보장.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from copy import deepcopy
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
        print("\n--- Building Final Fusion Model (Part-by-Part Assembly + Gated Attention) ---")
        
        mot_hidden_dim = self.cfg.PretrainMotion.hidden_dim
        emo_hidden_dim = self.cfg.PretrainEmotion.hidden_dim
        fusion_hidden_dim = mot_hidden_dim

        self.nets = nn.ModuleDict()
        self.projs = nn.ModuleDict()

        # --- 1. 서브-인코더 생성 (원본 Config 그대로 사용) ---
        # 이 방식은 각 인코더가 사전학습 때와 100% 동일한 구조로 생성됨을 보장합니다.
        mot_modalities = set(self.cfg.PretrainMotion.modalities_to_use)
        if 'imu' in mot_modalities:
            self.nets['motion_imu'] = IMUFeatureEncoder(self.cfg.Encoders.imu)
            self.projs['motion_imu'] = nn.Linear(self.cfg.Encoders.imu['encoder_dim'], mot_hidden_dim)
        if 'veh' in mot_modalities:
            self.nets['motion_veh'] = VehicleTCNEncoder(self.cfg.Encoders.veh)
            self.projs['motion_veh'] = nn.Linear(self.cfg.Encoders.veh['embed_dim'], mot_hidden_dim)

        emo_modalities = set(self.cfg.PretrainEmotion.modalities_to_use)
        if 'ppg' in emo_modalities:
            self.nets['emotion_ppg'] = PPGEncoder(self.cfg.Encoders.ppg)
            self.projs['emotion_ppg'] = nn.Linear(self.cfg.Encoders.ppg['embed_dim'] + 6, emo_hidden_dim)
        if 'sc' in emo_modalities:
            self.nets['emotion_sc'] = ScenarioEmbedding(self.cfg.Encoders.sc)
            self.projs['emotion_sc'] = nn.Linear(self.cfg.Encoders.sc['embed_dim'], emo_hidden_dim)
        if 'survey' in emo_modalities:
            self.nets['emotion_survey'] = PreSurveyEncoder(self.cfg.Encoders.survey)
            self.projs['emotion_survey'] = nn.Linear(self.cfg.Encoders.survey['embed_dim'], emo_hidden_dim)
        if 'veh' in emo_modalities:
            self.nets['emotion_veh'] = VehicleTCNEncoder(self.cfg.Encoders.veh)
            self.projs['emotion_veh'] = nn.Linear(self.cfg.Encoders.veh['embed_dim'], emo_hidden_dim)

        # --- 2. 예측 헤드 정의 ---
        self.motion_head = MotionHead(mot_hidden_dim, self.cfg.PretrainMotion.num_motion)

        # EmotionPredictor의 input_dim을 사전학습(emotion_encoder.py) 때와 동일한 방식으로 계산
        dynamic_emo_mods = sorted([m for m in emo_modalities if m != 'survey'])
        
        # emotion_encoder.py의 forward 로직에 따른 fused_dim 계산
        # 이 부분은 emotion_encoder.py의 forward 로직과 정확히 일치해야 함
        if 'imu' in emo_modalities:
             # imu(시퀀스) + 다른 pooled 모달리티들
             predictor_input_dim = emo_hidden_dim + emo_hidden_dim * (len(dynamic_emo_mods) - 1)
        else:
             # pooled 모달리티들만 존재
             predictor_input_dim = emo_hidden_dim * len(dynamic_emo_mods)

        self.valence_predictor = EmotionPredictor(predictor_input_dim, emo_hidden_dim, self.cfg.PretrainEmotion.num_valence)
        self.arousal_predictor = EmotionPredictor(predictor_input_dim, emo_hidden_dim, self.cfg.PretrainEmotion.num_arousal)

        self.final_valence_predictor = EmotionPredictor(fusion_hidden_dim, emo_hidden_dim, self.cfg.PretrainEmotion.num_valence)
        self.final_arousal_predictor = EmotionPredictor(fusion_hidden_dim, emo_hidden_dim, self.cfg.PretrainEmotion.num_arousal)

        # --- 3. Adapter 및 퓨전 모듈 ---
        if emo_hidden_dim != fusion_hidden_dim:
            self.emotion_adapter = nn.Linear(emo_hidden_dim, fusion_hidden_dim)

        dropout_p = getattr(self.cfg.FusionModel, 'dropout', 0.3)
        num_heads = getattr(self.cfg.FusionModel, 'num_heads', 4)
        self.cross_attention = nn.MultiheadAttention(fusion_hidden_dim, num_heads, dropout=dropout_p, batch_first=True)
        self.context_norm = nn.LayerNorm(fusion_hidden_dim)
        self.gate = nn.Sequential(nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim), nn.Sigmoid())
        
        # --- 4. 로스 및 유틸 ---
        self.log_vars = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        self.scaler = torch.cuda.amp.GradScaler()
        self.to(self.cfg.Project.device)

    def _load_pretrained_weights(self):
        device = self.cfg.Project.device
        def _safe_load(module, state_dict, prefix, tag):
            if state_dict is None: return
            sub_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)} if prefix else state_dict
            if not sub_dict: return
            module.load_state_dict(sub_dict, strict=False)
            print(f"[Load OK] {tag} (prefix: {prefix})")

        # --- Motion 가중치 로드 ---
        mot_cfg = self.cfg.PretrainMotion
        mot_ckpt = torch.load(getattr(mot_cfg, 'ckpt_path'), map_location=device)
        mot_enc_states = mot_ckpt.get('encoder')
        if 'imu' in mot_cfg.modalities_to_use:
            _safe_load(self.nets.motion_imu, mot_enc_states, 'imu.', 'Motion Encoder[imu]')
            _safe_load(self.projs.motion_imu, mot_enc_states, 'p_imu.', 'Motion Proj[imu]')
        if 'veh' in mot_cfg.modalities_to_use:
            _safe_load(self.nets.motion_veh, mot_enc_states, 'veh.', 'Motion Encoder[veh]')
            _safe_load(self.projs.motion_veh, mot_enc_states, 'p_veh.', 'Motion Proj[veh]')
        _safe_load(self.motion_head, mot_ckpt.get('head'), None, "Motion Head")

        # --- Emotion 가중치 로드 ---
        emo_cfg = self.cfg.PretrainEmotion
        emo_ckpt = torch.load(getattr(emo_cfg, 'ckpt_path'), map_location=device)
        emo_enc_states = emo_ckpt.get('encoder')
        if 'ppg' in emo_cfg.modalities_to_use:
            _safe_load(self.nets.emotion_ppg, emo_enc_states, 'nets.ppg.', 'Emotion Encoder[ppg]')
            _safe_load(self.projs.emotion_ppg, emo_enc_states, 'projs.ppg.', 'Emotion Proj[ppg]')
        if 'sc' in emo_cfg.modalities_to_use:
            _safe_load(self.nets.emotion_sc, emo_enc_states, 'nets.sc.', 'Emotion Encoder[sc]')
            _safe_load(self.projs.emotion_sc, emo_enc_states, 'projs.sc.', 'Emotion Proj[sc]')
        if 'veh' in emo_cfg.modalities_to_use:
            _safe_load(self.nets.emotion_veh, emo_enc_states, 'nets.veh.', 'Emotion Encoder[veh]')
            _safe_load(self.projs.emotion_veh, emo_enc_states, 'projs.veh.', 'Emotion Proj[veh]')
        if 'survey' in emo_cfg.modalities_to_use:
            _safe_load(self.nets.emotion_survey, emo_enc_states, 'nets.survey.', 'Emotion Encoder[survey]')
            _safe_load(self.projs.emotion_survey, emo_enc_states, 'projs.survey.', 'Emotion Proj[survey]')
        _safe_load(self.valence_predictor, emo_ckpt.get('valence_predictor'), None, "Valence Predictor")
        _safe_load(self.arousal_predictor, emo_ckpt.get('arousal_predictor'), None, "Arousal Predictor")

    def _set_encoder_grad(self, requires_grad: bool):
        status = "Unfrozen" if requires_grad else "Frozen"
        print(f"\n--- Setting Pre-trained Encoders to {status} ---")
        # nets와 projs의 모든 파라미터를 동결/해제
        for param in itertools.chain(self.nets.parameters(), self.projs.parameters()):
            param.requires_grad = requires_grad

    def _create_optimizer_stage1(self):
        print("\n--- Creating optimizer for Stage 1: Warm-up (Fusion & Heads) ---")
        base_lr, base_wd = self.cfg.MainTask.lr, self.cfg.MainTask.weight_decay
        
        trainable_params = [
            self.cross_attention.parameters(), self.context_norm.parameters(), self.gate.parameters(),
            self.motion_head.parameters(),
            self.final_valence_predictor.parameters(), # <--- 수정
            self.final_arousal_predictor.parameters()  # <--- 수정
        ]
        if hasattr(self, 'emotion_adapter'):
            trainable_params.append(self.emotion_adapter.parameters())

        optim_groups = [{'params': itertools.chain(*trainable_params), 'lr': base_lr, 'weight_decay': base_wd, 'name': 'Fusion & Heads'}]
        optim_groups.append({'params': self.log_vars, 'lr': base_lr, 'weight_decay': 0.0, 'name': 'LogVars'})
        self.optim = torch.optim.Adam(optim_groups)

    def _create_optimizer_stage2(self):
        print("\n--- Creating optimizer for Stage 2: Fine-tuning ---")
        base_lr, base_wd = self.cfg.MainTask.lr, self.cfg.MainTask.weight_decay
        encoder_lr = base_lr / 10

        encoder_params = itertools.chain(self.nets.parameters(), self.projs.parameters())
        
        fusion_params = [
            self.cross_attention.parameters(), self.context_norm.parameters(), self.gate.parameters(),
            self.motion_head.parameters(),
            self.final_valence_predictor.parameters(), # <--- 수정
            self.final_arousal_predictor.parameters()  # <--- 수정
        ]
        if hasattr(self, 'emotion_adapter'):
            fusion_params.append(self.emotion_adapter.parameters())
            
        optim_groups = [
            {'params': encoder_params, 'lr': encoder_lr, 'weight_decay': base_wd, 'name': 'Encoders (Low LR)'},
            {'params': itertools.chain(*fusion_params), 'lr': base_lr, 'weight_decay': base_wd, 'name': 'Fusion & Heads (Base LR)'}
        ]
        optim_groups.append({'params': self.log_vars, 'lr': base_lr, 'weight_decay': 0.0, 'name': 'LogVars'})
        self.optim = torch.optim.Adam(optim_groups)
    
    def _process_hrv(self, batch, device):
        ppg_rr = batch['ppg_rr_emotion'].to(device)
        hrv_list = []
        if ppg_rr.dim() > 1 and ppg_rr.shape[1] > 0:
            hrv_list.append(torch.nan_to_num(ppg_rr.mean(dim=1, keepdim=True), nan=0.0))
            std_dev = torch.std(ppg_rr, dim=1, keepdim=True); hrv_list.append(torch.nan_to_num(std_dev, nan=0.0))
            hrv_list.append(torch.min(ppg_rr, dim=1, keepdim=True).values); hrv_list.append(torch.max(ppg_rr, dim=1, keepdim=True).values)
        else: hrv_list.extend([torch.zeros(ppg_rr.shape[0], 1, device=device)] * 4)
        for key in ['ppg_rmssd_emotion', 'ppg_sdnn_emotion']:
            hrv_list.append(torch.nan_to_num(batch[key].to(device).unsqueeze(1), nan=0.0))
        return torch.cat(hrv_list, dim=1)
    
    def forward(self, batch):
        device = self.cfg.Project.device
        T_mot = batch['label_motion'].shape[1]

        # --- 1. 독립적인 특징 추출 ---

        # Motion 특징 추출 (기존과 동일)
        mot_parts = []
        mot_modalities = set(self.cfg.PretrainMotion.modalities_to_use)
        if 'imu' in mot_modalities:
            imu_len = (batch['imu_motion'].abs().sum(-1) > 0).sum(1)
            out = self.nets.motion_imu(batch['imu_motion'].to(device), imu_len)
            mot_parts.append(self.projs.motion_imu(out))
        if 'veh' in mot_modalities:
            out_pooled = self.nets.motion_veh(batch['veh_motion'].to(device).permute(0,2,1), return_pooled=True)
            out_seq = self.projs.motion_veh(out_pooled).unsqueeze(1).expand(-1, T_mot, -1)
            mot_parts.append(out_seq)
        
        # [개선 제안] 여러 모션 모달리티가 있을 경우를 대비해 평균을 사용 (더 안정적)
        if len(mot_parts) > 1:
            motion_feat_seq = torch.stack(mot_parts, dim=0).mean(dim=0)
        else:
            motion_feat_seq = mot_parts[0]

        # --- Emotion 특징 추출 (오류가 발생하는 핵심 부분 수정) ---
        emo_modalities = set(self.cfg.PretrainEmotion.modalities_to_use)
        pooled_parts = [] # Pooled된 특징만 담을 리스트

        # [수정] 모든 동적 감정 모달리티를 처리하여 pooled_parts에 추가
        dynamic_emo_mods = sorted([m for m in emo_modalities if m != 'survey'])
        for mod in dynamic_emo_mods:
            if mod == 'ppg':
                tcn_out = self.nets.emotion_ppg(batch['ppg_emotion'].to(device).permute(0, 2, 1))
                hrv = self._process_hrv(batch, device)
                combined = torch.cat([tcn_out, hrv], dim=1)
                pooled_parts.append(self.projs.emotion_ppg(combined))
            elif mod == 'sc':
                sc_out = self.nets.emotion_sc(batch['scenario_evt_e'].to(device), batch['scenario_type_e'].to(device), batch['phase_evt_e'].to(device), batch['scenario_time_e'].to(device))
                pooled_parts.append(self.projs.emotion_sc(sc_out))
            elif mod == 'veh':
                veh_out = self.nets.emotion_veh(batch['veh_emotion'].to(device).permute(0, 2, 1), return_pooled=True)
                pooled_parts.append(self.projs.emotion_veh(veh_out))
        
        # [핵심 수정 1] 퓨전을 위한 대표 감정 특징 생성
        # 모든 pooled 특징들의 평균을 내어 하나의 대표 벡터를 만듭니다. (차원 불일치 오류 해결)
        emotion_feat_pooled = sum(pooled_parts) / len(pooled_parts)
        # 이 대표 벡터를 모션 시퀀스 길이에 맞게 복제하여 시퀀스 특징으로 만듭니다.
        emotion_feat_seq = emotion_feat_pooled.unsqueeze(1).expand(-1, T_mot, -1)

        # 정적 특징 (Survey) (기존과 동일)
        static_feat = None
        if 'survey' in emo_modalities:
            static_feat = self.projs.emotion_survey(self.nets.emotion_survey(batch['survey_e'].to(device)))
            
        # --- 2. 모션 예측 (퓨전과 독립적으로 수행) ---
        mot_logits = self.motion_head(motion_feat_seq.permute(0, 2, 1))

        # --- 3. Adapter, Fusion, Gating ---
        if hasattr(self, 'emotion_adapter'):
            adapted_emotion_seq = self.emotion_adapter(emotion_feat_seq)
        else:
            adapted_emotion_seq = emotion_feat_seq # (B, T, fusion_hidden_dim)
        
        context_aware_feat, _ = self.cross_attention(query=adapted_emotion_seq, key=motion_feat_seq.detach(), value=motion_feat_seq.detach())
        context_aware_feat = self.context_norm(adapted_emotion_seq + context_aware_feat)

        gate_input = torch.cat([adapted_emotion_seq, context_aware_feat], dim=-1)
        g = self.gate(gate_input)
        final_emotion_feat_seq = g * context_aware_feat + (1 - g) * adapted_emotion_seq
        
        # --- 4. 최종 감정 예측 ---
        # [핵심 수정 2] 퓨전된 최종 특징으로 예측 수행
        # 사전 학습된 predictor가 아닌, 퓨전용으로 새로 만든 final_predictor를 사용합니다.
        # 입력으로 퓨전의 최종 결과물인 'final_emotion_feat_seq'를 전달합니다.
        valence_logits = self.final_valence_predictor({'fused': final_emotion_feat_seq}, context=static_feat)
        arousal_logits = self.final_arousal_predictor({'fused': final_emotion_feat_seq}, context=static_feat)

        return {'motion_logits': mot_logits, 'valence_logits': valence_logits, 'arousal_logits': arousal_logits}

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

                val_logits = out['valence_logits']
                raw_v = batch['valence_reg_emotion'].reshape(-1)
                tgt_v = torch.full_like(raw_v, -100, dtype=torch.long, device=device)
                tgt_v[raw_v < 4] = 0; tgt_v[(raw_v >= 4) & (raw_v < 7)] = 1; tgt_v[raw_v >= 7] = 2
                mask_v = (tgt_v != -100)
                loss_v = F.cross_entropy(val_logits[mask_v], tgt_v[mask_v]) if mask_v.any() else torch.tensor(0., device=device)

                aro_logits = out['arousal_logits']
                raw_a = batch['arousal_reg_emotion'].reshape(-1)
                tgt_a = torch.full_like(raw_a, -100, dtype=torch.long, device=device)
                tgt_a[raw_a < 4] = 0; tgt_a[(raw_a >= 4) & (raw_a < 7)] = 1; tgt_a[raw_a >= 7] = 2
                mask_a = (tgt_a != -100)
                loss_a = F.cross_entropy(aro_logits[mask_a], tgt_a[mask_a]) if mask_a.any() else torch.tensor(0., device=device)
                
                loss = (torch.exp(-self.log_vars[0]) * loss_mot + 0.5 * self.log_vars[0]) + \
                       (torch.exp(-self.log_vars[1]) * loss_v + 0.5 * self.log_vars[1]) + \
                       (torch.exp(-self.log_vars[2]) * loss_a + 0.5 * self.log_vars[2])

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
            
            p_v = out['valence_logits'].argmax(-1).cpu(); raw_v = batch['valence_reg_emotion'].view(-1).cpu()
            t_v = torch.full_like(raw_v, -1, dtype=torch.long); t_v[raw_v<4]=0; t_v[(raw_v>=4)&(raw_v<7)]=1; t_v[raw_v>=7]=2
            all_preds['valence_preds'].append(p_v[t_v != -1]); all_preds['valence_trues'].append(t_v[t_v != -1])
            
            p_a = out['arousal_logits'].argmax(-1).cpu(); raw_a = batch['arousal_reg_emotion'].view(-1).cpu()
            t_a = torch.full_like(raw_a, -1, dtype=torch.long); t_a[raw_a<4]=0; t_a[(raw_a>=4)&(raw_a<7)]=1; t_a[raw_a>=7]=2
            all_preds['arousal_preds'].append(p_a[t_a != -1]); all_preds['arousal_trues'].append(t_a[t_a != -1])
        for key in all_preds: 
            if len(all_preds[key]) > 0:
                all_preds[key] = torch.cat(all_preds[key])
        if return_preds: return all_preds
        acc_mot = accuracy_score(all_preds['motion_trues'], all_preds['motion_preds']) if len(all_preds['motion_trues']) > 0 else 0
        acc_v = accuracy_score(all_preds['valence_trues'], all_preds['valence_preds']) if len(all_preds['valence_trues']) > 0 else 0
        acc_a = accuracy_score(all_preds['arousal_trues'], all_preds['arousal_preds']) if len(all_preds['arousal_trues']) > 0 else 0
        return acc_mot, acc_v, acc_a
            
    def fusion_train(self, save_path="weights/best_fusion_v30.pt"): 
        tr_loader = make_multitask_loader(self.cfg, self.train_keys, shuffle=True, dp=self)
        va_loader = make_multitask_loader(self.cfg, self.val_keys, shuffle=False, dp=self)
        
        warmup_epochs = getattr(self.cfg.MainTask, 'warmup_epochs', 10)
        total_epochs = self.cfg.MainTask.epochs
        best_loss, patience_counter, best_performance = float('inf'), 0, {}

        # 1단계
        if warmup_epochs > 0:
            print("\n" + "="*50 + f"\n🚀 STARTING STAGE 1: WARM-UP FOR {warmup_epochs} EPOCHS\n" + "="*50)
            self._set_encoder_grad(requires_grad=False)
            self._create_optimizer_stage1()
            for epoch in range(1, warmup_epochs + 1):
                tr_loss = self.run_epoch(tr_loader, train=True)
                va_loss = self.run_epoch(va_loader, train=False)
                va_acc_mot, va_acc_v, va_acc_a = self.evaluate(va_loader)
                print(f"[WARM-UP] Epoch {epoch:02d}/{warmup_epochs} | Tr Loss {tr_loss:.4f} | Val Loss {va_loss:.4f} | Acc(M/V/A): {va_acc_mot:.3f}/{va_acc_v:.3f}/{va_acc_a:.3f}")
                if va_loss < best_loss:
                    best_loss, patience_counter = va_loss, 0
                    torch.save({'model_state_dict': self.state_dict()}, save_path)
                    print(f"  -> Best warm-up model saved.")

        # 2단계
        print("\n" + "="*50 + f"\n🚀 STARTING STAGE 2: FINE-TUNING FOR {total_epochs - warmup_epochs} EPOCHS\n" + "="*50)
        if warmup_epochs > 0:
            print(f"Loading best model from warm-up stage.")
            self.load_state_dict(torch.load(save_path)['model_state_dict'])
        self._set_encoder_grad(requires_grad=True)
        self._create_optimizer_stage2()
        patience_counter = 0 
        for epoch in range(warmup_epochs + 1, total_epochs + 1):
            tr_loss = self.run_epoch(tr_loader, train=True)
            va_loss = self.run_epoch(va_loader, train=False)
            va_acc_mot, va_acc_v, va_acc_a = self.evaluate(va_loader)
            print(f"[FINE-TUNE] Epoch {epoch:02d}/{total_epochs} | Tr Loss {tr_loss:.4f} | Val Loss {va_loss:.4f} | Acc(M/V/A): {va_acc_mot:.3f}/{va_acc_v:.3f}/{va_acc_a:.3f}")
            if va_loss < best_loss:
                best_loss, patience_counter = va_loss, 0
                best_performance = {'mot_acc': va_acc_mot, 'val_acc': va_acc_v, 'aro_acc': va_acc_a, 'best_loss': best_loss}
                torch.save({'model_state_dict': self.state_dict(), 'optimizer_state_dict': self.optim.state_dict()}, save_path)
                print(f"  -> Best fine-tuned model saved.")
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.MainTask.patience:
                    print(f"Early stopping at epoch {epoch}"); break
        
        print(f"\n--- Best Validation Performance (v30 Gated Fusion) ---")
        print(f"Loss: {best_performance.get('best_loss', 'N/A'):.4f} | Acc(M/V/A): {best_performance.get('mot_acc', 0):.3f}/{best_performance.get('val_acc', 0):.3f}/{best_performance.get('aro_acc', 0):.3f}")
        return best_performance