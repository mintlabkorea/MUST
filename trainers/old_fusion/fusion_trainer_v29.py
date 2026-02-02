"""
v29: Mid-Level Fusion
Late-Fusion의 한계를 극복하기 위해 패러다임을 전환.
- 4개의 전문가 인코더(imu, veh, ppg, survey)는 사전학습 가중치를 사용하고 동결(또는 미세조정).
- 이들의 출력을 결합하여, 처음부터 학습하는 새로운 'fusion_encoder'에 통과.
- fusion_encoder가 생성한 통합 특징을 기반으로 모든 태스크를 예측.
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
        
        self._build_model()
        self._load_pretrained_weights()

    def _build_model(self):
        print("\n--- Building V29: Mid-Level Fusion Architecture ---")
        hidden_dim = self.cfg.FusionModel.hidden_dim
        dropout_p = getattr(self.cfg.FusionModel, 'dropout', 0.3) # 과적합 방지를 위해 Dropout 소폭 증가
        C_mot = self.cfg.PretrainMotion.num_motion if getattr(self.cfg.FusionModel, "emo_use_mot_logits", True) else 0

        # --- 1. 전문가 인코더 및 프로젝터 정의 (sc 제외) ---
        self.nets = nn.ModuleDict()
        self.projs = nn.ModuleDict()

        # Motion 전문가
        self.nets['imu'] = IMUFeatureEncoder(self.cfg.Encoders.imu)
        self.projs['imu'] = nn.Linear(self.cfg.Encoders.imu['encoder_dim'], hidden_dim)
        self.nets['veh'] = VehicleTCNEncoder(self.cfg.Encoders.veh)
        self.projs['veh'] = nn.Linear(self.cfg.Encoders.veh['embed_dim'], hidden_dim)

        # Emotion 전문가
        self.nets['ppg'] = PPGEncoder(self.cfg.Encoders.ppg)
        self.projs['ppg'] = nn.Linear(self.cfg.Encoders.ppg['embed_dim'] + 6, hidden_dim)
        self.nets['survey'] = PreSurveyEncoder(self.cfg.Encoders.survey)
        self.projs['survey'] = nn.Linear(self.cfg.Encoders.survey['embed_dim'], hidden_dim)

        # --- 2. 통합 인코더 (Fusion Encoder) 정의 ---
        # 4개의 전문가 인코더 특징(각각 hidden_dim)이 합쳐진 차원을 입력으로 받음
        fusion_input_dim = hidden_dim * 4
        self.fusion_encoder = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # --- 3. 최종 예측 헤드 정의 ---
        # 모든 헤드는 fusion_encoder의 출력(hidden_dim)을 입력으로 받음
        self.motion_head = MotionHead(hidden_dim, self.cfg.PretrainMotion.num_motion)
        
        predictor_in = hidden_dim + C_mot
        self.emotion_valence_predictor = EmotionPredictor(predictor_in, hidden_dim, self.cfg.PretrainEmotion.num_valence)
        self.emotion_arousal_predictor = EmotionPredictor(predictor_in, hidden_dim, self.cfg.PretrainEmotion.num_arousal)
        
        # --- 4. 로스 및 유틸 ---
        init_log_vars = torch.tensor([0.0, 0.0, 0.0]) # 3개 태스크 (mot, v, a)
        self.log_vars = nn.Parameter(init_log_vars)
        self.scaler = torch.cuda.amp.GradScaler()
        self.to(self.cfg.Project.device)

    def _load_pretrained_weights(self):
        device = self.cfg.Project.device
        def _get_enc(ckpt): return ckpt.get('encoder', ckpt)
        def _safe_load(module, state_dict, prefix, tag):
            if state_dict is None: return
            sub_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)} if prefix else state_dict
            if not sub_dict: return
            module.load_state_dict(sub_dict, strict=False)
            print(f"[Load OK] {tag}")

        # Emotion 전문가 가중치 로드 (ppg, survey)
        emo_cfg = self.cfg.PretrainEmotion
        emo_modalities = getattr(emo_cfg, 'modalities_to_use', [])
        emo_ckpt = torch.load(getattr(emo_cfg, 'ckpt_path', ''), map_location=device) if emo_modalities else None
        if 'ppg' in emo_modalities:
            _safe_load(self.nets.ppg, emo_ckpt, 'nets.ppg.', 'Emotion Encoder[ppg]')
            _safe_load(self.projs.ppg, emo_ckpt, 'projs.ppg.', 'Emotion Projection[ppg]')
        if 'survey' in emo_modalities:
            _safe_load(self.nets.survey, emo_ckpt, 'nets.survey.', 'Emotion Encoder[survey]')
            _safe_load(self.projs.survey, emo_ckpt, 'projs.survey.', 'Emotion Projection[survey]')
        
        # Motion 전문가 가중치 로드 (imu, veh)
        mot_cfg = self.cfg.PretrainMotion
        mot_modalities = getattr(mot_cfg, 'modalities_to_use', [])
        mot_ckpt = torch.load(getattr(mot_cfg, 'ckpt_path', ''), map_location=device) if mot_modalities else None
        if 'imu' in mot_modalities:
            _safe_load(self.nets.imu, mot_ckpt, 'imu.', 'Motion Encoder[imu]')
            _safe_load(self.projs.imu, mot_ckpt, 'p_imu.', 'Motion Projection[imu]')
        if 'veh' in mot_modalities:
            _safe_load(self.nets.veh, mot_ckpt, 'veh.', 'Motion Encoder[veh]')
            _safe_load(self.projs.veh, mot_ckpt, 'p_veh.', 'Motion Projection[veh]')
        _safe_load(self.motion_head, mot_ckpt.get('head'), None, 'Motion Head')

    def _set_encoder_grad(self, requires_grad: bool):
        # 전문가 인코더들의 동결/해제 상태 제어
        status = "Unfrozen" if requires_grad else "Frozen"
        print(f"\n--- Setting Expert Encoders to {status} ---")
        expert_params = itertools.chain(self.nets.parameters(), self.projs.parameters())
        for param in expert_params:
            param.requires_grad = requires_grad

    def _create_optimizer_stage1(self):
        print("\n--- Creating optimizer for Stage 1: Warm-up (Fusion Layers Only) ---")
        base_lr, base_wd = self.cfg.MainTask.lr, self.cfg.MainTask.weight_decay
        
        # 새로 학습할 파라미터: fusion_encoder와 모든 예측 헤드
        trainable_params = itertools.chain(
            self.fusion_encoder.parameters(),
            self.motion_head.parameters(),
            self.emotion_valence_predictor.parameters(),
            self.emotion_arousal_predictor.parameters()
        )
        optim_groups = [{'params': trainable_params, 'lr': base_lr, 'weight_decay': base_wd, 'name': 'Fusion Layers'}]
        optim_groups.append({'params': self.log_vars, 'lr': base_lr, 'weight_decay': 0.0, 'name': 'LogVars'})
        
        self.optim = torch.optim.Adam(optim_groups)
        for group in self.optim.param_groups:
            print(f" - Group '{group['name']}': lr={group['lr']:.1e}, wd={group['weight_decay']:.1e}")
            
    def _create_optimizer_stage2(self):
        print("\n--- Creating optimizer for Stage 2: Fine-tuning (Differential LR) ---")
        base_lr, base_wd = self.cfg.MainTask.lr, self.cfg.MainTask.weight_decay
        encoder_lr = base_lr / 10

        # 전문가 인코더 그룹 (낮은 학습률)
        expert_params = itertools.chain(self.nets.parameters(), self.projs.parameters())
        
        # 퓨전/헤드 그룹 (기본 학습률)
        fusion_params = itertools.chain(
            self.fusion_encoder.parameters(),
            self.motion_head.parameters(),
            self.emotion_valence_predictor.parameters(),
            self.emotion_arousal_predictor.parameters()
        )
        optim_groups = [
            {'params': expert_params, 'lr': encoder_lr, 'weight_decay': base_wd, 'name': 'Expert Encoders (Low LR)'},
            {'params': fusion_params, 'lr': base_lr, 'weight_decay': base_wd, 'name': 'Fusion Layers (Base LR)'}
        ]
        optim_groups.append({'params': self.log_vars, 'lr': base_lr, 'weight_decay': 0.0, 'name': 'LogVars'})

        self.optim = torch.optim.Adam(optim_groups)
        for group in self.optim.param_groups:
            print(f" - Group '{group['name']}': lr={group['lr']:.1e}, wd={group['weight_decay']:.1e}")

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
        T_mot = batch['label_motion'].shape[1]

        # --- 1. 전문가 인코더로 특징 추출 ---
        # with torch.no_grad(): # Stage 2에서는 그래디언트 필요
        # Motion 전문가
        imu_feat_seq = self.projs['imu'](self.nets['imu'](batch['imu_motion'].to(device), (batch['imu_motion'].abs().sum(-1) > 0).sum(1)))
        imu_feat_pooled = imu_feat_seq.mean(dim=1)
        veh_feat = self.projs['veh'](self.nets['veh'](batch['veh_motion'].to(device).permute(0, 2, 1), return_pooled=True))
        
        # Emotion 전문가
        ppg_tcn = self.nets['ppg'](batch['ppg_emotion'].to(device).permute(0, 2, 1))
        ppg_combined = torch.cat([ppg_tcn, self._process_hrv(batch, device)], dim=1)
        ppg_feat = self.projs['ppg'](ppg_combined)
        survey_feat = self.projs['survey'](self.nets['survey'](batch['survey_e'].to(device)))

        # --- 2. 특징 결합 ---
        combined_features = torch.cat([imu_feat_pooled, veh_feat, ppg_feat, survey_feat], dim=1)

        # --- 3. 통합 인코더로 최종 특징 생성 ---
        fused_feature = self.fusion_encoder(combined_features)

        # --- 4. 최종 예측 ---
        fused_feature_seq = fused_feature.unsqueeze(1).expand(-1, T_mot, -1)
        mot_logits = self.motion_head(fused_feature_seq.permute(0, 2, 1))

        emotion_input = fused_feature
        if getattr(self.cfg.FusionModel, "emo_use_mot_logits", True):
            p_mot = F.softmax(mot_logits, dim=-1).mean(dim=1).detach()
            emotion_input = torch.cat([emotion_input, p_mot], dim=1)
        
        valence_logits = self.emotion_valence_predictor({'fused': emotion_input.unsqueeze(1)}, context=None)
        arousal_logits = self.emotion_arousal_predictor({'fused': emotion_input.unsqueeze(1)}, context=None)

        return {'motion_logits': mot_logits, 'valence_logits': valence_logits, 'arousal_logits': arousal_logits}

    def run_epoch(self, loader, train: bool):
        self.train(train)
        total_loss, num_batches = 0.0, 0
        device = self.cfg.Project.device

        for batch in loader:
            with torch.set_grad_enabled(train):
                out = self.forward(batch) 
                # Loss 계산 (Alignment Loss 제외)
                mot_logits, mot_labels = out['motion_logits'], batch['label_motion'].to(device)
                valid_mask = (mot_labels.reshape(-1) > 0) & (mot_labels.reshape(-1) != 4)
                loss_mot = F.cross_entropy(mot_logits.reshape(-1, mot_logits.shape[-1])[valid_mask], (mot_labels.reshape(-1)[valid_mask] - 1).long()) if valid_mask.any() else torch.tensor(0., device=device)

                val_logits, raw_v = out['valence_logits'], batch['valence_reg_emotion'].to(device).view(-1)
                tgt_v = torch.full_like(raw_v, -100, dtype=torch.long); tgt_v[raw_v < 4]=0; tgt_v[(raw_v>=4)&(raw_v<7)]=1; tgt_v[raw_v>=7]=2
                loss_v = F.cross_entropy(val_logits, tgt_v, ignore_index=-100)
                
                aro_logits, raw_a = out['arousal_logits'], batch['arousal_reg_emotion'].to(device).view(-1)
                tgt_a = torch.full_like(raw_a, -100, dtype=torch.long); tgt_a[raw_a < 4]=0; tgt_a[(raw_a>=4)&(raw_a<7)]=1; tgt_a[raw_a>=7]=2
                loss_a = F.cross_entropy(aro_logits, tgt_a, ignore_index=-100)

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

        for key in all_preds: all_preds[key] = torch.cat(all_preds[key])
        if return_preds: return all_preds
        
        acc_mot = accuracy_score(all_preds['motion_trues'], all_preds['motion_preds'])
        acc_v = accuracy_score(all_preds['valence_trues'], all_preds['valence_preds'])
        acc_a = accuracy_score(all_preds['arousal_trues'], all_preds['arousal_preds'])
        return acc_mot, acc_v, acc_a
            
    def fusion_train(self, save_path="weights/best_fusion_v29.pt"): 
        # ... (v28의 2-Stage 학습 로직과 동일, 함수 이름만 변경)
        tr_loader = make_multitask_loader(self.cfg, self.train_keys, shuffle=True, dp=self)
        va_loader = make_multitask_loader(self.cfg, self.val_keys, shuffle=False, dp=self)
        
        warmup_epochs = getattr(self.cfg.MainTask, 'warmup_epochs', 10)
        total_epochs = self.cfg.MainTask.epochs
        best_loss, patience_counter, best_performance = float('inf'), 0, {}

        # --- 1단계: 워밍업 ---
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

        # --- 2단계: 미세 조정 ---
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
        
        print(f"\n--- Best Validation Performance (v29 Mid-Level Fusion) ---")
        print(f"Loss: {best_performance.get('best_loss', 'N/A'):.4f} | Acc(M/V/A): {best_performance.get('mot_acc', 0):.3f}/{best_performance.get('val_acc', 0):.3f}/{best_performance.get('aro_acc', 0):.3f}")
        return best_performance