# V9 + PMF (soft freezing)
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

from models.fusion.fusion_block import SMFusionBlock, GMFusionBlock
from models.fusion.predictors import MotionPredictor, EmotionPredictor

class FusionTrainer(nn.Module, dataProcessor):
    def __init__(self, cfg: Config, train_keys, val_keys, test_keys):
        super().__init__()
        # Initialize dataProcessor for column settings
        dataProcessor.__init__(self, cfg)
        self.prepare()

        # Assign split keys
        self.train_keys, self.val_keys, self.test_keys = train_keys, val_keys, test_keys
        self.cfg = cfg

        # Build model components and training setup
        self._build_model()
        self._load_pretrained_weights()
        self._init_pmf() 
        self._create_optimizer()

    def _build_model(self):
        # emotion vehicle encoder (full channels + original embed_dim)
        veh_em_cfg = deepcopy(self.cfg.Encoders.veh)
        veh_encoder_emotion = VehicleTCNEncoder(veh_em_cfg)

        # motion vehicle encoder: only mode channel + fixed embed_dim=128
        veh_mot_cfg = deepcopy(self.cfg.Encoders.veh)
        veh_mot_cfg['input_dim'] = 1
        veh_mot_cfg['embed_dim'] = 128
        veh_encoder_motion = VehicleTCNEncoder(veh_mot_cfg)
        # projection input dim = actual encoder output dim
        motion_veh_embed_dim = veh_mot_cfg['embed_dim']

        self.nets = nn.ModuleDict({
            'imu': IMUFeatureEncoder(self.cfg.Encoders.imu),
            'ppg': PPGEncoder(self.cfg.Encoders.ppg),
            'sc' : ScenarioEmbedding(self.cfg.Encoders.sc),
            'veh_e': veh_encoder_emotion,
            'veh_m': veh_encoder_motion,
            'survey': PreSurveyEncoder(self.cfg.Encoders.survey),
        })


        hidden_dim = self.cfg.FusionModel.hidden_dim
        ppg_in_dim = self.cfg.Encoders.ppg['embed_dim'] + 6

        self.projs = nn.ModuleDict({
            'imu':    nn.Linear(self.cfg.Encoders.imu['encoder_dim'], hidden_dim),
            'ppg':    nn.Linear(ppg_in_dim, hidden_dim),
            'sc':     nn.Linear(self.cfg.Encoders.sc['embed_dim'],  hidden_dim),
            'survey': nn.Linear(self.cfg.Encoders.survey['embed_dim'], hidden_dim),
            'veh_e':  nn.Linear(veh_em_cfg['embed_dim'], hidden_dim),
            'veh_m':  nn.Linear(motion_veh_embed_dim, hidden_dim),
        })

        self.motion_modalities = ['imu', 'veh_m', 'sc']
        self.emotion_modalities = ['imu', 'ppg', 'veh_e', 'sc', 'survey']
        
        self.motion_gm_fuse = GMFusionBlock(modalities=self.motion_modalities, hidden_dim=hidden_dim, num_heads=self.cfg.FusionModel.num_heads)
        self.emotion_gm_fuse = GMFusionBlock(modalities=self.emotion_modalities, hidden_dim=hidden_dim, num_heads=self.cfg.FusionModel.num_heads)

        self.motion_predictor = MotionPredictor(hidden_dim, self.cfg.PretrainMotion.num_motion)
        self.emotion_valence_predictor = EmotionPredictor(hidden_dim, hidden_dim, self.cfg.PretrainEmotion.num_valence)
        self.emotion_arousal_predictor = EmotionPredictor(hidden_dim, hidden_dim, self.cfg.PretrainEmotion.num_arousal)
        
        init_log_vars = torch.tensor([0.22, 0.0, -0.6, 1.0])
        self.log_vars = nn.Parameter(init_log_vars)
        self.scaler = torch.cuda.amp.GradScaler()

        self.ce_mot = nn.CrossEntropyLoss(ignore_index=self.cfg.PretrainMotion.ignore_index)
        self.ce_v = nn.CrossEntropyLoss(ignore_index=self.cfg.PretrainEmotion.ignore_index)
        self.ce_a = nn.CrossEntropyLoss(ignore_index=self.cfg.PretrainEmotion.ignore_index)
        
        self.to(self.cfg.Project.device)

    def _init_pmf(self):
        """PMF를 위한 데이터 구조 초기화"""
        self.pmf_modalities = ['imu', 'ppg', 'veh_e', 'veh_m', 'sc', 'survey']
        self.ths = {
           'imu': 3.0,
           'ppg': 67.0,
           'veh_e': 155.0,
           'veh_m': 103,
           'sc': 1.6,
           'survey': 4.0
        }
        self.loss_mask = {} # 샘플별 마스크 (데이터 로드 후 생성)
        self.memory_mask = {} # 한번 얼린 샘플은 계속 유지하기 위한 마스크
        print("PMF module initialized.")


    def setup_pmf_masks(self, num_samples):
        """데이터셋 크기에 맞춰 마스크를 초기화"""
        for key in self.pmf_modalities:
            device = self.cfg.Project.device
            self.loss_mask[key] = torch.ones(num_samples, dtype=torch.float, device=device)
            self.memory_mask[key] = torch.ones(num_samples, dtype=torch.float, device=device)
        print(f"PMF masks created for {num_samples} samples.")

    def _load_pretrained_weights(self):
        device = self.cfg.Project.device
        # emotion weights
        emo_ckpt = torch.load('weights/best_emotion_tri.pt', map_location=device)['encoder']
        self.nets.ppg.load_state_dict({k.replace('nets.ppg.',''):v for k,v in emo_ckpt.items() if k.startswith('nets.ppg.')}, strict=False)
        self.nets.sc.load_state_dict({k.replace('nets.sc.',''):v for k,v in emo_ckpt.items() if k.startswith('nets.sc.')}, strict=False)
        self.nets.veh_e.load_state_dict({k.replace('nets.veh.',''):v for k,v in emo_ckpt.items() if k.startswith('nets.veh.')}, strict=False)
        # emotion projection 로드 (veh_e로)
        for proj in ['ppg','sc','survey','veh_e']:
            state = {
                k.split(f'projs.{proj}.',1)[1]: v
                for k, v in emo_ckpt.items() if k.startswith(f'projs.{proj}.')
            }
            if state:
                self.projs[proj].load_state_dict(state, strict=False)
        # motion weights
        mot_ckpt = torch.load('weights/best_motion.pt', map_location=device)['encoder']
        self.nets.imu.load_state_dict({k.replace('imu.',''):v for k,v in mot_ckpt.items() if k.startswith('imu.')}, strict=False)
        self.nets.veh_m.load_state_dict({k.replace('veh.',''):v for k,v in mot_ckpt.items() if k.startswith('veh.')}, strict=False)
        # motion projection 로드 (p_imu, p_veh_m 키 확인 후)
        p_imu = {k.replace('p_imu.',''):v for k,v in mot_ckpt.items() if k.startswith('p_imu.')}
        if p_imu:
            self.projs['imu'].load_state_dict(p_imu, strict=False)

        # 만약 p_veh_m prefix로 된 로딩 원한다면
        p_veh = {k.replace('p_veh_m.',''):v for k,v in mot_ckpt.items() if k.startswith('p_veh_m.')}
        if p_veh:
            self.projs['veh_m'].load_state_dict(p_veh, strict=False)

    
    def _create_optimizer(self):
        base_lr = self.cfg.MainTask.lr
        
        # 사전 학습된 인코더와 프로젝션 레이어는 작은 학습률 적용
        pre_trained_params = itertools.chain(self.nets.parameters(), self.projs.parameters())
        
        # 새로 추가된 퓨전, 예측 레이어는 기본 학습률 적용
        new_params = itertools.chain(
            self.motion_gm_fuse.parameters(), self.emotion_gm_fuse.parameters(),
            self.motion_predictor.parameters(), self.emotion_valence_predictor.parameters(),
            self.emotion_arousal_predictor.parameters()
        )

        param_groups = [
            {'params': pre_trained_params, 'lr': base_lr * 0.1}, # 1/10의 학습률
            {'params': new_params, 'lr': base_lr},
            {'params': [self.log_vars], 'lr': base_lr}
        ]

        self.optim = torch.optim.Adam(param_groups)
        print("Optimizer created with differential learning rates for adaptive fine-tuning.")

    def forward(self, batch, batch_indices=None):
        device = self.cfg.Project.device
        
        # --- 1. Base Feature Extraction ---
        imu_out = self.nets['imu'](batch['imu_emotion'].to(device), batch['imu_e_lens'].to(device))
        ppg_tcn_out = self.nets['ppg'](batch['ppg_emotion'].to(device))
        hrv = torch.cat([batch['ppg_rr_emotion'].to(device).mean(dim=1, keepdim=True), batch['ppg_rr_emotion'].to(device).std(dim=1, keepdim=True), torch.min(batch['ppg_rr_emotion'].to(device), dim=1, keepdim=True).values, torch.max(batch['ppg_rr_emotion'].to(device), dim=1, keepdim=True).values, batch['ppg_rmssd_emotion'].to(device).unsqueeze(1), batch['ppg_sdnn_emotion'].to(device).unsqueeze(1)], dim=1)
        sc_out = self.nets['sc'](batch['scenario_evt_e'].to(device), batch['scenario_type_e'].to(device), batch['phase_evt_e'].to(device), batch['scenario_time_e'].to(device))
        survey_enc = self.nets['survey'](batch['survey_e'].to(device))
        veh_all = batch['veh_emotion'].to(device)
        veh_mask = batch['veh_mask_emotion'].to(device).bool()
        veh_e = self.nets['veh_e'](veh_all.transpose(1, 2), mask=veh_mask, return_pooled=True)
        mode_idx = self.veh_cols.index([c for c in self.veh_cols if 'mode' in c][0])
        veh_m = self.nets['veh_m'](veh_all[:, :, mode_idx:mode_idx+1].transpose(1, 2), mask=veh_mask, return_pooled=True)
        
        # --- 2. Projection to Hidden Dim ---
        imu_emb = self.projs['imu'](imu_out) # (B, T, H)
        ppg_emb = self.projs['ppg'](torch.cat([ppg_tcn_out, hrv], dim=1)) # (B, H)
        sc_emb = self.projs['sc'](sc_out) # (B, H)
        survey_emb = self.projs['survey'](survey_enc) # (B, H)
        veh_emb_e = self.projs['veh_e'](veh_e) # (B, H)
        veh_emb_m = self.projs['veh_m'](veh_m) # (B, H)
        
        # --- 3. PMF Soft Freezing ---
        if self.training and batch_indices is not None:
            for m in self.pmf_modalities:
                mask = self.loss_mask[m][batch_indices].unsqueeze(1)
                if m == 'imu': imu_emb = imu_emb * mask.unsqueeze(1) # (B, T, H)
                if m == 'ppg': ppg_emb = ppg_emb * mask
                if m == 'sc': sc_emb = sc_emb * mask
                if m == 'survey': survey_emb = survey_emb * mask
                if m == 'veh_e': veh_emb_e = veh_emb_e * mask
                if m == 'veh_m': veh_emb_m = veh_emb_m * mask

        # --- 4. Bidirectional Fusion & Prediction ---
        # Motion Path
        motion_feat_dict = {'imu': imu_emb.mean(dim=1), 'veh_m': veh_emb_m, 'sc': sc_emb}
        motion_summary_feat = torch.mean(torch.stack(list(motion_feat_dict.values())), dim=0)
        fused_motion = self.motion_gm_fuse(motion_summary_feat, motion_feat_dict)

        # Emotion Path
        emotion_feat_dict = {'imu': imu_emb.mean(dim=1), 'ppg': ppg_emb, 'veh_e': veh_emb_e, 'sc': sc_emb, 'survey': survey_emb}
        emotion_summary_feat = torch.mean(torch.stack(list(emotion_feat_dict.values())), dim=0)
        fused_emotion = self.emotion_gm_fuse(emotion_summary_feat, emotion_feat_dict)

        # Motion Prediction (with Emotion Context)
        motion_input = {
            'imu': imu_emb,                  # (B, T, H) 형태로 이미 3D
            'veh': veh_emb_m.unsqueeze(1),  # (B, H) -> (B, 1, H)로 차원 추가
            'sc':  sc_emb.unsqueeze(1)      # (B, H) -> (B, 1, H)로 차원 추가
        }
        # TaskSpecificFusion 내부의 Cross-Attention은 모든 입력이 3D(B, T, D) 형태일 것으로 기대하므로,
        # 2D 텐서들에 unsqueeze(1)을 적용해줍니다.

        mot_repr, mot_logits = self.motion_predictor(motion_input, context=fused_emotion.detach(), return_feature=True)

        # Emotion Prediction (with Motion Context)
        emotion_input = {'fused': fused_emotion.unsqueeze(1)}
        valence_logits = self.emotion_valence_predictor(emotion_input, context=fused_motion.detach())
        arousal_logits = self.emotion_arousal_predictor(emotion_input, context=fused_motion.detach())

        return {
            'motion_logits': mot_logits, 'valence_logits': valence_logits, 'arousal_logits': arousal_logits,
            'fused_motion': fused_motion, 'fused_emotion': fused_emotion,
            # For PMF freezing step
            'raw_features': {'imu': imu_emb, 'ppg': ppg_emb, 'sc': sc_emb, 'survey': survey_emb, 'veh_e': veh_emb_e, 'veh_m': veh_emb_m}
        }

    
    @torch.no_grad()
    def modal_freezing(self, loader):
        """에폭 시작 시 호출, 전체 학습 데이터에 대한 관련성을 평가하여 마스크 업데이트"""
        print("\n--- Updating PMF Masks for All Training Data ---")
        # 모델을 평가 모드로 설정
        self.eval()
        
        # 각 모달리티별 임계값 업데이트
        for m in self.pmf_modalities:
            self.ths[m] = self.get_threshold(self.ths[m], m)

        # 전체 학습 데이터에 대해 피처 추출 및 관련성 점수 계산
        all_relevance_scores = {m: [] for m in self.pmf_modalities}
        
        for batch in loader:
            out = self.forward(batch)
            raw_features = out['raw_features']
            for m in self.pmf_modalities:
                feat = raw_features[m]
                # (B, T, H) -> (B, H)
                if feat.dim() == 3: feat = feat.mean(dim=1)
                # L2 Norm으로 관련성 점수 계산
                relevance = torch.norm(feat, p=2, dim=1)
                all_relevance_scores[m].append(relevance)
        
        # 계산된 점수로 loss_mask 업데이트
        for m in self.pmf_modalities:
            scores = torch.cat(all_relevance_scores[m], dim=0) # 전체 데이터셋에 대한 점수
            self.loss_mask[m] = (scores >= self.ths[m]).float()
            
            # 메모리 마스크 적용: 한번 0이 된 것은 계속 0
            self.loss_mask[m] *= self.memory_mask[m]
            self.memory_mask[m] = (self.loss_mask[m] > 0).float()
            
            frozen_ratio = 1 - self.loss_mask[m].mean().item()
            print(f"  - Modality [{m}]: Threshold={self.ths[m]:.4f}, Frozen Ratio={frozen_ratio:.2%}")
            
    def get_threshold(self, th, modal_name):
        # Config 경로 수정
        growing_rate = self.cfg.MainTask.pmf_growing_rate
        #max_theta = self.cfg.MainTask.pmf_max_theta
        th = th * growing_rate
        return th
        #return min(max_theta, th)

    def run_epoch(self, loader, train: bool):
        self.train(train)
        total_loss = 0.0
        num_batches = 0
        device = self.cfg.Project.device

        for i, batch in enumerate(loader):
            # 학습 시에는 배치의 원본 인덱스를 전달
            batch_indices = batch['indices'].to(device) if train else None
            
            with torch.set_grad_enabled(train):
                out = self.forward(batch, batch_indices)

                # --- 1) Main Task Losses ---
                mot_logits, mot_labels = out['motion_logits'], batch['label_motion'].to(device)
                B, T, C_m = mot_logits.shape
                valid_mask = (mot_labels.view(-1) > 0) & (mot_labels.view(-1) != 4)
                loss_mot = F.cross_entropy(mot_logits.view(-1, C_m)[valid_mask], (mot_labels.view(-1)[valid_mask] - 1).long()) if valid_mask.any() else torch.tensor(0., device=device)

                val_logits, raw_v = out['valence_logits'], batch['valence_reg_emotion'].to(device).reshape(-1)
                tgt_v = torch.full_like(raw_v, -100, dtype=torch.long); tgt_v[(raw_v >= 0)&(raw_v < 4)] = 0; tgt_v[(raw_v >= 4)&(raw_v < 7)] = 1; tgt_v[raw_v >= 7] = 2
                mask_v = (tgt_v >= 0) & (tgt_v < val_logits.shape[1])
                loss_v = F.cross_entropy(val_logits[mask_v], tgt_v[mask_v]) if mask_v.any() else torch.tensor(0., device=device)

                aro_logits, raw_a = out['arousal_logits'], batch['arousal_reg_emotion'].to(device).reshape(-1)
                tgt_a = torch.full_like(raw_a, -100, dtype=torch.long); tgt_a[(raw_a >= 0)&(raw_a < 4)] = 0; tgt_a[(raw_a >= 4)&(raw_a < 7)] = 1; tgt_a[raw_a >= 7] = 2
                mask_a = (tgt_a >= 0) & (tgt_a < aro_logits.shape[1])
                loss_a = F.cross_entropy(aro_logits[mask_a], tgt_a[mask_a]) if mask_a.any() else torch.tensor(0., device=device)
                
                # --- 2) Cross-modal Alignment Loss ---
                fused_motion = F.normalize(out['fused_motion'], p=2, dim=1)
                fused_emotion = F.normalize(out['fused_emotion'], p=2, dim=1)
                # Cosine Similarity 기반 Contrastive Loss
                cos_sim = (fused_motion * fused_emotion).sum(dim=1)
                loss_cross = (1 - cos_sim).mean()

                # --- 3) Total Loss with Uncertainty Weighting ---
                loss_mot_w = torch.exp(-self.log_vars[0]) * loss_mot + 0.5 * self.log_vars[0]
                loss_v_w = torch.exp(-self.log_vars[1]) * loss_v + 0.5 * self.log_vars[1]
                loss_a_w = torch.exp(-self.log_vars[2]) * loss_a + 0.5 * self.log_vars[2]
                loss_cross_w = torch.exp(-self.log_vars[3]) * loss_cross * self.cfg.MainTask.cross_modal_lambda + 0.5 * self.log_vars[3]
                loss = loss_mot_w + loss_v_w + loss_a_w + loss_cross_w

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
        self.eval()
        preds_mot, trues_mot = [], []
        preds_v, trues_v, preds_a, trues_a = [], [], [], []

        for batch in loader:
            out = self.forward(batch)
            
            # Motion Accuracy (프레임별)
            p_mot = out['motion_logits'].argmax(-1).cpu()      # (B, T)
            t_mot_raw = batch['label_motion'].cpu()            # (B, T)
            mask_mot = (t_mot_raw > 0) & (t_mot_raw != 4)
            # shift 레이블(1~→0~) 후 flatten
            preds_mot.append(p_mot[mask_mot])
            trues_mot.append((t_mot_raw[mask_mot] - 1))
            
            # Emotion Accuracy
            p_v = out['valence_logits'].argmax(-1).cpu()
            raw_v = batch['valence_reg_emotion'].reshape(-1).cpu()
            t_v = torch.full_like(raw_v, -1, dtype=torch.long)
            t_v[raw_v < 4] = 0
            t_v[(raw_v >= 4) & (raw_v < 7)] = 1
            t_v[raw_v >= 7] = 2
            m_v = t_v != -1
            preds_v.append(p_v[m_v]); trues_v.append(t_v[m_v])
            
            p_a = out['arousal_logits'].argmax(-1).cpu()
            raw_a = batch['arousal_reg_emotion'].reshape(-1).cpu()
            t_a = torch.full_like(raw_a, -1, dtype=torch.long)
            t_a[raw_a < 4] = 0
            t_a[(raw_a >= 4) & (raw_a < 7)] = 1
            t_a[raw_a >= 7] = 2
            m_a = t_a != -1
            preds_a.append(p_a[m_a]); trues_a.append(t_a[m_a])

        acc_mot = accuracy_score(torch.cat(trues_mot), torch.cat(preds_mot)) if preds_mot else 0.0
        acc_v   = accuracy_score(torch.cat(trues_v), torch.cat(preds_v))   if preds_v   else 0.0
        acc_a   = accuracy_score(torch.cat(trues_a), torch.cat(preds_a))   if preds_a   else 0.0

        if return_preds:
            # Concatenate all predictions and true labels
            final_preds = {
                'motion_preds': torch.cat(preds_mot),
                'motion_trues': torch.cat(trues_mot),
                'valence_preds': torch.cat(preds_v),
                'valence_trues': torch.cat(trues_v),
                'arousal_preds': torch.cat(preds_a),
                'arousal_trues': torch.cat(trues_a),}
            return final_preds
        else:
            return acc_mot, acc_v, acc_a
            
    def fusion_train(self):
        # 데이터 로더에 include_indices=True 추가 필요
        tr_loader = make_multitask_loader(self.cfg, self.train_keys, shuffle=True, dp=self, include_indices=True)
        va_loader = make_multitask_loader(self.cfg, self.val_keys, shuffle=False, dp=self)
        
        self.setup_pmf_masks(len(tr_loader.dataset))
        
        best_loss, patience_counter, best_performance = float('inf'), 0, {}
        
        for epoch in range(1, self.cfg.MainTask.epochs + 1):
            # Config 경로 수정
            if epoch >= self.cfg.MainTask.pmf_start_epoch:
                self.modal_freezing(tr_loader)

            tr_loss = self.run_epoch(tr_loader, train=True)
            va_loss = self.run_epoch(va_loader, train=False)
            
            va_acc_mot, va_acc_v, va_acc_a = self.evaluate(va_loader)
            print(f"Epoch {epoch:02d} | L_tr {tr_loss:.4f}  L_val {va_loss:.4f} | Acc(M/V/A): {va_acc_mot:.3f}/{va_acc_v:.3f}/{va_acc_a:.3f}")

            if va_loss < best_loss:
                best_loss, patience_counter = va_loss, 0
                best_performance = {'mot_acc': va_acc_mot, 'val_acc': va_acc_v, 'aro_acc': va_acc_a, 'best_loss': best_loss}
                
                torch.save({
                    'model_state_dict': self.state_dict(), 
                    'optimizer_state_dict': self.optim.state_dict(),
                    'best_performance': best_performance
                }, "weights/best_fusion_v10.pt")
            else:
                patience_counter += 1
                # Config 경로 수정
                if patience_counter >= self.cfg.MainTask.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        return best_performance