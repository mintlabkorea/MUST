# v5의 감정 +v6의 모션

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from sklearn.metrics import accuracy_score

from trainers.base_trainer import dataProcessor
from data.loader import make_multitask_loader # 멀티태스크 로더를 그대로 사용
from models.encoder.imu_encoder import IMUFeatureEncoder
from models.encoder.ppg_TCN_encoder import PPGEncoder
from models.encoder.veh_encoder import VehicleTCNEncoder
from models.encoder.sc_encoder import ScenarioEmbedding
from models.fusion.fusion_block import SMFusionBlock, GMFusionBlock
from models.fusion.predictors import MotionPredictor, EmotionPredictor

class FusionTrainer(nn.Module, dataProcessor):
    def __init__(self, cfg, train_keys, val_keys, test_keys):
        super().__init__()
        # dataProcessor의 기본 정보(veh_cols 등)만 생성
        dataProcessor.__init__(self, cfg) 
        self.prepare() # prepare()는 veh_cols 등을 만드므로 유지

        # main.py에서 할당해 준 key로 덮어쓰기
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.test_keys = test_keys # 최종 테스트를 위해 test_keys도 받아둠
        self.cfg = cfg

        # 나머지 빌드 및 학습 준비 과정은 동일
        self._build_model()
        self._load_pretrained_weights()
        self._freeze_base_modules()
        self._create_optimizer()
        
    def _build_model(self):
        # Vehicle Encoder 분리 생성
        veh_encoder_emotion = VehicleTCNEncoder(self.cfg)
        original_veh_dim = self.cfg.veh_params['input_dim']
        self.cfg.veh_params['input_dim'] = 1
        veh_encoder_motion = VehicleTCNEncoder(self.cfg)
        self.cfg.veh_params['input_dim'] = original_veh_dim

        self.nets = nn.ModuleDict({
            'imu': IMUFeatureEncoder(self.cfg), 'ppg': PPGEncoder(self.cfg),
            'sc' : ScenarioEmbedding(self.cfg),
            'veh_e': veh_encoder_emotion, 'veh_m': veh_encoder_motion,
        })
        
        ppg_proj_in_dim = self.cfg.ppg_params['embed_dim'] + 6
        self.projs = nn.ModuleDict({
            'imu': nn.Linear(self.cfg.imu_params['encoder_dim'], self.cfg.hidden),
            'ppg': nn.Linear(ppg_proj_in_dim, self.cfg.hidden), 'sc': nn.Linear(self.cfg.sc_params['embed_dim'], self.cfg.hidden),
            'survey': nn.Linear(self.cfg.survey_dim, self.cfg.hidden),
            'veh_e': nn.Linear(self.cfg.veh_params['embed_dim'], self.cfg.hidden),
            'veh_m': nn.Linear(self.cfg.veh_params['embed_dim'], self.cfg.hidden)
        })

        # --- V5 아키텍처로 통일 ---
        modalities = ['imu', 'ppg', 'veh', 'sc', 'survey']
        self.sm_fuse = SMFusionBlock(modalities=modalities, hidden_dim=self.cfg.hidden)
        self.bg_head = nn.Linear(self.cfg.hidden, self.cfg.num_behavior_groups)
        self.eg_val_head = nn.Linear(self.cfg.hidden, self.cfg.num_valence)
        self.eg_aro_head = nn.Linear(self.cfg.hidden, self.cfg.num_arousal)
        self.gm_fuse = GMFusionBlock(modalities=modalities, hidden_dim=self.cfg.hidden, num_heads=4)
        self.motion_predictor = MotionPredictor(feature_dim=self.cfg.hidden, output_dim=self.cfg.num_motion)
        self.emotion_predictor = EmotionPredictor(feature_dim=self.cfg.hidden, num_valence=self.cfg.num_valence, num_arousal=self.cfg.num_arousal)

        # Arousal > Valence > Motion 순으로 가중치 부여
        initial_biases = torch.tensor([
            0.22,    # Motion: 가중치 ~0.8 (상대적으로 낮게)
            0.0,     # Valence: 가중치 1.0 (기준)
            -0.6,    # Arousal: 가중치 ~1.8 (상대적으로 높게)
            1.6,     # Aux - bg: 가중치 ~0.2 (낮게 유지)
            1.6,     # Aux - eg_v: 가중치 ~0.2 (낮게 유지)
            1.6      # Aux - eg_a: 가중치 ~0.2 (낮게 유지)
        ])
        self.log_vars = nn.Parameter(initial_biases)
        self.scaler = torch.cuda.amp.GradScaler()
        self.ce_mot = nn.CrossEntropyLoss(ignore_index=self.cfg.ign_mot)
        self.ce_v = nn.CrossEntropyLoss(ignore_index=self.cfg.ign_emo)
        self.ce_a = nn.CrossEntropyLoss(ignore_index=self.cfg.ign_emo)
        self.ce_aux = nn.CrossEntropyLoss()
        self.to(self.cfg.device)

    def _load_pretrained_weights(self):
        print("▶ Loading pre-trained weights...")
        
        # --- 1. 감정 전문가 가중치 로드 ---
        emotion_ckpt = torch.load("weights/best_emotion.pt", map_location=self.cfg.device)
        emo_state_dict = emotion_ckpt['encoder']

        # 각 모듈별로 정확한 접두사를 제거하여 state_dict 로드
        self.nets.ppg.load_state_dict({k.replace('nets.ppg.', '', 1): v for k, v in emo_state_dict.items() if k.startswith('nets.ppg.')}, strict=False)
        self.nets.sc.load_state_dict({k.replace('nets.sc.', '', 1): v for k, v in emo_state_dict.items() if k.startswith('nets.sc.')}, strict=False)
        self.nets.veh_e.load_state_dict({k.replace('nets.veh.', '', 1): v for k, v in emo_state_dict.items() if k.startswith('nets.veh.')}, strict=False)
        
        self.projs.ppg.load_state_dict({k.replace('projs.ppg.', '', 1): v for k, v in emo_state_dict.items() if k.startswith('projs.ppg.')}, strict=False)
        self.projs.sc.load_state_dict({k.replace('projs.sc.', '', 1): v for k, v in emo_state_dict.items() if k.startswith('projs.sc.')}, strict=False)
        self.projs.survey.load_state_dict({k.replace('projs.survey.', '', 1): v for k, v in emo_state_dict.items() if k.startswith('projs.survey.')}, strict=False)
        self.projs.veh_e.load_state_dict({k.replace('projs.veh.', '', 1): v for k, v in emo_state_dict.items() if k.startswith('projs.veh.')}, strict=False)
        print("Weights from 'best_emotion.pt' loaded.")

        # --- 2. 모션 전문가 가중치 로드 (IMU, VEH 덮어쓰기) ---
        motion_ckpt = torch.load("weights/best_motion.pt", map_location=self.cfg.device)
        mot_state_dict = motion_ckpt['encoder']
        
        self.nets.imu.load_state_dict({k.replace('imu.', '', 1): v for k, v in mot_state_dict.items() if k.startswith('imu.')}, strict=False)
        self.nets.veh_m.load_state_dict({k.replace('veh.', '', 1): v for k, v in mot_state_dict.items() if k.startswith('veh.')}, strict=False)
        self.projs.imu.load_state_dict({k.replace('p_imu.', '', 1): v for k, v in mot_state_dict.items() if k.startswith('p_imu.')}, strict=False)
        self.projs.veh_m.load_state_dict({k.replace('p_veh.', '', 1): v for k, v in mot_state_dict.items() if k.startswith('p_veh.')}, strict=False)
        print("Overwrote IMU/VEH modules with weights from 'best_motion.pt'.")

    # _load_pretrained_weights 메소드 밖으로 이동하여 클래스 메소드로 정의
    def _freeze_base_modules(self):
        print("▶ Freezing base encoders (nets) and projections (projs)...")
        for param in self.nets.parameters(): param.requires_grad = False
        for param in self.projs.parameters(): param.requires_grad = False
        print("Base modules are frozen.")

    def _create_optimizer(self):
        trainable_params = itertools.chain(
            self.sm_fuse.parameters(), self.bg_head.parameters(),
            self.eg_val_head.parameters(), self.eg_aro_head.parameters(),
            self.gm_fuse.parameters(), self.motion_predictor.parameters(),
            self.emotion_predictor.parameters(), [self.log_vars]
        )
        self.optim = torch.optim.Adam(trainable_params, lr=self.cfg.lr)

    def forward(self, batch):
        # Base Feature Extraction
        imu_emb = self.projs['imu'](self.nets['imu'](batch['imu_emotion'].to(self.cfg.device), batch['imu_e_lens'].to(self.cfg.device)).mean(dim=1))
    
        ppg_tcn_out = self.nets['ppg'](batch['ppg_emotion'].to(self.cfg.device))
        hrv_features = torch.cat([
                batch['ppg_rr_emotion'].to(self.cfg.device).mean(dim=1, keepdim=True),
                batch['ppg_rr_emotion'].to(self.cfg.device).std(dim=1, keepdim=True),
                torch.min(batch['ppg_rr_emotion'].to(self.cfg.device), dim=1, keepdim=True).values,
                torch.max(batch['ppg_rr_emotion'].to(self.cfg.device), dim=1, keepdim=True).values,
                batch['ppg_rmssd_emotion'].to(self.cfg.device).unsqueeze(1),
                batch['ppg_sdnn_emotion'].to(self.cfg.device).unsqueeze(1)
            ], dim=1)
        ppg_emb = self.projs['ppg'](torch.cat([ppg_tcn_out, hrv_features], dim=1))
        
        sc_emb = self.projs['sc'](self.nets['sc'](
            batch['scenario_evt_e'].to(self.cfg.device),
            batch['scenario_type_e'].to(self.cfg.device),
            batch['phase_evt_e'].to(self.cfg.device),
            batch['scenario_time_e'].to(self.cfg.device)
        ))
        survey_emb = self.projs['survey'](batch['survey_e'].to(self.cfg.device))

        # Vehicle Embedding 
        veh_all_channels = batch['veh_emotion'].to(self.cfg.device)
        veh_mask = batch['veh_mask_emotion'].to(self.cfg.device).bool()
        veh_emb_e = self.projs['veh_e'](self.nets['veh_e'](veh_all_channels.transpose(1,2), mask=veh_mask, return_pooled=True))
        
        mode_idx = self.veh_cols.index([c for c in self.veh_cols if 'mode' in c][0])
        veh_one_channel = veh_all_channels[:, :, mode_idx:mode_idx+1]
        veh_emb_m = self.projs['veh_m'](self.nets['veh_m'](veh_one_channel.transpose(1,2), mask=veh_mask, return_pooled=True))
        
        # Emotion Path Features
        features_dict_emo = {'imu': imu_emb, 'ppg': ppg_emb, 'veh': veh_emb_e, 'sc': sc_emb, 'survey': survey_emb}
        
        # Aux Path
        X_aux = self.sm_fuse(features_dict_emo)
        bg_logits, eg_val_logits, eg_aro_logits = self.bg_head(X_aux), self.eg_val_head(X_aux), self.eg_aro_head(X_aux)
        
        # Intermediate Fusion
        fused_global = self.gm_fuse(X_aux, features_dict_emo)
        
        # Motion Path
        motion_input_dict = {'imu': imu_emb.unsqueeze(1), 'veh': veh_emb_m.unsqueeze(1), 'sc': sc_emb.unsqueeze(1)}
        mot_repr, mot_logits = self.motion_predictor(motion_input_dict, return_feature=True)
        
        # Emotion Predictor Path
        emotion_input_dict = {k: v.unsqueeze(1) for k, v in features_dict_emo.items()}
        emotion_input_dict['fused'] = fused_global.unsqueeze(1)
        emo_dict = self.emotion_predictor(emotion_input_dict, context=mot_repr.detach())
        
        return {
            'motion_logits': mot_logits.squeeze(1), 'valence_logits': emo_dict['valence_logits'],
            'arousal_logits': emo_dict['arousal_logits'], 'bg_logits': bg_logits,
            'eg_val_logits': eg_val_logits, 'eg_aro_logits': eg_aro_logits
        }
    
    def run_epoch(self, loader, train: bool):
        self.train(train)
        total_loss = 0
        
        for batch in loader:
            with torch.set_grad_enabled(train):
                out = self.forward(batch)
                
                # --- 손실 계산 ---
                # Motion (윈도우의 첫 프레임 레이블을 대표로 사용)
                tgt_mot_raw = batch["label_motion"][:, 0].long().to(self.cfg.device)
                mask_mot = (tgt_mot_raw > 0); tgt_mot = tgt_mot_raw[mask_mot] - 1
                loss_mot = self.ce_mot(out['motion_logits'][mask_mot], tgt_mot) if mask_mot.any() else torch.tensor(0., device=self.cfg.device)

                # Emotion
                raw_v = batch['valence_reg_emotion'].reshape(-1); tgt_v = torch.full_like(raw_v, -1, dtype=torch.long)
                tgt_v[raw_v < 4] = 0; tgt_v[(raw_v >= 4) & (raw_v < 7)] = 1; tgt_v[raw_v >= 7] = 2
                mask_v = (tgt_v != -1); loss_v = self.ce_v(out['valence_logits'][mask_v], tgt_v[mask_v].to(self.cfg.device)) if mask_v.any() else torch.tensor(0., device=self.cfg.device)
                
                raw_a = batch['arousal_reg_emotion'].reshape(-1); tgt_a = torch.full_like(raw_a, -1, dtype=torch.long)
                tgt_a[raw_a < 4] = 0; tgt_a[(raw_a >= 4) & (raw_a < 7)] = 1; tgt_a[raw_a >= 7] = 2
                mask_a = (tgt_a != -1); loss_a = self.ce_a(out['arousal_logits'][mask_a], tgt_a[mask_a].to(self.cfg.device)) if mask_a.any() else torch.tensor(0., device=self.cfg.device)
                
                # 보조 손실 (실제 그룹 레이블 키로 변경 필요: 예: batch['behavior_group'])
                tgt_bg = batch.get('behavior_group', torch.randint(0, self.cfg.num_behavior_groups, (out['bg_logits'].size(0),))).to(self.cfg.device).long()
                loss_bg = self.ce_aux(out['bg_logits'], tgt_bg)
                
                loss_eg_v = self.ce_aux(out['eg_val_logits'][mask_v], tgt_v[mask_v].to(self.cfg.device)) if mask_v.any() else torch.tensor(0., device=self.cfg.device)
                loss_eg_a = self.ce_aux(out['eg_aro_logits'][mask_a], tgt_a[mask_a].to(self.cfg.device)) if mask_a.any() else torch.tensor(0., device=self.cfg.device)

                # --- 최종 손실 (학습 가능한 가중치 적용) ---
                loss_mot_w = torch.exp(-self.log_vars[0]) * loss_mot + 0.5 * self.log_vars[0]
                loss_v_w = torch.exp(-self.log_vars[1]) * loss_v + 0.5 * self.log_vars[1]
                loss_a_w = torch.exp(-self.log_vars[2]) * loss_a + 0.5 * self.log_vars[2]
                loss_bg_w = torch.exp(-self.log_vars[3]) * loss_bg + 0.5 * self.log_vars[3]
                loss_eg_v_w = torch.exp(-self.log_vars[4]) * loss_eg_v + 0.5 * self.log_vars[4]
                loss_eg_a_w = torch.exp(-self.log_vars[5]) * loss_eg_a + 0.5 * self.log_vars[5]
                loss = loss_mot_w + loss_v_w + loss_a_w + loss_bg_w + loss_eg_v_w + loss_eg_a_w

            if train:
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()

            total_loss += loss.item() * out['motion_logits'].size(0)
            
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader):
        self.eval() # 전체 모델을 평가 모드로
        
        preds_mot, trues_mot = [], [] 
        preds_v, trues_v, preds_a, trues_a = [], [], [], []

        for batch in loader:
            out = self.forward(batch)
            
            # Motion Accuracy
            p_mot = out['motion_logits'].argmax(-1).cpu()
            t_mot_raw = batch["label_motion"][:, 0].cpu()
            m_mot = (t_mot_raw > 0)
            t_mot = t_mot_raw[m_mot] - 1
            preds_mot.append(p_mot[m_mot]); trues_mot.append(t_mot)
            
            # Emotion Accuracy
            p_v = out['valence_logits'].argmax(-1).cpu()
            raw_v = batch['valence_reg_emotion'].reshape(-1).cpu()
            t_v = torch.full_like(raw_v, -1, dtype=torch.long)
            t_v[raw_v < 4] = 0; t_v[(raw_v >= 4) & (raw_v < 7)] = 1; t_v[raw_v >= 7] = 2
            m_v = t_v != -1
            preds_v.append(p_v[m_v]); trues_v.append(t_v[m_v])
            
            p_a = out['arousal_logits'].argmax(-1).cpu()
            raw_a = batch['arousal_reg_emotion'].reshape(-1).cpu()
            t_a = torch.full_like(raw_a, -1, dtype=torch.long)
            t_a[raw_a < 4] = 0; t_a[(raw_a >= 4) & (raw_a < 7)] = 1; t_a[raw_a >= 7] = 2
            m_a = t_a != -1
            preds_a.append(p_a[m_a]); trues_a.append(t_a[m_a])

        acc_mot = accuracy_score(torch.cat(trues_mot), torch.cat(preds_mot)) if len(trues_mot) > 0 else 0.0
        acc_v = accuracy_score(torch.cat(trues_v), torch.cat(preds_v)) if len(trues_v) > 0 else 0.0
        acc_a = accuracy_score(torch.cat(trues_a), torch.cat(preds_a)) if len(trues_a) > 0 else 0.0
        
        return acc_mot, acc_v, acc_a
    
    def fusion_train(self, fold_num):
        tr_loader = make_multitask_loader(self.cfg, self.train_keys, shuffle=True, dp=self)
        va_loader = make_multitask_loader(self.cfg, self.val_keys, shuffle=False, dp=self)
        
        best_loss = float('inf')
        patience = 0
        best_performance = {}

        for epoch in range(1, self.cfg.epochs + 1):
            tr_loss = self.run_epoch(tr_loader, train=True)
            va_loss = self.run_epoch(va_loader, train=False)
            
            va_acc_mot, va_acc_v, va_acc_a = self.evaluate(va_loader)
            print(f"Epoch {epoch:02d} | L_tr {tr_loss:.4f}  L_val {va_loss:.4f} | Acc(M/V/A): {va_acc_mot:.3f}/{va_acc_v:.3f}/{va_acc_a:.3f}")

            if va_loss < best_loss:
                best_loss, patience = va_loss, 0
                best_performance = {'mot_acc': va_acc_mot, 'val_acc': va_acc_v, 'aro_acc': va_acc_a, 'best_loss': best_loss}
                
                save_path = f"weights/best_fusion_fold_{fold_num}.pt"
                torch.save({
                    'model_state_dict': self.state_dict(), 
                    'optimizer_state_dict': self.optim.state_dict(),
                    'best_performance': best_performance
                }, save_path)
            else:
                patience += 1
                if patience >= self.cfg.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # 해당 fold의 최고 성능 기록을 반환
        return best_performance