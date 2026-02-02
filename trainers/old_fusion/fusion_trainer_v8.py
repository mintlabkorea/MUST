# V8: v7 load_pretrain_weights 수정 버전, fold
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from sklearn.metrics import accuracy_score
from config.config import Config

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
        self._freeze_base_modules()
        self._create_optimizer()

    def _build_model(self):
        # save original vehicle settings
        original_veh_dim = self.cfg.Encoders.veh['input_dim']
        original_veh_embed = self.cfg.Encoders.veh['embed_dim']

        # emotion vehicle encoder
        veh_em_p = self.cfg.Encoders.veh
        veh_encoder_emotion = VehicleTCNEncoder(veh_em_p)

        # motion vehicle encoder: use only mode channel and fixed embed
        self.cfg.Encoders.veh['input_dim'] = 1
        self.cfg.Encoders.veh['embed_dim'] = 128
        veh_encoder_motion = VehicleTCNEncoder(self.cfg.Encoders.veh)

        motion_veh_embed_dim = self.cfg.Encoders.veh['embed_dim']

        # restore vehicle config
        self.cfg.Encoders.veh['input_dim'] = original_veh_dim
        self.cfg.Encoders.veh['embed_dim'] = original_veh_embed

        imu_p = self.cfg.Encoders.imu
        ppg_p = self.cfg.Encoders.ppg
        sc_p  = self.cfg.Encoders.sc

        self.nets = nn.ModuleDict({
            'imu': IMUFeatureEncoder(imu_p),
            'ppg': PPGEncoder(ppg_p),
            'sc' : ScenarioEmbedding(sc_p),
            'veh_e': veh_encoder_emotion,
            'veh_m': veh_encoder_motion,
            'survey': PreSurveyEncoder(self.cfg.Encoders.survey),
        })

        hidden_dim = self.cfg.FusionModel.hidden_dim
        ppg_in_dim = self.cfg.Encoders.ppg['embed_dim'] + 6

        self.projs = nn.ModuleDict({
            'imu':    nn.Linear(self.cfg.Encoders.imu['encoder_dim'], hidden_dim),
            'ppg':    nn.Linear(ppg_in_dim,                         hidden_dim),
            'sc':     nn.Linear(self.cfg.Encoders.sc['embed_dim'],  hidden_dim),
            # survey는 “원본 입력 차원(input_dim=12)”을 써야 합니다.
            'survey': nn.Linear(self.cfg.Encoders.survey['embed_dim'], hidden_dim),
            'veh_e':  nn.Linear(original_veh_embed,                 hidden_dim),
            'veh_m':  nn.Linear(motion_veh_embed_dim,               hidden_dim),
        })

        modalities = ['imu','ppg','veh','sc','survey']
        self.sm_fuse = SMFusionBlock(modalities=modalities, hidden_dim=hidden_dim)
        self.bg_head = nn.Linear(hidden_dim, self.cfg.FusionModel.num_behavior_groups)
        self.eg_val_head = nn.Linear(hidden_dim, self.cfg.PretrainEmotion.num_valence)
        self.eg_aro_head = nn.Linear(hidden_dim, self.cfg.PretrainEmotion.num_arousal)

        self.gm_fuse = GMFusionBlock(modalities=modalities, hidden_dim=hidden_dim, num_heads=self.cfg.FusionModel.num_heads)
        self.motion_predictor = MotionPredictor(hidden_dim, self.cfg.PretrainMotion.num_motion)
        self.emotion_valence_predictor = EmotionPredictor(hidden_dim, hidden_dim, self.cfg.PretrainEmotion.num_valence)
        self.emotion_arousal_predictor = EmotionPredictor(hidden_dim, hidden_dim, self.cfg.PretrainEmotion.num_arousal)

        init_log_vars = torch.tensor([0.22,0.0,-0.6,1.6,1.6,1.6])
        self.log_vars = nn.Parameter(init_log_vars)
        self.scaler = torch.cuda.amp.GradScaler()

        self.ce_mot = nn.CrossEntropyLoss(ignore_index=self.cfg.PretrainMotion.ignore_index)
        self.ce_v = nn.CrossEntropyLoss(ignore_index=self.cfg.PretrainEmotion.ignore_index)
        self.ce_a = nn.CrossEntropyLoss(ignore_index=self.cfg.PretrainEmotion.ignore_index)
        self.ce_aux = nn.CrossEntropyLoss()

        self.to(self.cfg.Project.device)

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


    # 1. _freeze_base_modules가 클래스 메소드가 되도록 올바른 위치로 이동
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
            self.emotion_valence_predictor.parameters(),  # valence 예측기의 파라미터 추가
            self.emotion_arousal_predictor.parameters(),  # arousal 예측기의 파라미터 추가
            [self.log_vars]
        )
        self.optim = torch.optim.Adam(trainable_params, lr=self.cfg.MainTask.lr)

    def forward(self, batch):
        device = self.cfg.Project.device

        # --- Base Feature Extraction ---
        imu_out = self.nets['imu'](
            batch['imu_emotion'].to(device),
            batch['imu_e_lens'].to(device)
        )  # (B, T, encoder_dim)
        imu_emb = self.projs['imu'](imu_out.mean(dim=1))  # (B, hidden_dim)

        ppg_tcn_out = self.nets['ppg'](batch['ppg_emotion'].to(device))  # (B, embed_dim)
        hrv = torch.cat([
            batch['ppg_rr_emotion'].to(device).mean(dim=1, keepdim=True),
            batch['ppg_rr_emotion'].to(device).std(dim=1, keepdim=True),
            torch.min(batch['ppg_rr_emotion'].to(device), dim=1, keepdim=True).values,
            torch.max(batch['ppg_rr_emotion'].to(device), dim=1, keepdim=True).values,
            batch['ppg_rmssd_emotion'].to(device).unsqueeze(1),
            batch['ppg_sdnn_emotion'].to(device).unsqueeze(1)
        ], dim=1)  # (B, 6)
        ppg_emb = self.projs['ppg'](torch.cat([ppg_tcn_out, hrv], dim=1))  # (B, hidden_dim)

        sc_out = self.nets['sc'](
            batch['scenario_evt_e'].to(device),
            batch['scenario_type_e'].to(device),
            batch['phase_evt_e'].to(device),
            batch['scenario_time_e'].to(device)
        )  # (B, embed_dim)
        sc_emb = self.projs['sc'](sc_out)  # (B, hidden_dim)

        raw_survey   = batch['survey_e'].to(device)    # (B, 12)
        survey_enc   = self.nets['survey'](raw_survey) # (B,  8)
        survey_emb   = self.projs['survey'](survey_enc) # (B, hidden_dim)

        veh_all = batch['veh_emotion'].to(device)              # (B, C, T)
        veh_mask = batch['veh_mask_emotion'].to(device).bool()
        veh_e = self.nets['veh_e'](
            veh_all.transpose(1, 2), mask=veh_mask, return_pooled=True
        )  # (B, embed_dim)
        veh_emb_e = self.projs['veh_e'](veh_e)  # (B, hidden_dim)

        mode_idx = self.veh_cols.index([c for c in self.veh_cols if 'mode' in c][0])
        veh_m = self.nets['veh_m'](
            veh_all[:, :, mode_idx:mode_idx+1].transpose(1, 2),
            mask=veh_mask, return_pooled=True
        )  # (B, embed_dim)
        veh_emb_m = self.projs['veh_m'](veh_m)  # (B, hidden_dim)

        # --- Emotion Path Features & Aux Path ---
        features_dict_emo = {
            'imu': imu_emb, 'ppg': ppg_emb,
            'veh': veh_emb_e, 'sc': sc_emb,
            'survey': survey_emb
        }
        X_aux = self.sm_fuse(features_dict_emo)
        bg_logits    = self.bg_head(X_aux)
        eg_val_logits = self.eg_val_head(X_aux)
        eg_aro_logits = self.eg_aro_head(X_aux)

        # --- Intermediate Fusion & prepare for Emotion Predictor ---
        fused_global = self.gm_fuse(X_aux, features_dict_emo)
        # fused_global may be a dict {'fused': tensor} or a tensor
        if isinstance(fused_global, dict):
            feat = fused_global['fused']
        else:
            feat = fused_global
        # ensure time dimension: (B, D) -> (B, 1, D)
        if feat.dim() == 2:
            feat = feat.unsqueeze(1)
        emotion_input_dict = {'fused': feat}  # (B, 1, hidden_dim)

        # --- Motion Path ---
        motion_input = {
            'imu': imu_emb.unsqueeze(1),
            'veh': veh_emb_m.unsqueeze(1),
            'sc':  sc_emb.unsqueeze(1)
        }
        mot_repr, mot_logits = self.motion_predictor(motion_input, return_feature=True)

        # --- Emotion Predictor Path ---
        emotion_features = emotion_input_dict['fused']         # (B, 1, hidden_dim)
        emotion_context  = mot_repr.detach().squeeze(1)        # (B, hidden_dim)

        valence_logits = self.emotion_valence_predictor(
            emotion_features, context=emotion_context
        )  # (B, num_valence)
        arousal_logits = self.emotion_arousal_predictor(
            emotion_features, context=emotion_context
        )  # (B, num_arousal)

        return {
            'motion_logits': mot_logits.squeeze(1),
            'valence_logits': valence_logits,
            'arousal_logits': arousal_logits,
            'bg_logits': bg_logits,
            'eg_val_logits': eg_val_logits,
            'eg_aro_logits': eg_aro_logits
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
                tgt_v[raw_v < 4] = 0
                tgt_v[(raw_v >= 4) & (raw_v < 7)] = 1
                tgt_v[raw_v >= 7] = 0
                mask_v = (tgt_v != -1); loss_v = self.ce_v(out['valence_logits'][mask_v], tgt_v[mask_v].to(self.cfg.device)) if mask_v.any() else torch.tensor(0., device=self.cfg.device)
                
                raw_a = batch['arousal_reg_emotion'].reshape(-1); tgt_a = torch.full_like(raw_a, -1, dtype=torch.long)
                tgt_a[raw_v < 4] = 0
                tgt_a[(raw_v >= 4) & (raw_v < 7)] = 1
                tgt_a[raw_v >= 7] = 0
                mask_a = (tgt_a != -1); loss_a = self.ce_a(out['arousal_logits'][mask_a], tgt_a[mask_a].to(self.cfg.device)) if mask_a.any() else torch.tensor(0., device=self.cfg.device)
                
                # 보조 손실 (실제 그룹 레이블 키로 변경 필요: 예: batch['behavior_group'])
                tgt_bg = batch.get('behavior_group', torch.randint(0, self.cfg.FusionModel.num_behavior_groups, (out['bg_logits'].size(0),))).to(self.cfg.device).long()
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
    def evaluate(self, loader, return_preds=False):
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
            t_v[raw_v < 4] = 0
            t_v[(raw_v >= 4) & (raw_v < 7)] = 1
            t_v[raw_v >= 7] = 0
            m_v = t_v != -1
            preds_v.append(p_v[m_v]); trues_v.append(t_v[m_v])
            
            p_a = out['arousal_logits'].argmax(-1).cpu()
            raw_a = batch['arousal_reg_emotion'].reshape(-1).cpu()
            t_a = torch.full_like(raw_a, -1, dtype=torch.long)
            t_a[raw_a < 4] = 0
            t_a[(raw_a >= 4) & (raw_a < 7)] = 1
            t_a[raw_a >= 7] = 0
            m_a = t_a != -1
            preds_a.append(p_a[m_a]); trues_a.append(t_a[m_a])

        acc_mot = accuracy_score(torch.cat(trues_mot), torch.cat(preds_mot)) if len(trues_mot) > 0 else 0.0
        acc_v = accuracy_score(torch.cat(trues_v), torch.cat(preds_v)) if len(trues_v) > 0 else 0.0
        acc_a = accuracy_score(torch.cat(trues_a), torch.cat(preds_a)) if len(trues_a) > 0 else 0.0
        
        if return_preds:
            # Concatenate all predictions and true labels
            final_preds = {
                'motion_preds': torch.cat(preds_mot) if len(preds_mot) > 0 else torch.empty(0, dtype=torch.long),
                'motion_trues': torch.cat(trues_mot) if len(trues_mot) > 0 else torch.empty(0, dtype=torch.long),
                'valence_preds': torch.cat(preds_v) if len(preds_v) > 0 else torch.empty(0, dtype=torch.long),
                'valence_trues': torch.cat(trues_v) if len(trues_v) > 0 else torch.empty(0, dtype=torch.long),
                'arousal_preds': torch.cat(preds_a) if len(preds_a) > 0 else torch.empty(0, dtype=torch.long),
                'arousal_trues': torch.cat(trues_a) if len(trues_a) > 0 else torch.empty(0, dtype=torch.long),
            }
            return final_preds
        else:
            return acc_mot, acc_v, acc_a
            
    def fusion_train(self, fold_num):
        tr_loader = make_multitask_loader(self.cfg, self.train_keys, shuffle=True, dp=self)
        va_loader = make_multitask_loader(self.cfg, self.val_keys, shuffle=False, dp=self)
        
        best_loss = float('inf')
        patience_counter = 0 # 변수 이름 변경 (patience와 혼동 방지)
        best_performance = {}
        
        # epochs, patience 참조 수정
        for epoch in range(1, self.cfg.MainTask.epochs + 1):
            tr_loss = self.run_epoch(tr_loader, train=True)
            va_loss = self.run_epoch(va_loader, train=False)
            
            va_acc_mot, va_acc_v, va_acc_a = self.evaluate(va_loader)
            print(f"Epoch {epoch:02d} | L_tr {tr_loss:.4f}  L_val {va_loss:.4f} | Acc(M/V/A): {va_acc_mot:.3f}/{va_acc_v:.3f}/{va_acc_a:.3f}")

            if va_loss < best_loss:
                best_loss, patience_counter = va_loss, 0
                best_performance = {'mot_acc': va_acc_mot, 'val_acc': va_acc_v, 'aro_acc': va_acc_a, 'best_loss': best_loss}
                
                save_path = f"weights/best_fusion_fold_{fold_num}.pt"
                torch.save({
                    'model_state_dict': self.state_dict(), 
                    'optimizer_state_dict': self.optim.state_dict(),
                    'best_performance': best_performance
                }, save_path)
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.MainTask.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        return best_performance