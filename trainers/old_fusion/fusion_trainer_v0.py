# GMFusion: 비대칭 없이 Gating만 써서, motion그대로, valence 떨어짐, arousal 올라감
import torch
import torch.nn as nn
import itertools
from sklearn.metrics import accuracy_score

from trainers.base_trainer import TrainerBase, dataProcessor
from data.loader import make_multitask_loader
from models.motion_encoder import MotionEncoder
from models.emotion_encoder import EmotionEncoder
from models.head.motion_head import MotionHead
from models.head.emotion_head import EmotionHead
from models.fusion.mumu_fusion import GMFusionBlock

class FusionTrainer(nn.Module, dataProcessor):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        dataProcessor.__init__(self, cfg)
        self.prepare() 
        self.cfg = cfg
        
        self._build_model() 
        self._load_pretrained_weights()
        self._freeze_encoders()
        self._create_optimizer()

    def _build_model(self):
        # MotionEncoder 생성 시 veh_dim=1 설정 (mode)
        original_veh_dim = self.cfg.veh_params['input_dim']
        self.cfg.veh_params['input_dim'] = 1
        self.motion_encoder = MotionEncoder(self.cfg)
        self.cfg.veh_params['input_dim'] = original_veh_dim
        
        self.emotion_encoder = EmotionEncoder(self.cfg)
        
        # Fusion 블록 및 Head 초기화
        self.fusion_block = GMFusionBlock(
            input_dims={'motion': self.cfg.hidden, 'emotion': self.cfg.hidden},
            hidden_dim=self.cfg.hidden * 2,
            output_dim=self.cfg.hidden
        )
        self.motion_head = MotionHead(self.cfg.hidden, self.cfg.num_motion)
        self.emotion_head = EmotionHead(self.cfg)  

        # 손실 함수 및 스케일러 정의
        self.scaler = torch.cuda.amp.GradScaler()
        self.ce_mot = torch.nn.CrossEntropyLoss(ignore_index=self.cfg.ign_mot)
        self.ce_v = torch.nn.CrossEntropyLoss(ignore_index=self.cfg.ign_emo)
        self.ce_a = torch.nn.CrossEntropyLoss(ignore_index=self.cfg.ign_emo)

        # 모델들을 device로 이동
        self.motion_encoder.to(self.cfg.device)
        self.emotion_encoder.to(self.cfg.device)
        self.fusion_block.to(self.cfg.device)
        self.motion_head.to(self.cfg.device)
        self.emotion_head.to(self.cfg.device)

    def _load_pretrained_weights(self):
        # Motion 가중치 로드
        motion_ckpt = torch.load("weights/best_motion.pt", map_location=self.cfg.device)
        self.motion_encoder.load_state_dict(motion_ckpt['encoder'])

        # Emotion 가중치 로드
        emotion_ckpt = torch.load("weights/best_emotion.pt", map_location=self.cfg.device)
        self.emotion_encoder.load_state_dict(emotion_ckpt['encoder'])

    def _freeze_encoders(self):
        print("Freezing motion and emotion encoders...")
        # Motion Encoder 동결
        for param in self.motion_encoder.parameters():
            param.requires_grad = False 

        # Emotion Encoder 동결
        for param in self.emotion_encoder.parameters():
            param.requires_grad = False

    def _create_optimizer(self):
        # 각 모듈의 파라미터를 넘기기
        trainable_params = itertools.chain(
            self.fusion_block.parameters(),
            self.motion_head.parameters(),
            self.emotion_head.parameters()
        )
        self.optim = torch.optim.Adam(trainable_params, lr=self.cfg.lr)

    def forward(self, batch):
        # 입력 텐서들을 모델과 동일한 device로 이동
        imu_e = batch["imu_emotion"].to(self.cfg.device)
        imu_e_lens = batch["imu_e_lens"].to(self.cfg.device)
        
        # --- 여기가 핵심 수정 부분 ---
        # 1. 먼저 12채널 전체 vehicle 데이터를 불러옵니다.
        veh_e_all_channels = batch["veh_emotion"].to(self.cfg.device)
        
        # 2. 'mode' 컬럼의 인덱스를 찾습니다.
        # self.veh_cols는 dataProcessor에서 생성된 전체 차량 특징 이름 리스트입니다.
        mode_col_name = [c for c in self.veh_cols if 'mode' in c][0]
        mode_idx = self.veh_cols.index(mode_col_name)
        
        # 3. motion_encoder를 위해 1채널('mode') 데이터만 슬라이싱합니다.
        # (B, T, 12) -> (B, T, 1)
        veh_e_for_motion = veh_e_all_channels[:, :, mode_idx:mode_idx+1]
        
        # emotion_encoder는 12채널 전체를 그대로 사용합니다.
        veh_mask_e = batch["veh_mask_emotion"].to(self.cfg.device)

        # 각 인코더에 "올바른 채널 수"의 데이터를 전달합니다.
        motion_feat_seq = self.motion_encoder(imu_e, imu_e_lens, veh_e_for_motion, veh_mask_e)
        emotion_feat_vec = self.emotion_encoder(batch) # emotion_encoder는 batch 전체를 받아 내부에서 처리

        # 융합을 위해 차원 맞추기
        T = motion_feat_seq.shape[2]
        emotion_feat_seq = emotion_feat_vec.unsqueeze(2).expand(-1, -1, T)
        
        motion_feat = motion_feat_seq.permute(0, 2, 1)
        emotion_feat = emotion_feat_seq.permute(0, 2, 1)

        # Fusion Block 통과
        fusion_inputs = {'motion': motion_feat, 'emotion': emotion_feat}
        fused_feat = self.fusion_block(fusion_inputs)

        # 각 Head에 전달하여 최종 예측
        motion_logits = self.motion_head(fused_feat.permute(0, 2, 1))
        emotion_logits = self.emotion_head(fused_feat.mean(dim=1))
        
        return {'motion': motion_logits, 'emotion': emotion_logits}

    def run_epoch(self, loader, train: bool):
        # 모델을 학습 또는 평가 모드로 설정
        self.fusion_block.train(train)
        self.motion_head.train(train)
        self.emotion_head.train(train)
        
        total_loss, total_frames = 0.0, 0
        
        for batch in loader:
            #print("Batch keys:", batch.keys()) 
            with torch.set_grad_enabled(train):
                # 1. Forward Pass
                out = self.forward(batch)
                motion_logits = out['motion']
                emotion_logits = out['emotion']

                # 2. Motion 손실 계산
                raw_mot = batch["label_motion"].long().to(self.cfg.device)
                valid_mot = (raw_mot > 0)
                tgt_mot = torch.zeros_like(raw_mot)
                tgt_mot[valid_mot] = raw_mot[valid_mot] - 1
                
                B, T, C = motion_logits.shape
                loss_mot = self.ce_mot(motion_logits.reshape(-1, C)[valid_mot.reshape(-1)], tgt_mot.reshape(-1)[valid_mot.reshape(-1)])

                # 3. Emotion 손실 계산 (Valence/Arousal)
                raw_v = batch['valence_reg_emotion'].reshape(-1).to(self.cfg.device)
                tgt_v = torch.full_like(raw_v, self.cfg.ign_emo, dtype=torch.long)
                tgt_v[raw_v < 4] = 0; tgt_v[(raw_v >= 4) & (raw_v < 7)] = 1; tgt_v[raw_v >= 7] = 2
                mask_v = (tgt_v != self.cfg.ign_emo)
                loss_v = self.ce_v(emotion_logits['valence_logits'][mask_v], tgt_v[mask_v]) if mask_v.any() else 0.0

                raw_a = batch['arousal_reg_emotion'].reshape(-1).to(self.cfg.device)
                tgt_a = torch.full_like(raw_a, self.cfg.ign_emo, dtype=torch.long)
                tgt_a[raw_a < 4] = 0; tgt_a[(raw_a >= 4) & (raw_a < 7)] = 1; tgt_a[raw_a >= 7] = 2
                mask_a = (tgt_a != self.cfg.ign_emo)
                loss_a = self.ce_a(emotion_logits['arousal_logits'][mask_a], tgt_a[mask_a]) if mask_a.any() else 0.0

                # 4. 손실 병합 (가중치는 cfg에서 조절 가능)
                loss = loss_mot + (loss_v + loss_a)

            # 5. Backward Pass
            if train:
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()

            total_loss += loss.item() * B
            total_frames += B

        return total_loss / max(total_frames, 1)

    @torch.no_grad()
    def evaluate(self, loader):
        self.eval() # 전체 모델을 평가 모드로
        
        # Motion 예측값/정답값 버퍼
        preds_mot, trues_mot = [], []
        # Emotion 예측값/정답값 버퍼
        preds_v, trues_v = [], []
        preds_a, trues_a = [], []

        for batch in loader:
            out = self.forward(batch)
            
            # Motion Accuracy 계산
            p_mot = out['motion'].argmax(-1).cpu()
            t_mot = (batch["label_motion"].cpu() - 1)
            m_mot = t_mot >= 0
            preds_mot.append(p_mot[m_mot]); trues_mot.append(t_mot[m_mot])
            
            # Emotion Accuracy 계산
            p_v = out['emotion']['valence_logits'].argmax(-1).cpu()
            raw_v = batch['valence_reg_emotion'].reshape(-1).cpu()
            t_v = torch.full_like(raw_v, -1, dtype=torch.long)
            t_v[raw_v < 4] = 0; t_v[(raw_v >= 4) & (raw_v < 7)] = 1; t_v[raw_v >= 7] = 2
            m_v = t_v != -1
            preds_v.append(p_v[m_v]); trues_v.append(t_v[m_v])
            
            p_a = out['emotion']['arousal_logits'].argmax(-1).cpu()
            raw_a = batch['arousal_reg_emotion'].reshape(-1).cpu()
            t_a = torch.full_like(raw_a, -1, dtype=torch.long)
            t_a[raw_a < 4] = 0; t_a[(raw_a >= 4) & (raw_a < 7)] = 1; t_a[raw_a >= 7] = 2
            m_a = t_a != -1
            preds_a.append(p_a[m_a]); trues_a.append(t_a[m_a])

        acc_mot = accuracy_score(torch.cat(trues_mot), torch.cat(preds_mot))
        acc_v = accuracy_score(torch.cat(trues_v), torch.cat(preds_v))
        acc_a = accuracy_score(torch.cat(trues_a), torch.cat(preds_a))
        
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