# trainers/emotion_trainer.py

import os, itertools, torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.cuda.amp import GradScaler
import pandas as pd
# 로컬 모듈 임포트
from trainers.base_trainer import TrainerBase, dataProcessor
from data.loader import make_emotion_loader
from models.emotion_encoder import EmotionEncoder
from models.fusion.predictors import EmotionPredictor
from utils.visualization import plot_confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR

class EmotionTrainer(TrainerBase, dataProcessor):
    def __init__(self, cfg, dp):
        self.cfg = cfg
        self.dp = dp  # Store the dataProcessor instance
        
        # Get keys directly from the stored dp object
        self.train_keys, self.val_keys, self.test_keys = dp.train_keys, dp.val_keys, dp.test_keys
        
        self._build_model()
        self._move_model_to_device()
        self._create_optimizer()

        os.makedirs('weights', exist_ok=True)
        os.makedirs('results/pretrain', exist_ok=True)

        print("--- Checking Label Distribution for Training Set ---")
        valence_counts = pd.Series(dtype='int')
        arousal_counts = pd.Series(dtype='int')

        # self.train_keys와 self.data_map을 사용
        for key in self.train_keys:
            labels = self.dp.data_map[key]['label'] # dp를 통해 data_map 접근
            valence_counts = pd.concat([valence_counts, labels['label_valence']])
            arousal_counts = pd.concat([arousal_counts, labels['label_arousal']])

        # -100과 같은 무시 인덱스를 제외하고 실제 레이블(1~9)의 개수만 집계
        print("Valence Distribution:\n", valence_counts[(valence_counts >= 1) & (valence_counts < 10)].value_counts().sort_index())
        print("\nArousal Distribution:\n", arousal_counts[(arousal_counts >= 1) & (arousal_counts < 10)].value_counts().sort_index())
                
    def _build_model(self):
        """인코더와 2개의 예측기(Predictor)를 생성하고 손실 함수를 정의합니다."""
        self.encoder = EmotionEncoder(self.cfg)
        hidden_dim = self.cfg.PretrainEmotion.hidden_dim
        
        # Survey는 context로 빠지므로, 융합되는 동적 모달리티의 수만 계산
        modalities = set(self.cfg.PretrainEmotion.modalities_to_use)
        self.modalities = modalities
        num_dynamic_modalities = len(modalities - {'survey'})
        predictor_input_dim = hidden_dim * num_dynamic_modalities
        
        print(f"[EmotionTrainer] Predictor input_dim calculated: {predictor_input_dim} "
              f"({num_dynamic_modalities} dynamic modalities * {hidden_dim} hidden_dim)")

        self.valence_predictor = EmotionPredictor(predictor_input_dim, hidden_dim, self.cfg.PretrainEmotion.num_valence)
        self.arousal_predictor = EmotionPredictor(predictor_input_dim, hidden_dim, self.cfg.PretrainEmotion.num_arousal)
        
        device = self.cfg.Project.device
        self.v_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(self.cfg.PretrainEmotion.valence_weights, device=device))
        self.a_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(self.cfg.PretrainEmotion.arousal_weights, device=device))
        self.scaler = GradScaler()
        
    def _create_optimizer(self):
        """모든 모델 파라미터를 포함하는 옵티마이저를 생성합니다."""
        self.optim = torch.optim.Adam(
            itertools.chain(
                self.encoder.parameters(), 
                self.valence_predictor.parameters(), 
                self.arousal_predictor.parameters()
            ),
            lr=self.cfg.PretrainEmotion.lr, 
            weight_decay=self.cfg.PretrainEmotion.weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optim, T_max=self.cfg.PretrainEmotion.epochs, eta_min=1e-6)


    def _move_model_to_device(self):
        """학습에 필요한 모든 모듈을 GPU로 이동시킵니다."""
        device = self.cfg.Project.device
        self.encoder.to(device)
        self.valence_predictor.to(device)
        self.arousal_predictor.to(device)

    def make_loader(self, keys, shuffle):
        """데이터 로더 생성 헬퍼 함수"""
        return make_emotion_loader(self.cfg, keys, shuffle, dp=self.dp)
    
    def forward(self, batch):
        features = self.encoder(batch)  # {'fused': (B, T, D), 'static': (B, D)}

        # extract the static (survey) context
        context = features['static']

        # pass both features dict and context tensor
        valence_logits = self.valence_predictor(features, context)
        arousal_logits = self.arousal_predictor(features, context)

        return {
            'valence_logits': valence_logits,
            'arousal_logits': arousal_logits
        }

    def run_epoch(self, loader, train: bool):
        """한 에폭 동안의 학습 또는 검증을 수행합니다."""
        self.encoder.train(train)
        self.valence_predictor.train(train)
        self.arousal_predictor.train(train)
        
        total_v_loss, total_a_loss, v_frames, a_frames = 0.0, 0.0, 0, 0
        iterator = tqdm(loader, desc=f"Emotion (Train)" if train else "Emotion (Val)")
        #iterator = loader

        for batch in iterator:
            # 원본 값 (1~9)을 클래스 인덱스 (0~8)로 변환
            raw_v = batch['valence_reg_emotion'].reshape(-1)
            # 1~9 범위를 벗어나는 값은 무시하도록 마스크 생성
            mask_v = (raw_v >= 1) & (raw_v < 10)
            # 유효한 값들만 인덱스로 변환, 나머지는 ignore_index로 채움
            tgt_v = torch.full_like(raw_v, self.cfg.PretrainEmotion.ignore_index, dtype=torch.long)
            tgt_v[mask_v] = raw_v[mask_v].long() - 1

            raw_a = batch['arousal_reg_emotion'].reshape(-1)
            mask_a = (raw_a >= 1) & (raw_a < 10)
            tgt_a = torch.full_like(raw_a, self.cfg.PretrainEmotion.ignore_index, dtype=torch.long)
            tgt_a[mask_a] = raw_a[mask_a].long() - 1

            tgt_v = tgt_v.to(self.cfg.Project.device)
            tgt_a = tgt_a.to(self.cfg.Project.device)
            # 마스크는 ignore_index가 아닌 샘플만 사용하도록 재정의
            mask_v = (tgt_v != self.cfg.PretrainEmotion.ignore_index)
            mask_a = (tgt_a != self.cfg.PretrainEmotion.ignore_index)

            with torch.set_grad_enabled(train):
                out = self.forward(batch)
                
                # Valence 손실
                l_v = self.v_loss_fn(out['valence_logits'][mask_v], tgt_v[mask_v]) if mask_v.any() else torch.tensor(0., device=self.cfg.Project.device)
                # Arousal 손실
                l_a = self.a_loss_fn(out['arousal_logits'][mask_a], tgt_a[mask_a]) if mask_a.any() else torch.tensor(0., device=self.cfg.Project.device)
                
                loss = self.cfg.PretrainEmotion.lambda_valence * l_v + self.cfg.PretrainEmotion.lambda_arousal * l_a

            if train and loss.requires_grad:
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(itertools.chain(self.encoder.parameters(), self.valence_predictor.parameters(), self.arousal_predictor.parameters()), 1.0)
                self.scaler.step(self.optim)
                self.scaler.update()

            total_v_loss += l_v.item() * mask_v.sum().item()
            total_a_loss += l_a.item() * mask_a.sum().item()
            v_frames += mask_v.sum().item()
            a_frames += mask_a.sum().item()

        return total_v_loss / max(v_frames, 1), total_a_loss / max(a_frames, 1)

    @torch.no_grad()
    def evaluate(self, split):
        """검증/테스트 세트에 대한 정확도를 계산합니다."""
        loader = make_emotion_loader(self.cfg, getattr(self, f"{split}_keys"), shuffle=False, dp=self.dp)
        self.encoder.eval()
        self.valence_predictor.eval()
        self.arousal_predictor.eval()
        
        preds = self.get_predictions(loader)
        val_acc = accuracy_score(preds['val_t'], preds['val_p'])
        aro_acc = accuracy_score(preds['aro_t'], preds['aro_p'])
        return val_acc, aro_acc

    @torch.no_grad()
    def get_predictions(self, loader):
        """[수정] 예측값, 실제값과 함께 각 샘플의 시점 ID도 수집합니다."""
        self.encoder.eval()
        self.valence_predictor.eval()
        self.arousal_predictor.eval()

        val_trues, val_preds, val_phases = [], [], []
        aro_trues, aro_preds, aro_phases = [], [], []

        for batch in loader:
            out = self.forward(batch)
            vp = out['valence_logits'].argmax(-1).cpu()
            ap = out['arousal_logits'].argmax(-1).cpu()

            # collate_fn으로부터 시점 ID 텐서를 받음
            phases = batch['phase_evt_e'].cpu()

            # Valence 처리
            raw_v = batch['valence_reg_emotion']
            # 실제값(True)을 0-8 인덱스로 변환
            vt_multiclass = torch.full_like(raw_v, self.cfg.PretrainEmotion.ignore_index, dtype=torch.long)
            valid_mask_v = (raw_v >= 1) & (raw_v < 10)
            vt_multiclass[valid_mask_v] = raw_v[valid_mask_v].long() - 1
            mask_v = vt_multiclass != self.cfg.PretrainEmotion.ignore_index
            # .squeeze()를 추가하여 [Batch, 1] 모양을 [Batch]로 변경
            mask_v = mask_v.squeeze()


            # 예측값(Prediction)은 모델 출력(0-8) 그대로 사용
            vp_multiclass = out['valence_logits'].argmax(-1).cpu().reshape(-1)

            # --- 정확도 평가를 위해 이진으로 변환 ---
            # True values (vt_binary -> vt_ternary로 변수명 변경)
            vt_ternary = torch.full_like(vt_multiclass, -1)
            vt_ternary[(vt_multiclass >= 0) & (vt_multiclass <= 3)] = 0  # Low: 1,2,3점
            vt_ternary[(vt_multiclass >= 4) & (vt_multiclass <= 8)] = 1  # Medium: 4,5,6점

            # Predicted values (vp_binary -> vp_ternary로 변수명 변경)
            vp_ternary = torch.full_like(vp_multiclass, -1)
            vp_ternary[(vp_multiclass >= 0) & (vp_multiclass <= 3)] = 0
            vp_ternary[(vp_multiclass >= 4) & (vp_multiclass <= 8)] = 1

            val_trues.append(vt_ternary[mask_v])
            val_preds.append(vp_ternary[mask_v])
            val_phases.append(phases[mask_v])

            # Arousal 처리
            raw_a = batch['arousal_reg_emotion'].reshape(-1)
            # 실제값(True)을 0-8 인덱스로 변환
            at_multiclass = torch.full_like(raw_a, self.cfg.PretrainEmotion.ignore_index, dtype=torch.long)
            valid_mask_a = (raw_a >= 1) & (raw_a < 10)
            at_multiclass[valid_mask_a] = raw_a[valid_mask_a].long() - 1
            mask_a = at_multiclass != self.cfg.PretrainEmotion.ignore_index
            mask_a.squeeze()

            # 예측값(Prediction)은 모델 출력(0-8) 그대로 사용
            ap_multiclass = out['arousal_logits'].argmax(-1).cpu().reshape(-1)


            # --- 정확도 평가를 위해 이진으로 변환 ---
            at_ternary = torch.full_like(at_multiclass, -1)
            at_ternary[(at_multiclass >= 0) & (at_multiclass <= 3)] = 0
            at_ternary[(at_multiclass >= 4) & (at_multiclass <= 8)] = 1
            
            # [참고] Arousal 예측값(ap_multiclass)은 Valence 예측값(vp_multiclass)을 잘못 사용하고 있을 수 있으니 확인이 필요합니다.
            # 아래 코드는 ap_multiclass가 올바르다는 가정 하에 작성되었습니다.
            ap_ternary = torch.full_like(ap_multiclass, -1)
            ap_ternary[(ap_multiclass >= 0) & (ap_multiclass <= 3)] = 0
            ap_ternary[(ap_multiclass >= 4) & (ap_multiclass <= 8)] = 1

            aro_trues.append(at_ternary[mask_a])
            aro_preds.append(ap_ternary[mask_a])
            aro_phases.append(phases[mask_a])
            
        return {
            'val_t': torch.cat(val_trues).numpy(), 'val_p': torch.cat(val_preds).numpy(),
            'aro_t': torch.cat(aro_trues).numpy(), 'aro_p': torch.cat(aro_preds).numpy(),
            'val_phases': torch.cat(val_phases).numpy(),
            'aro_phases': torch.cat(aro_phases).numpy(),
        }

    
    def train(self, save_path=None):
        """전체 학습 루프를 실행합니다."""
        tr_loader = self.make_loader(self.train_keys, True)
        val_loader = self.make_loader(self.val_keys, False)

        best_loss, patience = float('inf'), 0
        for epoch in range(1, self.cfg.PretrainEmotion.epochs + 1):
            tr_loss_v, tr_loss_a = self.run_epoch(tr_loader, True)
            va_loss_v, va_loss_a = self.run_epoch(val_loader, False)
            va_acc_v, va_acc_a = self.evaluate("val")

            self.scheduler.step()

            print(f"Epoch {epoch:02d} | Loss(V/A): {tr_loss_v:.4f}/{tr_loss_a:.4f} | Val Loss(V/A): {va_loss_v:.4f}/{va_loss_a:.4f} | Val Acc(V/A): {va_acc_v:.3f}/{va_acc_a:.3f}")

            current_loss = va_loss_v + va_loss_a
            if current_loss < best_loss:
                best_loss, patience = current_loss, 0
                modalities_str = "_".join(sorted(list(self.modalities)))
                final_save_path = save_path if save_path else f"weights/best_emotion_{modalities_str}_bi.pt"
        
                torch.save({
                    'encoder': self.encoder.state_dict(),
                    'valence_predictor': self.valence_predictor.state_dict(),
                    'arousal_predictor': self.arousal_predictor.state_dict(),
                }, final_save_path)
            else:
                patience += 1
                if patience >= self.cfg.PretrainEmotion.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # --- 최종 평가 및 혼동 행렬 저장 ---
        print("\n--- Final Evaluation & Saving Confusion Matrix (Emotion Pre-training) ---")
        ckpt = torch.load(final_save_path, map_location=self.cfg.device)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.valence_predictor.load_state_dict(ckpt['valence_predictor'])
        self.arousal_predictor.load_state_dict(ckpt['arousal_predictor'])
        
        val_preds = self.get_predictions(val_loader)
        test_loader = make_emotion_loader(self.cfg, self.test_keys, False, dp=self.dp)
        test_preds = self.get_predictions(test_loader)
        
        print(f"Final Val  Acc (V/A): {accuracy_score(val_preds['val_t'], val_preds['val_p']):.4f} / {accuracy_score(val_preds['aro_t'], val_preds['aro_p']):.4f}")
        print(f"Final Test Acc (V/A): {accuracy_score(test_preds['val_t'], test_preds['val_p']):.4f} / {accuracy_score(test_preds['aro_t'], test_preds['aro_p']):.4f}")

        plot_confusion_matrix(val_preds['val_t'], val_preds['val_p'], ["Low",  "High"], "Pretrain Emotion Val - Valence", "./results/pretrain/emo_val_valence_bi_cm.png")
        plot_confusion_matrix(val_preds['aro_t'], val_preds['aro_p'], ["Low", "High"], "Pretrain Emotion Val - Arousal", "./results/pretrain/emo_val_arousal_bi_cm.png")
        plot_confusion_matrix(test_preds['val_t'], test_preds['val_p'], ["Low", "High"], "Pretrain Emotion Test - Valence", "./results/pretrain/emo_test_valence_bi_cm.png")
        plot_confusion_matrix(test_preds['aro_t'], test_preds['aro_p'], ["Low", "High"], "Pretrain Emotion Test - Arousal", "./results/pretrain/emo_test_arousal_bi_cm.png")
        print("Pre-training confusion matrices for emotion saved to ./results/pretrain/")

        final_test_acc_v = accuracy_score(test_preds['val_t'], test_preds['val_p'])
        final_test_acc_a = accuracy_score(test_preds['aro_t'], test_preds['aro_p'])
        print(f"Final Test Acc (V/A): {final_test_acc_v:.4f} / {final_test_acc_a:.4f}")
        
        print("\n" + "="*50)
        print("📊 Analysis of Accuracy by Phase (Test Set)")
        print("="*50)

        # 시점 ID를 실제 이름('before', 'on', 'after')으로 매핑
        phase_map = {0: 'before', 1: 'on', 2: 'after'}

        # Valence 분석
        df_val = pd.DataFrame({
            'true': test_preds['val_t'].flatten(),
            'pred': test_preds['val_p'].flatten(),
            'phase_id': test_preds['val_phases'].flatten()
        })
        df_val['phase'] = df_val['phase_id'].map(phase_map)
        df_val['correct'] = df_val['true'] == df_val['pred']
        val_phase_acc = df_val.groupby('phase')['correct'].mean()

        # Arousal 분석
        df_aro = pd.DataFrame({
            'true': test_preds['aro_t'].flatten(),
            'pred': test_preds['aro_p'].flatten(),
            'phase_id': test_preds['aro_phases'].flatten()
        })
        df_aro['phase'] = df_aro['phase_id'].map(phase_map)
        df_aro['correct'] = df_aro['true'] == df_aro['pred']
        aro_phase_acc = df_aro.groupby('phase')['correct'].mean()

        # 결과 테이블 생성 및 출력
        phase_analysis_df = pd.DataFrame({
            'Valence Acc': val_phase_acc,
            'Arousal Acc': aro_phase_acc
        }).fillna(0).reindex(['before', 'on', 'after']) # 순서 고정
        
        pd.options.display.float_format = '{:,.3f}'.format
        print(phase_analysis_df)
        print("="*50 + "\n")
        # ------------------------------------

        return {'test_acc_v': final_test_acc_v, 'test_acc_a': final_test_acc_a}
