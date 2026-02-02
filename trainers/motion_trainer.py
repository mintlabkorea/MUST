import warnings

# checkpoint use_reentrant 경고 무시
warnings.filterwarnings(
    "ignore",
    message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly"
)

# cpu.amp.autocast FutureWarning 무시
warnings.filterwarnings(
    "ignore",
    message="`torch\\.cpu\\.amp\\.autocast` is deprecated"
)
import os, itertools, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt; import seaborn as sns; plt.switch_backend("Agg")
import numpy as np
import matplotlib 
matplotlib.use('Agg')
from config.config import Config
from trainers.base_trainer import TrainerBase, dataProcessor
from models.motion_encoder import MotionEncoder
from models.head.motion_head import MotionHead
from data.loader import make_motion_loader
from utils.visualization import plot_confusion_matrix


class MotionTrainer(TrainerBase, dataProcessor):
    def __init__(self, cfg, train_keys, val_keys, test_keys):
        # 1. dataProcessor 초기화
        dataProcessor.__init__(self, cfg)
        self.prepare()
        
        # 2. main.py로부터 전달받은 key로 덮어쓰기
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.test_keys = test_keys

        self.modalities = cfg.PretrainMotion.modalities_to_use  
        # ➋ drive 레이블 값 (raw label 중 제외할 값)
        self.drive_label = 4

        # 3. 나머지 Trainer 초기화
        self.cfg = cfg
        self._build_model()
        self._create_optimizer()
        self._move_all_to_device() # 순서 변경: 옵티마이저 생성 후 GPU로 이동

        os.makedirs('weights', exist_ok=True)
        os.makedirs('results/pretrain', exist_ok=True)

    def load(self, state):
        """
        state가 아래 두 케이스 모두를 지원:
        (A) {'encoder': sd, 'head': sd, 'optim': sd, 'scaler': sd}
        (B) {'encoder.xxx': tensor, 'head.xxx': tensor, ...} 플랫 dict
        """
        if not isinstance(state, dict):
            raise ValueError("Invalid checkpoint: expected dict")

        def _maybe_load_opt_and_scaler(d):
            if "optim" in d and hasattr(self, "optim"):
                try: self.optim.load_state_dict(d["optim"])
                except Exception: pass
            if "scaler" in d and hasattr(self, "scaler"):
                try: self.scaler.load_state_dict(d["scaler"])
                except Exception: pass

        # (A) 서브모듈 dict 형태
        if "encoder" in state or "head" in state:
            if "encoder" in state:
                self.encoder.load_state_dict(state["encoder"], strict=False)
            if "head" in state:
                self.head.load_state_dict(state["head"], strict=False)
            _maybe_load_opt_and_scaler(state)
            return

        # (B) 플랫 dict 형태 → 접두사별로 분리
        enc_sd = {k.split("encoder.", 1)[1]: v for k, v in state.items() if k.startswith("encoder.")}
        head_sd = {k.split("head.", 1)[1]: v for k, v in state.items() if k.startswith("head.")}
        if enc_sd:
            self.encoder.load_state_dict(enc_sd, strict=False)
        if head_sd:
            self.head.load_state_dict(head_sd, strict=False)
        # optim/scaler는 플랫 dict에 거의 없지만 혹시 몰라 체크
        _maybe_load_opt_and_scaler(state)

    # ---------- model ----------
    def _build_model(self):
        self.encoder = MotionEncoder(self.cfg)
        self.head = MotionHead(
            self.cfg.PretrainMotion.hidden_dim, 
            self.cfg.PretrainMotion.num_motion
        )
        self.loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(self.cfg.PretrainMotion.class_weights, device=self.cfg.Project.device),
            ignore_index=self.cfg.PretrainMotion.ignore_index
        )
        self.scaler  = GradScaler()

    def _create_optimizer(self):
        self.optim = torch.optim.Adam(
            itertools.chain(self.encoder.parameters(), self.head.parameters()),
            lr=self.cfg.PretrainMotion.lr
        )

    # ---------- dataloader ----------
    def make_loader(self, keys, shuffle):
        return make_motion_loader(self.cfg, keys, shuffle, dp=self)

    def _move_all_to_device(self):
        # self.__dict__를 순회하며 모든 nn.Module을 GPU로 이동
        for name, obj in self.__dict__.items():
            if isinstance(obj, nn.Module):
                obj.to(self.cfg.Project.device)

    # ---------- forward ----------
    def forward(self, batch):
        d = self.cfg.Project.device

        modal = self.modalities
        kwargs = {}
        if 'imu' in modal:
            imu = batch["imu_motion"].to(d)
            kwargs['imu_seq'] = imu
            kwargs['imu_len'] = (imu.abs().sum(-1) > 0).sum(1)
            
        if 'veh' in modal:
            veh  = batch["veh_motion"].to(d)
            mask = batch["veh_mask_motion"].to(d).bool()
            kwargs['veh_seq']  = veh
            kwargs['veh_mask'] = mask
        
        # PPG 데이터가 없을 경우를 대비한 안전 로직 추가
        if 'ppg' in modal and 'ppg_emotion' in batch:
            kwargs['ppg_seq'] = batch["ppg_emotion"].to(d)
            if 'ppg_rr_emotion' in batch:
                kwargs['ppg_rr']    = batch["ppg_rr_emotion"].to(d)
            if 'ppg_rmssd_emotion' in batch:
                kwargs['ppg_rmssd'] = batch["ppg_rmssd_emotion"].to(d)
            if 'ppg_sdnn_emotion' in batch:
                kwargs['ppg_sdnn']  = batch["ppg_sdnn_emotion"].to(d)

        if 'sc' in modal:
            kwargs['scenario_ids']   = batch["sc_motion_evt"].to(d)
            kwargs['scenario_types'] = batch["sc_motion_type"].to(d)
            kwargs['phase_ids']      = batch["sc_motion_phase"].to(d)
            kwargs['timestamps']     = batch["sc_motion_time"].to(d)

        if 'survey' in modal:
            kwargs['survey'] = batch["survey_e"].to(d)
 
        feat = self.encoder(**kwargs)
        out  = self.head(feat)
        return out

    # ---------- 1 epoch ----------
    def run_epoch(self, loader, train=True):
        self.encoder.train(train)
        self.head.train(train)
        iterator = tqdm(loader, desc="Train" if train else "Eval")
        total_loss, total_frames = 0.0, 0

        for batch in iterator:
            # NaN/Inf 값 처리
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

            # 1) 원본 raw 레이블 (0은 패딩/무시, 1~4가 실제 클래스)
            raw = batch["label_motion"].long().to(self.cfg.Project.device)  # shape (B,T)

            # 2) valid mask: raw>0 인 프레임만 학습에 사용
            valid = (raw > 0) & (raw != self.drive_label) 

            # 3) 0-based 변환: valid 위치만 -1
            tgt = torch.zeros_like(raw)
            tgt[valid] = raw[valid] - 1   # 이제 tgt in {0,1,2,3} for valid, 0 for invalid

            # 4) forward + flatten
            logits = self.forward(batch)          # (B,T,C)
            B, T, C = logits.shape
            logits_flat = logits.reshape(-1, C)     # (B*T, C)
            tgt_flat    = tgt.reshape(-1)           # (B*T,)
            mask_flat   = valid.reshape(-1)         # (B*T,)
            
            # 5) loss (valid 프레임에 한정)
            if mask_flat.any():
                loss = self.loss_fn(logits_flat[mask_flat], tgt_flat[mask_flat])
            else:
                loss = torch.tensor(0., device=self.cfg.Project.device)

            # 6) backward / optimizer step
            if train:
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                self.scaler.step(self.optim)
                self.scaler.update()

            total_loss   += loss.item() * mask_flat.sum().item()
            total_frames += mask_flat.sum().item()

        return total_loss / max(total_frames, 1)


    # ---------- evaluation ----------
    @torch.no_grad()
    def evaluate(self, split):
        loader = self.make_loader(getattr(self, f"{split}_keys"), shuffle=False)
        preds, trues = [], []

        for batch in loader:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            
            logits = self.forward(batch)               # (B,T,C)
            p      = logits.argmax(-1).cpu()           # (B,T)
            raw = batch["label_motion"].cpu()               # (B,T)
            t   = raw - 1                                   # 0~3 로 변환
            # ignore padding(0→-1) & drive_label(4)
            m   = (t != self.cfg.PretrainMotion.ignore_index) & (raw != self.drive_label)

            preds.append(p[m]); trues.append(t[m])

        if not preds or not any(p.numel() > 0 for p in preds):
            return 0.0

        y_p = torch.cat(preds)
        y_t = torch.cat(trues)
        return accuracy_score(y_t, y_p)

    @torch.no_grad()
    def get_predictions(self, loader):
        """주어진 로더에 대한 예측값과 실제값을 반환합니다."""
        self.encoder.eval()
        self.head.eval()
        preds, trues = [], []
        for batch in loader:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            
            logits = self.forward(batch)
            p = logits.argmax(-1).cpu()
            raw = batch["label_motion"].cpu()
            t   = raw - 1
            m   = (t != self.cfg.PretrainMotion.ignore_index) & (raw != self.drive_label)
            preds.append(p[m]); trues.append(t[m])

        # 핸들링: 예측값이 없는 경우 빈 배열 반환
        if not trues or not preds or not any(t.numel() > 0 for t in trues) or not any(p.numel() > 0 for p in preds):
            return np.array([]), np.array([])

        return torch.cat(trues).numpy(), torch.cat(preds).numpy()

    # ---------- train loop ----------
    def train(self, save_path=None):
        """전체 학습 루프를 실행합니다."""
        tr_loader = self.make_loader(self.train_keys, True)
        va_loader = self.make_loader(self.val_keys, False)

        best_loss, patience = float('inf'), 0
        # 모달리티 조합에 따라 파일 이름 동적 생성
        modalities_str = "_".join(sorted(list(self.modalities)))
        final_save_path = save_path if save_path else f"weights/best_motion_{modalities_str}.pt"
        
        best_state = None  # 최적의 모델 상태를 저장할 변수

        for epoch in range(1, self.cfg.PretrainMotion.epochs + 1):
            tr_loss = self.run_epoch(tr_loader, True)
            va_loss = self.run_epoch(va_loader, False)
            va_acc = self.evaluate("val")

            print(f"Epoch {epoch:02d} | L_tr {tr_loss:.4f}  L_val {va_loss:.4f}  Acc {va_acc:.3f}")

            if va_loss < best_loss:
                best_loss, patience = va_loss, 0
                # 최적의 모델 상태를 딕셔너리로 저장
                best_state = {
                    "encoder": self.encoder.state_dict(),
                    "head": self.head.state_dict(),
                    "optim": self.optim.state_dict(),
                    "scaler": self.scaler.state_dict()
                }
                torch.save(best_state, final_save_path)
            else:
                patience += 1
                if patience >= self.cfg.PretrainMotion.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # --- 최종 평가 및 결과 반환 로직 ---
        print("\n--- Final Evaluation & Saving Confusion Matrix ---")
        
        # NOTE: 최적의 모델 상태를 로드하여 최종 평가에 사용
        if best_state:
            self.encoder.load_state_dict(best_state["encoder"], strict=False)
            self.head.load_state_dict(best_state["head"], strict=False)
        else:
            print("Warning: No best model state found. Evaluating with the last model.")

        val_loader = self.make_loader(self.val_keys, False)
        val_t, val_p = self.get_predictions(val_loader)
        test_loader = self.make_loader(self.test_keys, False)
        test_t, test_p = self.get_predictions(test_loader)

        final_val_acc = accuracy_score(val_t, val_p) if val_t.size > 0 else 0.0
        final_test_acc = accuracy_score(y_true=test_t, y_pred=test_p) if test_t.size > 0 else 0.0

        print(f">>> FINAL VAL  ACC = {final_val_acc:.4f}")
        print(f">>> FINAL TEST ACC = {final_test_acc:.4f}")
        
        # --- 혼동 행렬 저장 로직 ---
        class_labels = self.cfg.PretrainMotion.class_labels
        plot_confusion_matrix(val_t, val_p, class_labels, "Pretrain Motion - Val", "./results/pretrain/motion_val_cm.png")
        plot_confusion_matrix(test_t, test_p, class_labels, "Pretrain Motion - Test", "./results/pretrain/motion_test_cm.png")
        print("Pre-training confusion matrices for motion saved to ./results/pretrain/")
        # ------------------------------------

        # 자동화 스크립트를 위해 최종 테스트 정확도 반환
        return {'test_acc_mot': final_test_acc}

# class MotionTrainer(TrainerBase, dataProcessor):

#     def __init__(self, cfg, train_keys, val_keys, test_keys):
#         # 1. dataProcessor 초기화
#         dataProcessor.__init__(self, cfg)
#         self.prepare()
        
#         # 2. main.py로부터 전달받은 key로 덮어쓰기
#         self.train_keys = train_keys
#         self.val_keys = val_keys
#         self.test_keys = test_keys

#         self.modalities = cfg.PretrainMotion.modalities_to_use  
#         # ➋ drive 레이블 값 (raw label 중 제외할 값)
#         self.drive_label = 4

#         # 3. 나머지 Trainer 초기화
#         self.cfg = cfg
#         self._build_model()
        

#         self._create_optimizer()
#         self._move_all_to_device() # 순서 변경: 옵티마이저 생성 후 GPU로 이동

#         os.makedirs('weights', exist_ok=True)
#         os.makedirs('results/pretrain', exist_ok=True)

#     # ---------- model ----------
#     def _build_model(self):
#         self.encoder = MotionEncoder(self.cfg)
#         self.head = MotionHead(
#             self.cfg.PretrainMotion.hidden_dim, 
#             self.cfg.PretrainMotion.num_motion
#         )
#         self.loss_fn = nn.CrossEntropyLoss(
#             weight=torch.tensor(self.cfg.PretrainMotion.class_weights, device=self.cfg.Project.device),
#             ignore_index=self.cfg.PretrainMotion.ignore_index
#         )
#         self.scaler  = GradScaler()

#     def _create_optimizer(self):
#         self.optim = torch.optim.Adam(
#             itertools.chain(self.encoder.parameters(), self.head.parameters()),
#             lr=self.cfg.PretrainMotion.lr
#         )

#     # ---------- dataloader ----------
#     def make_loader(self, keys, shuffle):
#         return make_motion_loader(self.cfg, keys, shuffle, dp=self)

#     def _move_all_to_device(self):
#         # self.__dict__를 순회하며 모든 nn.Module을 GPU로 이동
#         for name, obj in self.__dict__.items():
#             if isinstance(obj, nn.Module):
#                 obj.to(self.cfg.Project.device)

#     # ---------- forward ----------
#     def forward(self, batch):
#         d = self.cfg.device

#         modal = self.modalities
#         kwargs = {}
#         if 'imu' in modal:
#             imu = batch["imu_motion"].to(d)
#             kwargs['imu_seq'] = imu
#             kwargs['imu_len'] = (imu.abs().sum(-1) > 0).sum(1)
            
#         if 'veh' in modal:
#             veh  = batch["veh_motion"].to(d)
#             mask = batch["veh_mask_motion"].to(d).bool()
#             kwargs['veh_seq']  = veh
#             kwargs['veh_mask'] = mask
        
#         if 'ppg' in modal:
#             # (B,T,D_ppg)
#             kwargs['ppg_seq'] = batch["ppg_emotion"].to(d)
#             kwargs['ppg_rr'] = batch["ppg_rr_emotion"].to(d)
#             kwargs['ppg_rmssd'] = batch["ppg_rmssd_emotion"].to(d)
#             kwargs['ppg_sdnn'] = batch["ppg_sdnn_emotion"].to(d)

#         if 'sc' in modal:
#             kwargs['scenario_ids']   = batch["sc_motion_evt"].to(d)
#             kwargs['scenario_types'] = batch["sc_motion_type"].to(d)
#             kwargs['phase_ids']      = batch["sc_motion_phase"].to(d)
#             kwargs['timestamps']     = batch["sc_motion_time"].to(d)

#         if 'survey' in modal:
#             # (B, SURVEY_FEATURE_DIM)
#             kwargs['survey'] = batch["survey_e"].to(d)  # collate_fn_unified 에서 'survey_e' 로 생성 :contentReference[oaicite:7]{index=7}
 
#         # 3) encode → (B,H,T)
#         feat = self.encoder(**kwargs)

#         # 4) head → (B,T,C)
#         out  = self.head(feat)
#         return out

#     # ---------- 1 epoch ----------
#     def run_epoch(self, loader, train=True):
#         self.encoder.train(train)
#         self.head.train(train)
#         iterator = tqdm(loader, desc="Train" if train else "Eval")
#         total_loss, total_frames = 0.0, 0

#         for batch in iterator:
#             for key, value in batch.items():
#                 if isinstance(value, torch.Tensor):
#                     batch[key] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

#             # 1) 원본 raw 레이블 (0은 패딩/무시, 1~4가 실제 클래스)
#             raw = batch["label_motion"].long().to(self.cfg.device)  # shape (B,T)

#             # 2) valid mask: raw>0 인 프레임만 학습에 사용
#             valid = (raw > 0) & (raw != self.drive_label) 

#             # 3) 0-based 변환: valid 위치만 -1
#             tgt = torch.zeros_like(raw)
#             tgt[valid] = raw[valid] - 1   # 이제 tgt in {0,1,2,3} for valid, 0 for invalid

#             # 4) forward + flatten
#             logits = self.forward(batch)          # (B,T,C)
#             B, T, C = logits.shape
#             logits_flat = logits.reshape(-1, C)     # (B*T, C)
#             tgt_flat    = tgt.reshape(-1)           # (B*T,)
#             mask_flat   = valid.reshape(-1)         # (B*T,)

#             if torch.isinf(logits_flat).any():
#                 print("!!! Inf DETECTED in logits before loss calculation!")
#             if torch.isnan(logits_flat).any():
#                 print("!!! NaN DETECTED in logits before loss calculation!")
                    
#             # 5) loss (valid 프레임에 한정)
#             if mask_flat.any():
#                 loss = self.loss_fn(logits_flat[mask_flat], tgt_flat[mask_flat])
#             else:
#                 loss = torch.tensor(0., device=self.cfg.device)

#             # 6) backward / optimizer step
#             if train:
#                 self.optim.zero_grad()
#                 self.scaler.scale(loss).backward()
#                 self.scaler.unscale_(self.optim)
#                 torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
#                 self.scaler.step(self.optim)
#                 self.scaler.update()

#             total_loss   += loss.item() * mask_flat.sum().item()
#             total_frames += mask_flat.sum().item()

#         return total_loss / max(total_frames, 1)


#     # ---------- evaluation ----------
#     @torch.no_grad()
#     def evaluate(self, split):
#         loader = self.make_loader(getattr(self, f"{split}_keys"), shuffle=False)
#         preds, trues = [], []

#         for batch in loader:
#             logits = self.forward(batch)               # (B,T,C)
#             p      = logits.argmax(-1).cpu()           # (B,T)
#             raw = batch["label_motion"].cpu()               # (B,T)
#             t   = raw - 1                                   # 0~3 로 변환
#             # ignore padding(0→-1) & drive_label(4)
#             m   = (t != self.cfg.PretrainMotion.ignore_index) & (raw != self.drive_label)

#             preds.append(p[m]); trues.append(t[m])

#         if not preds:
#             return 0.0

#         y_p = torch.cat(preds)
#         y_t = torch.cat(trues)
#         return accuracy_score(y_t, y_p)

#     @torch.no_grad()
#     def get_predictions(self, loader):
#         """주어진 로더에 대한 예측값과 실제값을 반환합니다."""
#         self.encoder.eval()
#         self.head.eval()
#         preds, trues = [], []
#         for batch in loader:
#             logits = self.forward(batch)
#             p = logits.argmax(-1).cpu()
#             raw = batch["label_motion"].cpu()
#             t   = raw - 1
#             m   = (t != self.cfg.PretrainMotion.ignore_index) & (raw != self.drive_label)
#             preds.append(p[m]); trues.append(t[m])

#         # 핸들링: 예측값이 없는 경우 빈 배열 반환
#         if not trues or not preds:
#             return np.array([]), np.array([])

#         return torch.cat(trues).numpy(), torch.cat(preds).numpy()

#     # ---------- train loop ----------
#     def train(self):
#         """전체 학습 루프를 실행합니다."""
#         tr_loader = self.make_loader(self.train_keys, True)
#         va_loader = self.make_loader(self.val_keys,   False)

#         best_loss, patience = float('inf'), 0
#         for epoch in range(1, self.cfg.PretrainMotion.epochs + 1):
#             tr_loss = self.run_epoch(tr_loader, True)
#             va_loss = self.run_epoch(va_loader, False)
#             va_acc  = self.evaluate("val")

#             print(f"Epoch {epoch:02d} | L_tr {tr_loss:.4f}  L_val {va_loss:.4f}  Acc {va_acc:.3f}")

#             if va_loss < best_loss:
#                 best_loss, patience = va_loss, 0
#                 torch.save({
#                     "encoder": self.encoder.state_dict(),
#                     "head":    self.head.state_dict(),
#                     "optim":   self.optim.state_dict(),
#                     "scaler":  self.scaler.state_dict()
#                 }, "weights/best_motion.pt")
#             else:
#                 patience += 1
#                 if patience >= self.cfg.PretrainMotion.patience:
#                     print(f"Early stopping at epoch {epoch}")
#                     break

#         # --- 최종 평가 및 결과 반환 로직 ---
#         print("\n--- Final Evaluation & Saving Confusion Matrix ---")
#         # best ckpt 로드
#         ckpt = torch.load("weights/best_motion.pt", map_location=self.cfg.Project.device)
#         self.encoder.load_state_dict(ckpt["encoder"], strict=False)
#         self.head.load_state_dict(ckpt["head"], strict=False)
        
#         # Validation 및 Test 데이터에 대한 예측값과 실제값 가져오기
#         val_loader = self.make_loader(self.val_keys, False)
#         val_t, val_p = self.get_predictions(val_loader)
        
#         test_loader = self.make_loader(self.test_keys, False)
#         test_t, test_p = self.get_predictions(test_loader)
        
#         # 정확도 계산
#         final_val_acc = accuracy_score(val_t, val_p)
#         final_test_acc = accuracy_score(test_t, test_p)

#         print(f">>> FINAL VAL  ACC = {final_val_acc:.4f}")
#         print(f">>> FINAL TEST ACC = {final_test_acc:.4f}")
        
#         # --- [복원] 혼동 행렬 저장 로직 ---
#         class_labels = self.cfg.PretrainMotion.class_labels
#         plot_confusion_matrix(val_t, val_p, class_labels, "Pretrain Motion - Val", "./results/pretrain/motion_val_cm.png")
#         plot_confusion_matrix(test_t, test_p, class_labels, "Pretrain Motion - Test", "./results/pretrain/motion_test_cm.png")
#         print("Pre-training confusion matrices for motion saved to ./results/pretrain/")
#         # ------------------------------------

#         # 자동화 스크립트를 위해 최종 테스트 정확도 반환
#         return {'test_acc_mot': final_test_acc}