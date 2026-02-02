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
import warnings

# checkpoint use_reentrant 경고 무시
warnings.filterwarnings(
    "ignore",
    message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly"
)

warnings.filterwarnings(
    "ignore",
    message="`torch\\.cpu\\.amp\\.autocast` is deprecated"
)
# cpu.amp.autocast FutureWarning 무시
warnings.filterwarnings(
    "ignore",
    message="`torch\\.cpu\\.amp\\.autocast` is deprecated"
)

class MotionTrainer(TrainerBase, dataProcessor):
    def __init__(self, cfg):
        # 0) dataProcessor init & prepare
        dataProcessor.__init__(self, cfg)
        self.prepare()   # 만약 .prepare()로 래핑했다면, 아니면 self._prepare_data()
        print(">> veh_params keys:", cfg.veh_params.keys())

        cfg.veh_params['input_dim'] = 1
        # 1) 기본 설정
        self.cfg = cfg
        self._build_model()
        self._move_all_to_device()
        self._create_optimizer()

        os.makedirs('weights', exist_ok=True)
        os.makedirs('results', exist_ok=True)

    # ---------- model ----------
    def _build_model(self):
        self.encoder = MotionEncoder(self.cfg)
        self.head    = MotionHead(self.cfg.hidden, self.cfg.num_motion)
        self.loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(self.cfg.motion_weights, device=self.cfg.device),
            ignore_index=self.cfg.ign_mot
        )
        self.scaler  = GradScaler()

    def _create_optimizer(self):
        self.optim = torch.optim.Adam(
            itertools.chain(self.encoder.parameters(), self.head.parameters()),
            lr=self.cfg.lr
        )

    # ---------- dataloader ----------
    def make_loader(self, keys, shuffle):
        return make_motion_loader(self.cfg, keys, shuffle)

    def _move_all_to_device(self):
        device = self.cfg.device
        # 1) TrainerMultiTask 속성 순회
        for name, obj in list(self.__dict__.items()):
            # 서브 모듈(nn.Module)인 경우
            if isinstance(obj, nn.Module):
                obj.to(self.cfg.device)
            # 파라미터(nn.Parameter)인 경우: leaf 유지하며 이동
            elif isinstance(obj, nn.Parameter):
                obj.data = obj.data.to(self.cfg.device)

    # ---------- forward ----------
    def forward(self, batch):
        d = self.cfg.device

        # 1) IMU + length
        imu = batch["imu_motion"].to(d)                           # (B,T,D_imu)
        imu_len = (imu.abs().sum(-1) > 0).sum(1)                   # (B,)

        # 2) VEH + mask
        veh  = batch["veh_motion"].to(d)                          # (B,T,1)
        mask = batch["veh_mask_motion"].to(d).bool()              # (B,T) or (B,1,T)

        # 3) encode → (B,H,T)
        feat = self.encoder(imu, imu_len, veh, mask)

        # 4) head → (B,T,C)
        out  = self.head(feat)
        return out

    # ---------- 1 epoch ----------
    def run_epoch(self, loader, train=True):
        self.encoder.train(train)
        self.head.train(train)
        iterator = tqdm(loader, desc="Train" if train else "Eval")
        total_loss, total_frames = 0.0, 0

        for batch in iterator:
            # 1) 원본 raw 레이블 (0은 패딩/무시, 1~4가 실제 클래스)
            raw = batch["labels"]["label_motion"].long().to(self.cfg.device)  # shape (B,T)

            # 2) valid mask: raw>0 인 프레임만 학습에 사용
            valid = (raw > 0)

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
                loss = torch.tensor(0., device=self.cfg.device)

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
            logits = self.forward(batch)               # (B,T,C)
            p      = logits.argmax(-1).cpu()           # (B,T)
            t      = (batch["labels"]["label_motion"].cpu() - 1)  # (B,T)
            m      = t != self.cfg.ign_mot

            preds.append(p[m]); trues.append(t[m])

        if not preds:
            return 0.0

        y_p = torch.cat(preds)
        y_t = torch.cat(trues)
        return accuracy_score(y_t, y_p)

    # ---------- train loop ----------
    def train(self):
        tr_loader = self.make_loader(self.train_keys, True)
        va_loader = self.make_loader(self.val_keys,   False)

        best_loss, patience = float('inf'), 0
        for epoch in range(1, self.cfg.epochs+1):
            tr_loss = self.run_epoch(tr_loader, True)
            va_loss = self.run_epoch(va_loader, False)
            va_acc  = self.evaluate("val")

            print(f"Epoch {epoch:02d} | L_tr {tr_loss:.4f}  L_val {va_loss:.4f}  Acc {va_acc:.3f}")

            if va_loss < best_loss:
                best_loss, patience = va_loss, 0
                torch.save({
                    "encoder": self.encoder.state_dict(),
                    "head":    self.head.state_dict(),
                    "optim":   self.optim.state_dict(),
                    "scaler":  self.scaler.state_dict()
                }, "weights/best_motion.pt")
            else:
                patience += 1
                if patience >= self.cfg.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # --- 최종 검증/테스트 ---
        # best ckpt 로드
        ckpt = torch.load("weights/best_motion.pt", map_location=self.cfg.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.head.load_state_dict(ckpt["head"])
        print(">>> FINAL VAL  ACC =", self.evaluate("val"))
        print(">>> FINAL TEST ACC =", self.evaluate("test"))


if __name__ == "__main__":
    trainer = MotionTrainer(Config())
    trainer.train()