import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import dataProcessor
from torch.utils.data import DataLoader

from models.encoder.imu_encoder import IMUFeatureEncoder
from data.uni_dataset import UnimodalMotionDataset, unimodal_collate

class UniConformerMotion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.imu_enc    = IMUFeatureEncoder(cfg)
        self.proj_imu   = nn.Linear(cfg.imu_params['encoder_dim'], cfg.hidden)
        self.mode_emb   = nn.Embedding(2, cfg.hidden) # mode = 2
        self.classifier = nn.Linear(cfg.hidden, cfg.num_motion)

    def forward(self, imu, mode_idx):
        imu_len = (imu.abs().sum(-1)>0).sum(1)
        h_imu   = self.imu_enc(imu, imu_len).mean(1)
        h_imu   = self.proj_imu(h_imu)
        h_mode  = self.mode_emb(mode_idx)
        return self.classifier(h_imu + h_mode)


def train_unimodal(cfg):
    # --- 데이터 로드(기존 _prepare_data() 활용) ---
    trainer = dataProcessor(cfg)
    trainer._prepare_data()
    trainer._debug_dataset()
    # unimodal 전용 loader
    train_loader = DataLoader(
        UnimodalMotionDataset(
            participant_ids=trainer.train_keys,
            mode=cfg.mot_mode,
            data_map=trainer.data_map,
            imu_cols=trainer.imu_cols,
            ppg_cols=trainer.ppg_cols,       # 추가
            sc_cols=trainer.sc_cols,         # 추가
            veh_cols=trainer.veh_cols,
            label_cols=trainer.label_cols,
            window_sec=cfg.window_sec_mot,
            window_stride=cfg.window_stride_mot,
            fs=cfg.fs,
        ),
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=lambda b: unimodal_collate(b, cfg)
    )
    val_loader = DataLoader(
        UnimodalMotionDataset(
            participant_ids=trainer.val_keys,
            mode=cfg.mot_mode,
            data_map=trainer.data_map,
            imu_cols=trainer.imu_cols,
            ppg_cols=trainer.ppg_cols,       # 추가
            sc_cols=trainer.sc_cols,         # 추가
            veh_cols=trainer.veh_cols,
            label_cols=trainer.label_cols,
            window_sec=cfg.window_sec_mot,
            window_stride=cfg.window_stride_mot,
            fs=cfg.fs,
        ),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=lambda b: unimodal_collate(b, cfg)
    )

    # --- 모델/옵티마이저/손실 ---
    model     = UniConformerMotion(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.ign_mot)

    # --- pre-train loop ---
    best_acc = 0.0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        total_loss = count = 0
        for batch in train_loader:
            imu   = batch['imu'].to(cfg.device)                    # (B,T,imu_dim)
            mode  = batch['mode'].to(cfg.device)                   # (B,)
            tgt   = (batch['label_motion'] - 1).clamp(min=0).to(cfg.device)  # (B,)

            # --- filter out mode == -100 frames ---
            valid = mode >= 0
            if not valid.any():
                continue
            imu  = imu[valid]
            mode = mode[valid]
            tgt  = tgt[valid]   # 이제 같은 디바이스이므로 바로 인덱싱 가능

            logits = model(imu, mode)
            loss   = criterion(logits, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        # 검증
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                imu   = batch['imu'].to(cfg.device)
                mode  = batch['mode'].to(cfg.device)
                tgt   = (batch['label_motion'] - 1).clamp(min=0).cpu().numpy()

                pred  = model(imu, mode).argmax(-1).cpu().numpy()
                correct += (pred == tgt).sum()
                total   += len(tgt)

        acc = correct / total
        print(f"[Pretrain] Ep{ep:02d} loss={total_loss/count:.4f}  val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "weights/unimodal_best.pt")