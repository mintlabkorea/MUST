import os
import torch
import torch.nn as nn
import numpy as np
import itertools
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.amp import GradScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
# from models.encoder.ppg_lstm_encoder import PPGEncoder
from models.encoder.imu_encoder import IMUFeatureEncoder
from models.encoder.veh_encoder import VehicleTCNEncoder
from models.encoder.ms_tcn2 import MS_TCN2_PG
# from models.encoder.sc_encoder import ScenarioEmbedding
# from models.encoder.survey_encoder import PreSurveyEncoder

from data.code.pkl_dataloader import PKLMultiModalDataset
from data.code.collate_fn_mod import collate_fn_unified

# from models.fusion.mot_predictor_mumu import MotionPredictor
# from models.fusion.emo_predictor_mumu   import EmotionPredictor, EmotionTemporalClassifier
# from models.fusion.fusion_block   import SMFusionBlock,GMFusionBlock

# from utils.losses import compute_losses, masked_mse
# from utils.visualization import visualize

from trainers.base_trainer import TrainerBase, dataProcessor

class TrainerMotionOnly(TrainerBase, dataProcessor):
    def __init__(self, cfg):
        self.cfg = cfg
        self._prepare_data()
        self.cfg.veh_params['input_dim'] = 1
        self._debug_dataset() 
        self._build_models()
        self._move_all_to_device()
        self._create_optimizer()

        os.makedirs('weights', exist_ok=True)
        os.makedirs('results', exist_ok=True)

    def _prepare_data(self):
        super()._prepare_data()
        # 모든 veh_cols 는 그대로 두고…
        # motion 전용으로 mode만 뽑는 리스트를 추가
        self.veh_mode_cols = [c for c in self.veh_cols if 'mode' in c]

    def make_motion_loader(self, keys, shuffle):
        # dataset_kwargs 복사
        kw = dict(self.dataset_kwargs)
        # 여기서 veh_cols 만 mode 컬럼으로 교체
        kw['veh_cols'] = self.veh_mode_cols

        return DataLoader(
            PKLMultiModalDataset(
                participant_ids=keys,
                mode=self.cfg.mot_mode,
                **kw
            ),
            batch_size=self.cfg.batch_size, shuffle=shuffle,
            num_workers=self.cfg.num_workers, pin_memory=True,
            collate_fn=lambda b: collate_fn_unified(
                b,
                veh_dim=len(self.veh_mode_cols),  # 1
                seq_len=self.cfg.seq_len,
                win_samp=int(self.cfg.window_sec_mot*self.cfg.fs)
            )
        )


    def _build_models(self):
        # ── 1) 인코더 & 프로젝션 2개만 ───────────────────────────
        self.nets = nn.ModuleDict({
            'imu': IMUFeatureEncoder(self.cfg),
            # 여기서만 in_channels_override=1 전달
            'veh': VehicleTCNEncoder(self.cfg, in_channels_override=1)
        })
        self.projs = nn.ModuleDict({
            'imu': nn.Linear(self.cfg.imu_params['encoder_dim'], self.cfg.hidden),
            'veh': nn.Linear(self.cfg.veh_params['embed_dim'],  self.cfg.hidden)
        })
        # ── 2) 모션 전용 Linear head ─────────────────────────────
        # Frame-wise prediction head: Conv1d(hidden → num_motion) 
        # 입력 시퀀스 (B, T, hidden) → permute → (B, hidden, T)
        self.motion_head = nn.Conv1d(self.cfg.hidden, 
                                    self.cfg.num_motion,
                                    kernel_size=1)

        device = self.cfg.device
        # ─── 3) scaler & loss 정의 ─────────────────────────────────────
        self.scaler = GradScaler()
        self.ce_m   = nn.CrossEntropyLoss(
            weight=torch.tensor(self.cfg.motion_weights,
                                 device=device),
            ignore_index=self.cfg.ign_mot)
    
    def forward_motion_only(self, batch):
        d = self.cfg.device
        # 1) IMU
        imu = batch['imu_motion'].to(d)
        imu_len = (imu.abs().sum(-1) > 0).sum(1)
        # Conformer 출력 (B, T, encoder_dim)
        imu_seq = self.nets['imu'](imu, imu_len)
        # → projection to hidden space (B, T, hidden)
        imu_seq = self.projs['imu'](imu_seq)
        # proj if needed, but we assume encoder_dim == hidden
        imu_emb = imu_seq 
  
        # 2) VEH (mode 한 컬럼뿐)
        veh = batch['veh_motion'].to(d)
        veh_mask = batch['veh_mask_motion'].to(d).bool()
        veh_emb = self.projs['veh'](
            self.nets['veh'](veh.transpose(1,2), None,
                             return_pooled=True, mask=veh_mask)
        )
        # repeat veh_emb along time
        B, T, H = imu_emb.shape
        veh_seq = veh_emb.unsqueeze(2).expand(-1, -1, T)  # (B, H, T) or .permute

        # 3) 단순 합으로 fusion
        # imu_emb: (B, T, H) → (B, H, T), veh_seq already (B, H, T)
        fused = imu_emb.permute(0,2,1) + veh_seq         # (B, H, T)

        # 4) framewise logits
        # motion_head is Conv1d(hidden→num_motion)
        logits = self.motion_head(fused)                  # (B, C, T)
        # permute back to (B, T, C)
        return logits.permute(0,2,1)                      # (B, T, C)

    def _create_optimizer(self):
        params = itertools.chain(
            self.nets['imu'].parameters(),
            self.nets['veh'].parameters(),
            self.projs['imu'].parameters(),
            self.projs['veh'].parameters(),
            self.motion_head.parameters()
        )
        self.optim = torch.optim.Adam(params, lr=self.cfg.lr)

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
        # self.beta.to(device)
        # # 2) 만약 dict 형식으로 관리하는 모듈(nets, projs)도
        # for d in ('nets', 'projs'):
        #     if hasattr(self, d):
        #         container = getattr(self, d)
        #         for k, module in container.items():
        #             if isinstance(module, nn.Module):
        #                 container[k] = module.to(device)

    def run_motion_epoch(self, loader, train: bool):
        """Motion 전용 epoch 루프 (frame-wise)"""
        # train / eval 모드 설정
        for m in [*self.nets.values(), *self.projs.values(), self.motion_head]:
            m.train() if train else m.eval()
        iterator = tqdm(loader, desc="Train Motion") if train else loader

        total_loss, total_frames = 0.0, 0
        for batch in iterator:
            # 1) Prepare targets & mask: (B, T)
            tgt_seq = batch['labels']['label_motion'].long()      # 원래 1-based labels
            tgt0    = (tgt_seq - 1).clamp(0, self.cfg.num_motion-1).to(self.cfg.device)
            mask    = (tgt_seq != self.cfg.ign_mot).to(self.cfg.device)  # ignore-marked frames

            # 2) Forward: (B, T, C)
            logits = self.forward_motion_only(batch).to(self.cfg.device)

            # 3) Flatten for loss: (B*T, C) vs (B*T,)
            B, T, C = logits.shape
            logits_flat = logits.reshape(-1, C)
            tgt_flat    = tgt0.reshape(-1)
            mask_flat   = mask.reshape(-1)

            # 4) Compute loss only on valid frames
            if mask_flat.any():
                loss = self.ce_m(logits_flat[mask_flat], tgt_flat[mask_flat])
            else:
                loss = torch.tensor(0., device=self.cfg.device)

            # 5) Backward & step
            if train:
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(
                        *[m.parameters() for m in [*self.nets.values(),
                                                *self.projs.values()]],
                        self.motion_head.parameters()
                    ), max_norm=1.0
                )
                self.scaler.step(self.optim)
                self.scaler.update()

            total_loss   += loss.item() * mask_flat.sum().item()  # frame 수 곱해 합산
            total_frames += mask_flat.sum().item()

        avg_loss = total_loss / max(total_frames, 1)
        return avg_loss


        
    def evaluate_motion(self, loader, split_name="test"):
        """
        Evaluate motion performance on a DataLoader.

        Args:
            loader (DataLoader): DataLoader providing batches.
            split_name (str): 'val' or 'test' 등, 파일명에 사용할 문자열.

        Returns:
            dict: {'mot_acc': accuracy}
        """
        # 1) 전체 예측/실제 프레임별로 모아서
        all_preds, all_trues = [], []
        with torch.no_grad():
            for batch in loader:
                logits = self.forward_motion_only(batch)               # (B, T, C)
                preds  = logits.argmax(-1).cpu().numpy()               # (B, T)
                raw_m  = batch['labels']['label_motion'].cpu().numpy() # (B, T)
                trues  = raw_m.astype(int) - 1                         # (B, T)
                # mask out ignore_index
                mask   = (raw_m != self.cfg.ign_mot)
                # flatten and collect only valid frames
                all_preds.append(preds[mask])
                all_trues.append(trues[mask])
        
        
        if not all_preds:
            print(f"[{split_name}] 유효 프레임이 없습니다. mot_acc=0.0 반환")
            return {'mot_acc': 0.0}
        
        y_pred = np.concatenate(all_preds)  # 1D array of all valid frames
        y_true = np.concatenate(all_trues)

        # 2) Confusion Matrix
        cm = confusion_matrix(y_true, y_pred,
                              labels=list(range(self.cfg.num_motion)))
        # ticklabels: config에 정의가 있으면 쓰고, 없으면 숫자 문자열로 대체
        if hasattr(self.cfg, 'motion_labels'):
            labels = self.cfg.motion_labels
        else:
            labels = [str(i) for i in range(self.cfg.num_motion)]

        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels,
                    yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{split_name.upper()} Confusion Matrix")
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/{split_name}_confusion_matrix.png")
        plt.close()

        # 3) Frame-by-frame 시계열 플롯
        plt.figure(figsize=(12,4))
        frames = np.arange(len(y_true))
        plt.plot(frames, y_true, color='blue', label='True', linewidth=1)
        plt.plot(frames, y_pred, color='red',  label='Pred', linewidth=1, alpha=0.7)
        plt.xlabel("Window Index")
        plt.ylabel("Motion Label")
        plt.title(f"{split_name.upper()} True vs Pred Labels")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f"results/{split_name}_labels_timeseries.png")
        plt.close()

        # 4) 최종 Accuracy 리턴
        acc = accuracy_score(y_true, y_pred)
        print(f"[{split_name}] MOT_ACC = {acc:.4f}  (plots saved in results/)")
        return {'mot_acc': acc}



    def evaluate_motion_only(self, split_name: str):
        # split_name 에 맞춰 loader를 만들고, split_name도 넘겨줍니다.
        keys = getattr(self, f"{split_name}_keys")
        loader = self.make_motion_loader(keys, False)
        return self.evaluate_motion(loader, split_name)


    def _get_state(self):
        return {
            'nets':        {k:m.state_dict() for k,m in self.nets.items()},
            'projs':       {k:m.state_dict() for k,m in self.projs.items()},
            'motion_head': self.motion_head.state_dict(),
            'optim':       self.optim.state_dict(),
            'scaler':      self.scaler.state_dict(),
            'epoch':       getattr(self, 'current_epoch', None),
        }

    def _load_state(self, path: str):
        """
        Override base _load_state to match motion_head naming.
        """
        ckpt = torch.load(path, map_location=self.cfg.device)
        # 1) nets
        for k, net in self.nets.items():
            net.load_state_dict(ckpt['nets'][k])
        # 2) projs
        for k, proj in self.projs.items():
            proj.load_state_dict(ckpt['projs'][k])
        # 3) motion_head 대신 motion_predictor이 아니라 motion_head 로드
        self.motion_head.load_state_dict(ckpt['motion_head'])
        # 4) optim & scaler
        self.optim.load_state_dict(ckpt['optim'])
        self.scaler.load_state_dict(ckpt['scaler'])
        # 5) epoch (optional)
        self.current_epoch = ckpt.get('epoch', None)
        print(f"Loaded checkpoint from '{path}' (epoch {self.current_epoch})")


    def train(self, build_model: bool = True):
        mot_tr = self.make_motion_loader(self.train_keys, True)
        mot_va = self.make_motion_loader(self.val_keys,   False)
 
        best_metric = float('inf')
        epochs_no_improve = 0

        for ep in range(1, self.cfg.epochs+1):
            # 1) 한 epoch 학습
            tr_loss = self.run_motion_epoch(mot_tr, True)
            # 2) 검증
            va_loss = self.run_motion_epoch(mot_va, False)
            va_acc  = self.evaluate_motion_only('val')['mot_acc']

            print(f"Epoch{ep:02d} | MOT TrainL={tr_loss:.3f} ValL={va_loss:.3f} ValAcc={va_acc:.3f}")

            if va_loss < best_metric:
                best_metric = va_loss
                epochs_no_improve = 0
                torch.save(self._get_state(), 'weights/best_motion.pt')
                # 매 epoch마다 IMU 그룹 게이트 확인
                imu_enc = self.nets['imu']
                with torch.no_grad():
                    raw = imu_enc.group_weight.detach().cpu().numpy()
                    gated = F.softmax(imu_enc.group_weight, dim=0).cpu().numpy()
                print(f" → [Epoch{ep:02d}] raw_weights: {raw.round(3)}  gated: {gated.round(3)}")
                print(" → New best checkpoint!")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.cfg.patience:
                    print(f"Early stopping at epoch {ep}")
                    break

        # --- after training loop: load best and run one final full eval ---
        self._load_state('weights/best_motion.pt')
        print(">>> FINAL VALIDATION RESULTS")
        self.evaluate_motion_only('val')
        print(">>> FINAL TEST RESULTS")
        self.evaluate_motion_only('test')