import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")  # 서버 환경에서 그림 저장용
import matplotlib.pyplot as plt
from utils.losses import FocalCrossEntropy


# ---------------------------
# 공통 유틸
# ---------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def _pool_logits_if_seq2one(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    라벨이 (B,) or (B,1) 인데 logits가 (B,T,C)라면, T에 대해 평균해 (B,C)로 맞춘다.
    그 외엔 그대로 반환.
    """
    if logits.dim() == 3 and targets.dim() == 1:
        return logits.mean(dim=1)
    if logits.dim() == 3 and targets.dim() == 2 and targets.size(1) == 1:
        return logits.mean(dim=1)
    return logits

@torch.no_grad()
def _confmat_from_logits(logits: torch.Tensor,
                         targets: torch.Tensor,
                         num_classes: int,
                         ignore_index: int = -100) -> torch.Tensor:
    device = logits.device
    # ★ 시퀀스-투-원 맞춤
    logits = _pool_logits_if_seq2one(logits, targets)
    if logits.dim() == 3:
        B, T, C = logits.shape
        pred = logits.argmax(-1).reshape(-1)
        tgt  = targets.reshape(-1).to(device)
    else:
        C = logits.shape[-1]
        pred = logits.argmax(-1)
        tgt  = targets.to(device)
    mask = (tgt != ignore_index) & (tgt >= 0) & (tgt < C)
    if mask.sum() == 0:
        return torch.zeros(C, C, dtype=torch.long, device=device)
    cm = pred[mask] * C + tgt[mask]
    return torch.bincount(cm, minlength=C*C).reshape(C, C)

def _save_confmat_png(cm: torch.Tensor, out_path: str, title: str = "Confusion Matrix"):
    cm_np = cm.detach().cpu().numpy()
    plt.figure()
    plt.imshow(cm_np, interpolation='nearest')
    plt.title(title)
    plt.xlabel('Predicted'); plt.ylabel('Target')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def _save_loss_plot(history: dict, out_path: str, title: str = "Loss Curve"):
    """
    history: {'epoch': [...], 'train': [...], 'val': [...]}
    """
    plt.figure()
    plt.plot(history['epoch'], history['train'], label='Train')
    plt.plot(history['epoch'], history['val'],   label='Val')
    plt.title(title)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def _class_weights_from_loader(loader, num_classes, device):
    import torch
    counts = torch.zeros(num_classes, dtype=torch.float)
    for batch in loader:
        y = batch.get("label", batch.get("label_act"))
        if y is None: continue
        y = y.view(-1)
        mask = (y != -100)
        if mask.any():
            counts += torch.bincount(y[mask].long(), minlength=num_classes).float()
    w = counts.sum() / (counts + 1e-6)
    w = (w / w.mean().clamp_min(1e-6)).to(device)
    return w, counts.cpu()

def _get_label_from_batch(batch, primary, fallback=None):
    if primary in batch:
        return batch[primary]
    if fallback and fallback in batch:
        return batch[fallback]
    return None

def _assert_time_aligned(preds, targets, name="ACT"):
    """
    ACT(회귀)에서만 사용. preds/targets을 (B,T)로 정규화한 뒤 T가 다르면 경고만 출력.
    """
    if preds is None or targets is None:
        return
    # (B,T,1)->(B,T), (B,) -> (B,1)
    if preds.dim() == 3 and preds.size(-1) == 1:
        preds = preds.squeeze(-1)
    if targets.dim() == 3 and targets.size(-1) == 1:
        targets = targets.squeeze(-1)
    if preds.dim() == 1:
        preds = preds.unsqueeze(1)
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)
    if preds.dim()==2 and targets.dim()==2 and preds.size(1) != targets.size(1):
        print(f"[{name}] TIME MISMATCH: preds_T={preds.size(1)}, target_T={targets.size(1)}")

@torch.no_grad()
def evaluate_tot_metrics(model,
                         loader,
                         device,
                         ce_loss: torch.nn.Module = None,
                         pooling: str = "mean",
                         desc: str = "[EVAL] TOT"):
    """
    공통 TOT 평가 함수
    - 모델 출력: 텐서 또는 dict({'tot_logits' | 'logits' | 'out': ...}) 모두 지원
    - 라벨: (B,), (B,1), (B,T) 지원
    - logits: (B,C), (B,T,C) 지원
    - 시퀀스→원 과제일 때 시간축 풀링(pooling='mean'|'last'|'max')
    - ignore_index=-100 마스킹 적용
    반환: (평균 CE 로스, 정확도)
    """
    def _pick_logits(out):
        if isinstance(out, dict):
            for k in ("tot_logits", "logits", "out"):
                if k in out:
                    return out[k]
            raise ValueError("dict output has no 'tot_logits'/'logits'/'out'")
        elif torch.is_tensor(out):
            return out
        else:
            raise TypeError(f"Unexpected output type: {type(out)}")

    def _pool_seq_logits(logits: torch.Tensor, mode: str):
        if logits.dim() != 3:
            return logits
        if mode == "mean":
            return logits.mean(dim=1)          # (B,C)
        elif mode == "last":
            return logits[:, -1, :]            # (B,C)
        elif mode == "max":
            return logits.max(dim=1).values    # (B,C)
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")

    if ce_loss is None:
        ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)

    model.eval()
    total_loss, n_batches = 0.0, 0
    all_pred, all_true = [], []

    for batch in tqdm(loader, desc=desc):
        # move to device
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        out = model(batch)
        logits = _pick_logits(out)                  # (B,C) or (B,T,C)

        # 라벨 우선순위: TOT는 'label' 우선
        tgt = batch.get("label", batch.get("label_act", None))
        if tgt is None:
            continue

        # 시퀀스→원 과제: logits (B,T,C) 이고, 라벨이 (B,) 또는 (B,1)인 경우
        if logits.dim() == 3 and (tgt.dim() == 1 or (tgt.dim() == 2 and tgt.size(1) == 1)):
            logits_eval = _pool_seq_logits(logits, pooling)  # (B,C)
            targets_eval = tgt.long() if tgt.dim() == 1 else tgt.squeeze(1).long()
            loss = ce_loss(logits_eval, targets_eval)

            pred = logits_eval.argmax(-1)           # (B,)
            true = targets_eval                      # (B,)

            mask = (true != -100)
            if mask.any():
                total_loss += float(loss.item())
                n_batches += 1
                all_pred.append(pred[mask].detach().cpu())
                all_true.append(true[mask].detach().cpu())
            continue

        # 시퀀스→시퀀스: (B,T,C) vs (B,T)
        if logits.dim() == 3 and tgt.dim() == 2 and logits.size(1) == tgt.size(1):
            B, T, C = logits.shape
            loss = ce_loss(logits.reshape(B * T, C), tgt.reshape(B * T).long())

            pred = logits.argmax(-1).reshape(-1)    # (B*T,)
            true = tgt.reshape(-1)                   # (B*T,)

            mask = (true != -100)
            if mask.any():
                total_loss += float(loss.item())
                n_batches += 1
                all_pred.append(pred[mask].detach().cpu())
                all_true.append(true[mask].detach().cpu())
            continue

        # 배치 단일: (B,C) vs (B,)
        if logits.dim() == 2 and tgt.dim() == 1:
            loss = ce_loss(logits, tgt.long())

            pred = logits.argmax(-1)                # (B,)
            true = tgt                               # (B,)

            mask = (true != -100)
            if mask.any():
                total_loss += float(loss.item())
                n_batches += 1
                all_pred.append(pred[mask].detach().cpu())
                all_true.append(true[mask].detach().cpu())
            continue

        # 그 외 모양은 미스매치 → 스킵(로그만)
        print(f"[WARN] evaluate_tot_metrics shape mismatch: logits={tuple(logits.shape)}, tgt={tuple(tgt.shape)}")

    if n_batches == 0 or len(all_true) == 0:
        return float("nan"), 0.0

    import numpy as np
    from sklearn.metrics import accuracy_score
    all_pred = torch.cat(all_pred).numpy()
    all_true = torch.cat(all_true).numpy()
    acc = accuracy_score(all_true, all_pred)
    return (total_loss / n_batches), float(acc)

# ===========================
# Baseline Trainer (TOT/ACT)
# ===========================
class BaselineTrainer:
    """
    - TOT(분류): CrossEntropyLoss(ignore_index=-100), 혼동행렬/정확도 저장
    - ACT(회귀): MSELoss(reduction='none') + -100 마스킹, 라인플롯 저장 (혼동행렬은 생략)
    """
    def __init__(self, model, cfg, train_loader, val_loader, task_name,
                 save_dir: str = "outputs", num_classes: int = None):
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task_name = task_name.lower()  # 'tot' or 'act'
        self.device = cfg.Project.device
        self.model.to(self.device)

        self.save_dir = _ensure_dir(getattr(cfg.Project, "save_dir", save_dir))

        self.num_classes = num_classes if num_classes is not None else getattr(model, "num_classes", 10)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.BaselineTask.lr,
            weight_decay=cfg.BaselineTask.weight_decay
        )

        if self.task_name == 'tot':
            # ✨ train 로더로 클래스 분포 집계 → 가중치 산출
            w, cnt = _class_weights_from_loader(self.train_loader, self.num_classes, self.device)
            self.criterion = FocalCrossEntropy(weight=w, gamma = 2.0, ignore_index=-100)
            print(f"[TOT/Baseline] train class_counts={cnt.tolist()}  "
                f"class_weights={w.detach().cpu().numpy().round(3).tolist()}")
        else:  # ACT
            self.criterion = nn.MSELoss(reduction='none')


        self.history = {'epoch': [], 'train': [], 'val': []}

    def _extract_logits(self, out):
        if isinstance(out, dict):
            for k in ('tot_logits', 'logits', 'out', 'act_logits'):
                if k in out:
                    return out[k]
            raise ValueError("BaselineTrainer: dict output has no 'tot_logits'/'logits'/'out'/'act_logits'")
        elif torch.is_tensor(out):
            return out
        else:
            raise TypeError(f"BaselineTrainer: unexpected output type {type(out)}")

    def _loss_tot(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 3:
            B, T, C = logits.shape
            return self.criterion(logits.reshape(B*T, C),
                                targets.reshape(B*T).long())
        else:
            return self.criterion(logits, targets.long())


    def _loss_act(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        veh1_act 전용
        preds:   (B,T,1) 또는 (B,T)  # 모델은 (B,T,1)로 고정
        targets: (B,T,1) 또는 (B,T) 또는 (B,)
        -100 마스킹 후 MSE
        """
        preds = preds.to(self.device).float()
        targets = targets.to(self.device).float()

        # (B,T,1) -> (B,T); (B,T,3) 같은 구버전 출력이면 첫 채널만 사용 (안전 가드)
        if preds.dim() == 3:
            if preds.size(-1) == 1:
                preds_use = preds.squeeze(-1)        # (B,T)
            else:
                preds_use = preds[..., 0]            # ★ 구버전 대비 안전 가드
        else:
            preds_use = preds                        # (B,T) or (B,)

        # targets: (B,T,1)->(B,T)
        if targets.dim() == 3 and targets.size(-1) == 1:
            targets = targets.squeeze(-1)
        # (B,)가 오면 (B,1)로 확장해 브로드캐스트 오인 방지
        if targets.dim() == 1 and preds_use.dim() == 2 and preds_use.size(0) == targets.size(0):
            targets = targets.unsqueeze(1)

        # 시간축 길이 정렬(예방): 다르면 최소길이로 crop
        if preds_use.dim() == 2 and targets.dim() == 2:
            Tp, Tt = preds_use.size(1), targets.size(1)
            if Tp != Tt:
                T = min(Tp, Tt)
                preds_use = preds_use[:, :T]
                targets   = targets[:, :T]

        p_flat = preds_use.reshape(-1)
        t_flat = targets.reshape(-1)

        mask = (t_flat != -100.0) & torch.isfinite(t_flat) & torch.isfinite(p_flat)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return ((p_flat[mask] - t_flat[mask]) ** 2).mean()

    def _calculate_loss(self, preds, batch):
        if self.task_name == 'tot':
            logits = self._extract_logits(preds)        # ★ 추가
            tgt = batch['label']
            return self._loss_tot(logits, tgt)
        else:  # ACT
            tgt = batch['label']
            return self._loss_act(preds, tgt)           # ACT는 원래대로(회귀 텐서)

    def _run_epoch(self, loader, is_train: bool) -> float:
        self.model.train(is_train)
        total_loss, n_used = 0.0, 0
        pbar = tqdm(loader, desc=f"[{'Train' if is_train else 'Valid'}] {self.task_name.upper()}-Baseline")

        for batch in pbar:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            with torch.set_grad_enabled(is_train):
                preds = self.model(batch)
                loss = self._calculate_loss(preds, batch)

            # NaN/Inf 가드
            if not torch.isfinite(loss):
                pbar.set_postfix_str("skip(non-finite loss)")
                continue

            if is_train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += float(loss.item())
            n_used += 1
            pbar.set_postfix(loss=float(loss.item()))

        return (total_loss / n_used) if n_used else float('nan')

    @torch.no_grad()
    def _evaluate_val_acc_and_cm(self):
        if self.task_name != 'tot':
            return 0.0, torch.zeros(self.num_classes, self.num_classes, dtype=torch.long, device=self.device)

        self.model.eval()
        all_preds, all_trues = [], []
        cm_total = None

        for batch in self.val_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            out = self.model(batch)
            logits = self._extract_logits(out)           # ★ 변경
            tgt = batch['label']
            C = logits.shape[-1]
            cm_cur = _confmat_from_logits(logits, tgt, num_classes=C, ignore_index=-100)
            if cm_total is None:
                cm_total = torch.zeros(C, C, dtype=torch.long, device=self.device)
            cm_total += cm_cur

            if logits.dim() == 3:
                pred = logits.argmax(-1).reshape(-1)
                t = tgt.reshape(-1)
            else:
                pred = logits.argmax(-1)
                t = tgt
            mask = (t != -100)
            if mask.any():
                all_preds.append(pred[mask].detach().cpu())
                all_trues.append(t[mask].detach().cpu())

        if len(all_trues) == 0:
            if cm_total is None:
                cm_total = torch.zeros(1, 1, dtype=torch.long, device=self.device)
            return 0.0, cm_total

        all_preds = torch.cat(all_preds).numpy()
        all_trues = torch.cat(all_trues).numpy()
        acc = accuracy_score(all_trues, all_preds)
        return acc, cm_total


    
    def train(self, test_loader=None, save_name=None):
        print(f"--- Start Baseline Training for {self.task_name.upper()} ---")
        best_val_loss = float('inf')
        best_path = os.path.join("weights", f"best_baseline_{self.task_name}.pt") if save_name is None else save_name
        _ensure_dir(os.path.dirname(best_path))

        for epoch in range(1, self.cfg.BaselineTask.epochs + 1):
            tr = self._run_epoch(self.train_loader, is_train=True)
            va = self._run_epoch(self.val_loader,   is_train=False)

            # TOT는 정확도도 로그
            val_acc = 0.0
            if self.task_name == 'tot':
                val_acc, _ = self._evaluate_val_acc_and_cm()

            self.history['epoch'].append(epoch)
            self.history['train'].append(tr if tr == tr else float('nan'))
            self.history['val'].append(va if va == va else float('nan'))

            print(f"Epoch {epoch:02d} | Train Loss: {tr:.4f} | Val Loss: {va:.4f} | Val Acc: {val_acc:.3f}")

            # 베스트 갱신 시: 가중치 저장 + (TOT만) 검증 혼동행렬/손실곡선 저장
            if va < best_val_loss:
                best_val_loss = va
                torch.save(self.model.state_dict(), best_path)
                print(f"  -> [Baseline] Best model saved with Val Loss: {best_val_loss:.4f}")
        
        print(f"--- Finished Baseline Training for {self.task_name.upper()} ---")


# ===========================
# Enhancer Trainer (TOT/ACT)
#  - fusion_head만 학습
#  - 베스트 에폭/테스트에서 저장
# ===========================
class EnhancerTrainer:
    def __init__(self, enhancer_model, cfg, train_loader, val_loader, task_name,
                 save_dir: str = "outputs", num_classes: int = None):
        self.model = enhancer_model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task_name = task_name.lower()
        self.device = cfg.Project.device
        self.model.to(self.device)

        self.save_dir = _ensure_dir(getattr(cfg.Project, "save_dir", save_dir))
        self.num_classes = num_classes if num_classes is not None else getattr(enhancer_model, "num_classes", 10)

        # fusion_head만 학습
        self.optimizer = torch.optim.Adam(
            self.model.fusion_head.parameters(),
            lr=cfg.EnhancerTask.lr,
            weight_decay=cfg.EnhancerTask.weight_decay
        )

        if self.task_name == 'tot':
            w, cnt = _class_weights_from_loader(self.train_loader, self.num_classes, self.device)
            self.criterion = FocalCrossEntropy(weight=w, gamma = 2.0, ignore_index=-100)
            print(f"[TOT/Enhancer] train class_counts={cnt.tolist()}  "
                f"class_weights={w.detach().cpu().numpy().round(3).tolist()}")
        else:
            self.mse_none = nn.MSELoss(reduction='none')

        self.history = {'epoch': [], 'train': [], 'val': []}

    def _extract_logits_from_out(self, out):
        """
        TOT용: 분류 로그릿 추출
        우선순위: 'tot_logits' -> 'logits' -> 'out'
        """
        if isinstance(out, dict):
            for k in ('tot_logits', 'logits', 'out'):
                if k in out:
                    return out[k]
            raise ValueError("dict output has no 'tot_logits'/'logits'/'out'")
        elif torch.is_tensor(out):
            return out
        else:
            raise TypeError(f"Unexpected output type: {type(out)}")


    def _extract_preds_from_out(self, out):
        """
        ACT용: 회귀 예측 추출
        우선순위: 'act_preds' -> 'act_logits' -> 'logits' -> 'out'
        """
        if isinstance(out, dict):
            for k in ('act_preds', 'act_logits', 'logits', 'out'):
                if k in out:
                    return out[k]
            raise ValueError("dict output has no 'act_preds'/'act_logits'/'logits'/'out'")
        elif torch.is_tensor(out):
            return out
        else:
            raise TypeError(f"Unexpected output type: {type(out)}")

    
    # 1) _compute_loss_tot
    def _compute_loss_tot(self, out: dict, batch: dict) -> torch.Tensor:
        logits = self._extract_logits_from_out(out)   # (B,C) or (B,T,C)

        tgt = batch.get('label', batch.get('label_act', None))
        if tgt is None:
            raise KeyError("TOT labels missing: expected 'label' (or fallback 'label_act') in batch.")
        targets = tgt.to(logits.device).long()

        if logits.dim() == 3:
            B, T, C = logits.shape
            return self.ce(logits.reshape(B*T, C), targets.reshape(B*T))
        elif logits.dim() == 2:
            return self.ce(logits, targets)
        else:
            raise ValueError(f"Unexpected logits dim: {tuple(logits.shape)}")


    def _compute_loss_act(self, out: dict, batch: dict) -> torch.Tensor:
        preds = self._extract_preds_from_out(out)  # dict/tensor 모두 OK
        tgt = batch.get('label_act', batch.get('label', None))
        if tgt is None:
            raise KeyError("ACT labels missing: expected 'label_act' or 'label'")

        preds = preds.to(self.device).float()
        tgt   = tgt.to(self.device).float()

        # (B,T,1)->(B,T); (B,T,3) 구버전이면 첫 채널만 사용
        if preds.dim() == 3:
            if preds.size(-1) == 1:
                preds_use = preds.squeeze(-1)
            else:
                preds_use = preds[..., 0]
        else:
            preds_use = preds

        if tgt.dim() == 3 and tgt.size(-1) == 1:
            tgt = tgt.squeeze(-1)
        if tgt.dim() == 1 and preds_use.dim() == 2 and preds_use.size(0) == tgt.size(0):
            tgt = tgt.unsqueeze(1)

        # 시간축 길이 정렬(예방)
        if preds_use.dim() == 2 and tgt.dim() == 2:
            Tp, Tt = preds_use.size(1), tgt.size(1)
            if Tp != Tt:
                T = min(Tp, Tt)
                preds_use = preds_use[:, :T]
                tgt       = tgt[:, :T]

        p_flat = preds_use.reshape(-1)
        t_flat = tgt.reshape(-1)
        mask = (t_flat != -100.0) & torch.isfinite(t_flat) & torch.isfinite(p_flat)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return ((p_flat[mask] - t_flat[mask]) ** 2).mean()


    def _compute_loss(self, out, batch) -> torch.Tensor:
        if self.task_name == 'tot':
            return self._compute_loss_tot(out, batch)
        else:
            return self._compute_loss_act(out, batch)


    def _run_epoch(self, loader, is_train: bool) -> float:
        # 전문가(backbone)들은 freeze 되어 있다고 가정, 모드 전환은 fusion_head 기준으로만
        self.model.fusion_head.train(is_train)
        total_loss, n_used = 0.0, 0
        pbar = tqdm(loader, desc=f"[{'Train' if is_train else 'Valid'}] Enhancer-{self.task_name.upper()}")

        for batch in pbar:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            with torch.set_grad_enabled(is_train):
                out = self.model(batch)

                if self.task_name == 'act':
                    preds_for_check = out if torch.is_tensor(out) else (out.get('act_preds') or out.get('logits') or out.get('out'))
                    tgt_for_check   = _get_label_from_batch(batch, 'label_act', 'label')
                    _assert_time_aligned(preds_for_check, tgt_for_check, name="ACT")

                loss = self._compute_loss(out, batch)

            if not torch.isfinite(loss):
                pbar.set_postfix_str("skip(non-finite loss)")
                continue

            if is_train and loss.requires_grad:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.fusion_head.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += float(loss.item())
            n_used += 1

        return (total_loss / n_used) if n_used else float('nan')

    @torch.no_grad()
    def _evaluate_val_confmat_if_tot(self):
        if self.task_name != 'tot':
            return None
        self.model.fusion_head.eval()
        cm_total = None
        for batch in self.val_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            out = self.model(batch)
            logits = self._extract_logits_from_out(out)

            # 2) _evaluate_val_confmat_if_tot
            tgt = batch.get('label', batch.get('label_act', None))  # label 우선

            if tgt is None:
                continue
            
            logits = _pool_logits_if_seq2one(logits, tgt)
            C = logits.shape[-1]
            cm_cur = _confmat_from_logits(logits, tgt, num_classes=C, ignore_index=-100)
            if cm_total is None:
                cm_total = torch.zeros(C, C, dtype=torch.long, device=self.device)
            cm_total += cm_cur
        return cm_total

 

    def train(self, save_path, test_loader=None):
        print(f"--- Start Enhancer Training for {self.task_name.upper()} ---")
        _ensure_dir(os.path.dirname(save_path))
        best_val = float('inf')

        for epoch in range(1, self.cfg.EnhancerTask.epochs + 1):
            tr = self._run_epoch(self.train_loader, is_train=True)
            va = self._run_epoch(self.val_loader,   is_train=False)

            self.history['epoch'].append(epoch)
            self.history['train'].append(tr if tr == tr else float('nan'))
            self.history['val'].append(va if va == va else float('nan'))
            print(f"Epoch {epoch:02d} | Train Loss: {tr:.4f} | Val Loss: {va:.4f}")

            if va < best_val:
                best_val = va
                torch.save(self.model.state_dict(), save_path)
                print(f"  -> [Enhancer] Best model saved with Val Loss: {best_val:.4f}")


        print(f"--- Finished Enhancer Training for {self.task_name.upper()} ---")
