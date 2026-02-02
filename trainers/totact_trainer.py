# trainers/totact_trainer.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")  # headless 환경
from utils.losses import FocalCrossEntropy


# ---------------------------
# 작은 유틸
# ---------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True); return path

def _get_cfg_tot(cfg):
    # cfg.TOT이 없으면 기본값 채움
    class _D:
        use_focal=True; focal_gamma=2.0
        class_weighting="auto"  # "auto" | "none" | list/ndarray/torch.Tensor
        ignore_index=-100
        eval_pooling="mean"     # "mean" | "last" | "max"
    return getattr(cfg, "TOT", _D)

def _pick_logits(out):
    if isinstance(out, dict):
        for k in ("tot_logits", "logits", "out"):
            if k in out: return out[k]
        raise KeyError("dict output has no 'tot_logits'/'logits'/'out'")
    elif torch.is_tensor(out):
        return out
    else:
        raise TypeError(f"Unexpected output type: {type(out)}")

def _pool_seq_logits(logits: torch.Tensor, targets: torch.Tensor, mode: str):
    # 시퀀스→원 과제면 시간축 풀링
    if logits.dim() != 3: return logits
    if targets.dim() == 1 or (targets.dim()==2 and targets.size(1)==1):
        if mode == "mean": return logits.mean(dim=1)
        if mode == "last": return logits[:, -1, :]
        if mode == "max":  return logits.max(dim=1).values
    return logits

def _valid_mask_labels(y, num_classes, ignore_index):
    return (y != ignore_index) & (y >= 0) & (y < num_classes)

def _class_weights_from_loader(loader, num_classes, device, ignore_index=-100):
    # 라벨 분포(0..C-1만 집계)로 역비중 가중치 산출
    counts = torch.zeros(num_classes, dtype=torch.float, device="cpu")
    for batch in loader:
        y = batch.get("label", None)
        if y is None: continue
        y = y.view(-1).long().cpu()
        mask = _valid_mask_labels(y, num_classes, ignore_index)
        if mask.any():
            bc = torch.bincount(y[mask], minlength=num_classes).float()
            counts += bc
    # 0회피 + 평균으로 정규화
    w = counts.sum() / (counts + 1e-6)
    w = (w / w.mean().clamp_min(1e-6)).to(device)
    return w, counts


# ---------------------------
# 평가 루틴(TOT 전용)
# ---------------------------
@torch.no_grad()
def evaluate_tot_metrics(model, loader, device, ce_loss=None, pooling="mean", desc="[EVAL] TOT"):
    if ce_loss is None:
        ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    model.eval()
    total_loss, n_batches = 0.0, 0
    all_pred, all_true = [], []

    for batch in tqdm(loader, desc=desc):
        for k, v in batch.items():
            if torch.is_tensor(v): batch[k] = v.to(device)

        logits = _pick_logits(model(batch))
        tgt = batch.get("label", None)
        if tgt is None: continue

        logits_eval = _pool_seq_logits(logits, tgt, pooling)
        if logits_eval.dim()==3 and tgt.dim()==2 and logits_eval.size(1)==tgt.size(1):
            # seq2seq
            B, T, C = logits_eval.shape
            loss = ce_loss(logits_eval.reshape(B*T, C), tgt.reshape(B*T).long())
            pred = logits_eval.argmax(-1).reshape(-1)
            true = tgt.reshape(-1)
        elif logits_eval.dim()==2 and tgt.dim()==1:
            # seq2one or plain
            loss = ce_loss(logits_eval, tgt.long())
            pred = logits_eval.argmax(-1)
            true = tgt
        else:
            # 기타 모양 미스매치는 스킵
            continue

        mask = (true != -100)
        if mask.any():
            total_loss += float(loss.item()); n_batches += 1
            all_pred.append(pred[mask].detach().cpu())
            all_true.append(true[mask].detach().cpu())

    if n_batches == 0 or len(all_true) == 0:
        return float("nan"), 0.0

    all_pred = torch.cat(all_pred).numpy()
    all_true = torch.cat(all_true).numpy()
    acc = accuracy_score(all_true, all_pred)
    return (total_loss / n_batches), float(acc)


# ===========================
# Baseline Trainer (TOT 전용)
#  - 모델은 선택한 모달리티로 logits만 내면 됨(dict/tensor OK)
#  - FocalCE, 클래스 가중치(auto/none/custom)
# ===========================
class TOTBaselineTrainer:
    def __init__(self, model, cfg, train_loader, val_loader, task_name="tot",
                 num_classes: int = 3, save_dir: str = "outputs"):
        if task_name.lower() != "tot":
            raise NotImplementedError("This simplified trainer supports TOT only.")
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = cfg.Project.device
        self.model.to(self.device)
        self.save_dir = _ensure_dir(getattr(cfg.Project, "save_dir", save_dir))

        TOT = _get_cfg_tot(cfg)
        self.ignore_index = TOT.ignore_index

        # 옵티마이저
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.BaselineTask.lr,
            weight_decay=cfg.BaselineTask.weight_decay
        )

        # 손실: focal vs CE + 가중치
        weight = None
        if TOT.class_weighting == "auto":
            weight, counts = _class_weights_from_loader(
                self.train_loader, self.num_classes, self.device, self.ignore_index
            )
            print(f"[TOT/Baseline] class_counts={counts.tolist()}  class_weights={weight.detach().cpu().numpy().round(3).tolist()}")
        elif TOT.class_weighting == "none":
            weight = None
        else:
            # 리스트/ndarray/tensor 지원
            w = torch.as_tensor(TOT.class_weighting, dtype=torch.float, device=self.device)
            assert w.numel()==self.num_classes, "len(class_weighting) must equal num_classes"
            weight = w
        
        if getattr(TOT, "use_focal", True):
            self.criterion = FocalCrossEntropy(weight=weight, gamma=TOT.focal_gamma,
                                            ignore_index=self.ignore_index)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=weight,
                                                ignore_index=self.ignore_index)
        
        self.history = {'epoch': [], 'train': [], 'val': []}

    def _loss_tot(self, logits, targets):
        if targets.dim() == 2 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        if logits.dim()==3:
            B,T,C = logits.shape
            return self.criterion(logits.reshape(B*T, C), targets.reshape(B*T).long())
        return self.criterion(logits, targets.long())

    def _run_epoch(self, loader, is_train: bool) -> float:
        self.model.train(is_train)
        total_loss, n_used = 0.0, 0
        pbar = tqdm(loader, desc=f"[{'Train' if is_train else 'Valid'}] TOT-Baseline")
        for batch in pbar:
            for k, v in batch.items():
                if torch.is_tensor(v): batch[k] = v.to(self.device)

            logits = _pick_logits(self.model(batch))
            tgt = batch['label']

            with torch.set_grad_enabled(is_train):
                loss = self._loss_tot(_pool_seq_logits(logits, tgt, _get_cfg_tot(self.cfg).eval_pooling), tgt)

            if not torch.isfinite(loss): continue
            if is_train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += float(loss.item()); n_used += 1
            pbar.set_postfix(loss=float(loss.item()))
        return (total_loss / n_used) if n_used else float('nan')

    def train(self, test_loader=None, save_name=None):
        print(f"--- Start Baseline Training (TOT) ---")
        best_val_loss = float('inf')
        best_path = os.path.join("weights", "best_baseline_tot.pt") if save_name is None else save_name
        _ensure_dir(os.path.dirname(best_path))

        for epoch in range(1, self.cfg.BaselineTask.epochs + 1):
            tr = self._run_epoch(self.train_loader, True)
            va = self._run_epoch(self.val_loader,   False)
            val_ce, val_acc = evaluate_tot_metrics(
                self.model, self.val_loader, self.device,
                ce_loss=self.criterion, pooling=_get_cfg_tot(self.cfg).eval_pooling,
                desc="[ValEval] TOT"
            )

            self.history['epoch'].append(epoch)
            self.history['train'].append(tr if tr==tr else float('nan'))
            self.history['val'].append(va if va==va else float('nan'))
            print(f"Epoch {epoch:02d} | Train {tr:.4f} | Val {va:.4f} | Val CE {val_ce:.4f} | Val Acc {val_acc:.3f}")

            if va < best_val_loss:
                best_val_loss = va
                torch.save(self.model.state_dict(), best_path)
                print(f"  -> [Baseline] Best model saved @ {best_path}")

        print(f"--- Finished Baseline Training (TOT) ---")


# ===========================
# Enhancer Trainer (스태킹용, TOT 전용)
#  - 컨텍스트 모델(모션/감정)의 최종 클래스 확률을 feature로 받는 모델 학습
#  - 모델이 dict/tensor logits만 내면 OK
#  - 모델에 fusion_head가 있으면 그 파라미터만 학습(없으면 전체 학습)
# ===========================
class TOTEnhancerTrainer:
    def __init__(self, model, cfg, train_loader, val_loader, task_name="tot",
                 num_classes: int = 3, save_dir: str = "outputs"):
        if task_name.lower() != "tot":
            raise NotImplementedError("This simplified trainer supports TOT only.")
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = cfg.Project.device
        self.model.to(self.device)
        self.save_dir = _ensure_dir(getattr(cfg.Project, "save_dir", save_dir))

        # --- 어떤 헤드를 학습할지 결정 ---
        learnable_module_name = None
        if hasattr(self.model, "fusion_head"):
            learnable_module_name = "fusion_head"
        elif hasattr(self.model, "stack_head"):
            learnable_module_name = "stack_head"

        for _, p in self.model.named_parameters():
            p.requires_grad = False

        # 결합 헤드만 학습
        if hasattr(self.model, "stack_head"):
            for _, p in self.model.stack_head.named_parameters():
                p.requires_grad = True
            train_mods = ["stack_head." + n for n,_ in self.model.stack_head.named_parameters()]
        elif hasattr(self.model, "fusion_head"):
            for _, p in self.model.fusion_head.named_parameters():
                p.requires_grad = True
            train_mods = ["fusion_head." + n for n,_ in self.model.fusion_head.named_parameters()]
        else:
            # 최소 안전장치
            for n,p in self.model.named_parameters():
                if "classifier" in n or "head" in n:
                    p.requires_grad = True
            train_mods = [n for n,p in self.model.named_parameters() if p.requires_grad]

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            params, lr=cfg.EnhancerTask.lr, weight_decay=cfg.EnhancerTask.weight_decay
        )
        print(f"[Enhancer] trainable params: {sum(p.numel() for p in params)}  modules={train_mods}")
        assert len(params) > 0, "No trainable params! 결합 헤드 이름 확인 필요(fusion_head/stack_head)."
        # class weight + focal/ce
        weight, counts = _class_weights_from_loader(self.train_loader, self.num_classes, self.device, -100)
        print(f"[TOT/Enhancer] class_counts={counts.tolist()}  class_weights={weight.detach().cpu().numpy().round(3).tolist()}")
        self.criterion = FocalCrossEntropy(weight=weight, gamma=_get_cfg_tot(cfg).focal_gamma, ignore_index=-100)

        self.history = {'epoch': [], 'train': [], 'val': []}

    def _ce_for(self, logits, targets):
        """pool → CE (targets와 시간축 맞춤)"""
        pooled = _pool_seq_logits(logits, targets, _get_cfg_tot(self.cfg).eval_pooling)
        if targets.dim()==2 and targets.size(1)==1: targets = targets.squeeze(1)
        if pooled.dim()==3:
            B,T,C = pooled.shape
            return self.criterion(pooled.reshape(B*T, C), targets.reshape(B*T).long())
        return self.criterion(pooled, targets.long())

    def _loss_tot(self, logits, targets):
        return self._ce_for(logits, targets)
    
    def _grad_norm(self):
        sq = 0.0
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                g = p.grad.detach()
                sq += g.pow(2).sum().item()
        return sq ** 0.5


    def _run_epoch(self, loader, is_train: bool) -> float:
        # 동결/학습 모드 설정
        if hasattr(self.model, "stack_head"):
            self.model.train(False)
            self.model.stack_head.train(is_train)
        elif hasattr(self.model, "fusion_head"):
            self.model.train(False)
            self.model.fusion_head.train(is_train)
        else:
            self.model.train(is_train)

        total_loss, n_used = 0.0, 0
        pbar = tqdm(loader, desc=f"[{'Train' if is_train else 'Valid'}] Enhancer-TOT")
        first_batch_debug_done = False

        for batch in pbar:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device)

            out = self.model(batch)
            logits = _pick_logits(out)                         # fused (기본 선택)
            tgt = batch['label']
            base_logits = out['base_logits'] if isinstance(out, dict) and ('base_logits' in out) else None
            
            # --- 선택적 퓨전 gating ---
            fusion_enabled = False
            if isinstance(out, dict) and ('fusion_enabled' in out):
                fusion_enabled = bool(out['fusion_enabled'])
            else:
                # 모델이 플래그를 안 주는 경우 배치 키로 보조 판정
                fusion_enabled = any(k in batch for k in (
                    'ppg_emotion', 'veh_emotion', 'scenario_evt_e', 'survey_e', 'imu_motion'
                ))

            # 퓨전 불가면 fused 대신 base를 사용하고, 최적화 스텝은 생략
            if (not fusion_enabled) and (base_logits is not None):
                logits_for_loss = base_logits
            else:
                logits_for_loss = logits

            with torch.set_grad_enabled(is_train):
                loss = self._loss_tot(logits_for_loss, tgt)

            # ---- 첫 배치 디버그 (루프 안으로 이동) ----
            if (not first_batch_debug_done) and (base_logits is not None):
                try:
                    ce_base  = float(self._ce_for(base_logits.detach(), tgt).item())
                    ce_fused = float(self._ce_for(logits, tgt).item())
                    delta    = ce_fused - ce_base
                    print(f"[DEBUG] CE(base)={ce_base:.4f}  CE(fused)={ce_fused:.4f}  Δ={delta:+.4f}")

                    # 분포 차이/크기
                    fb = _pool_seq_logits(logits, tgt, _get_cfg_tot(self.cfg).eval_pooling)
                    bb = _pool_seq_logits(base_logits, tgt, _get_cfg_tot(self.cfg).eval_pooling)
                    diff = (fb - bb).detach()
                    with torch.no_grad():
                        kl = F.kl_div(F.log_softmax(fb, dim=-1), F.softmax(bb, dim=-1), reduction="batchmean").item()
                    print(f"[DEBUG] ||fused-base||_2={diff.norm().item():.6f}  KL(f||b)={kl:.6e}")

                    if is_train and fusion_enabled:
                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward(retain_graph=True)
                        print(f"[DEBUG] grad_norm={self._grad_norm():.5f}")
                        self.optimizer.zero_grad(set_to_none=True)
                except Exception as e:
                    print(f"[DEBUG] first-batch debug skipped: {e}")
                first_batch_debug_done = True
            # ---------------------------------------

            if not torch.isfinite(loss):
                continue

            if is_train and fusion_enabled:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                self.optimizer.step()

            total_loss += float(loss.item()); n_used += 1
            if fusion_enabled:
                pbar.set_postfix(loss=float(loss.item()))
            else:
                pbar.set_postfix(loss=float(loss.item()), note="skip-update(no-emo)")
        return (total_loss / n_used) if n_used else float('nan')

    
    @torch.no_grad()
    def _eval_epoch_ceacc(self, loader):
        self.model.eval()
        ce_sum, n_batches = 0.0, 0
        correct_total, denom_total = 0, 0

        first_batch_debug_done = False

        for batch in loader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device)

            out   = self.model(batch)
            fused = out['logits']                         # 반드시 fused
            base  = out.get('base_logits', None)
            y     = batch['label'].long()

            # 1) CE: 여기서는 풀링을 내부에서 '한 번만' 수행
            ce = self._ce_for(fused, y)
            ce_sum += float(ce.item()); n_batches += 1

            # 2) ACC: 정확도 계산을 위해서만 풀링 결과 사용
            fused_pooled = _pool_seq_logits(fused, y, _get_cfg_tot(self.cfg).eval_pooling)
            # 레이블 차원 정규화: (B,1) → (B,), (B,)는 그대로
            if y.dim() == 2 and y.size(1) == 1:
                y_eval = y.squeeze(1)
            else:
                y_eval = y

            pred = fused_pooled.argmax(-1)
            mask = (y_eval != self.ignore_index)
            correct_total += (pred[mask] == y_eval[mask]).sum().item()
            denom_total   += mask.sum().item()

            # 3) 첫 배치 디버그: base vs fused CE 비교
            if (not first_batch_debug_done) and (base is not None):
                ce_base  = float(self._ce_for(base,  y).item())
                ce_fused = float(self._ce_for(fused, y).item())
                print(f"[DEBUG] first-batch CE: base={ce_base:.4f}, fused={ce_fused:.4f}, delta={ce_base - ce_fused:+.4f}")
                first_batch_debug_done = True

        ce_mean = ce_sum / max(n_batches, 1)
        acc = (correct_total / max(denom_total, 1)) if denom_total > 0 else 0.0
        return ce_mean, acc


    
    def _criterion(self, logits, y, class_weights=None):
        # 베이스라인과 동일 레시피 유지: focal/ce + class_weight
        if getattr(self.cfg.TOT, "loss_type", "focal") == "focal":
            gamma = getattr(self.cfg.TOT, "focal_gamma", 1.5)
            return FocalCrossEntropy(
                logits, y, class_weights, gamma=gamma,
                ignore_index=getattr(self.cfg.TOT, "ignore_index", -100)
            )
        else:
            return F.cross_entropy(
                logits, y, weight=class_weights,
                ignore_index=getattr(self.cfg.TOT, "ignore_index", -100)
            )


    def train(self, save_path, test_loader=None):
        print(f"--- Start Enhancer Training (TOT) ---")
        _ensure_dir(os.path.dirname(save_path))
        best_val = float('inf')

        # (선택) gate 워밍업/초깃값 — 필요 없으면 제거 가능
        sh = getattr(self.model, "stack_head", None)
        if sh is not None and hasattr(sh, "gate"):
            with torch.no_grad():
                sh.gate.data.fill_(float(getattr(self.cfg.TOT, "enh_gate_init", 0.10)))
            sh.gate.requires_grad = True

        for epoch in range(1, self.cfg.EnhancerTask.epochs + 1):
            tr = self._run_epoch(self.train_loader, True)
            va = self._run_epoch(self.val_loader,   False)

            val_ce, val_acc = evaluate_tot_metrics(
                self.model, self.val_loader, self.device,
                ce_loss=self.criterion, pooling=_get_cfg_tot(self.cfg).eval_pooling,
                desc="[ValEval] Enhancer-TOT"
            )

            self.history['epoch'].append(epoch)
            self.history['train'].append(tr if tr==tr else float('nan'))
            self.history['val'].append(va if va==va else float('nan'))
            print(f"Epoch {epoch:02d} | Train {tr:.4f} | Val {va:.4f} | Val CE {val_ce:.4f} | Val Acc {val_acc:.3f}")

            if val_ce < best_val:
                best_val = val_ce
                torch.save(self.model.state_dict(), save_path)
                print(f"  -> [Enhancer] Best model saved @ {save_path}")
        print(f"--- Finished Enhancer Training (TOT) ---")

# ===========================
# ACT 공용 유틸
# ===========================
def _pick_preds_act(out):
    """모델 출력에서 회귀 예측 텐서를 뽑아온다."""
    if isinstance(out, dict):
        for k in ('act_preds', 'logits', 'out'):
            if k in out:
                return out[k]
        raise KeyError("dict output has no 'act_preds'/'logits'/'out'")
    elif torch.is_tensor(out):
        return out
    else:
        raise TypeError(f"Unexpected output type: {type(out)}")

def _get_act_targets(batch):
    tgt = batch.get('label_act', batch.get('label', None))
    if tgt is None:
        raise KeyError("ACT labels missing: expected 'label_act' or 'label' in batch.")
    return tgt

def _mse_masked(preds: torch.Tensor,
                targets: torch.Tensor,
                ignore_index: float = -100.0) -> torch.Tensor:
    """
    preds:   (B,T,1) or (B,T) or (B,)   # 모델은 보통 (B,T,1)
    targets: (B,T,1) or (B,T) or (B,)   # 데이터셋 상황에 따라 다름
    -100 마스킹 후 MSE
    """
    preds = preds.float()
    targets = targets.float()

    # (B,T,1) -> (B,T)
    if preds.dim() == 3 and preds.size(-1) == 1:
        preds = preds.squeeze(-1)
    if targets.dim() == 3 and targets.size(-1) == 1:
        targets = targets.squeeze(-1)

    # (B,) vs (B,T) 방지
    if targets.dim() == 1 and preds.dim() == 2 and preds.size(0) == targets.size(0):
        targets = targets.unsqueeze(1)

    # 시간축 길이 다르면 최소 길이로 crop
    if preds.dim() == 2 and targets.dim() == 2 and preds.size(1) != targets.size(1):
        T = min(preds.size(1), targets.size(1))
        preds   = preds[:, :T]
        targets = targets[:, :T]

    p = preds.reshape(-1)
    t = targets.reshape(-1)
    mask = (t != float(ignore_index)) & torch.isfinite(t) & torch.isfinite(p)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
    return ((p[mask] - t[mask]) ** 2).mean()

@torch.no_grad()
def evaluate_act_metrics(model, loader, device, desc="[EVAL] ACT"):
    """
    마스크드 MSE/RMSE 평가.
    """
    model.eval()
    mse_sum, denom = 0.0, 0
    for batch in tqdm(loader, desc=desc):
        for k, v in batch.items():
            if torch.is_tensor(v): batch[k] = v.to(device)
        out   = model(batch)
        preds = _pick_preds_act(out)
        tgt   = _get_act_targets(batch)
        loss  = _mse_masked(preds, tgt, ignore_index=-100.0)
        if torch.isfinite(loss):
            # 평균 MSE 누적: 에폭 평균과 일치하도록 단순 평균
            mse_sum += float(loss.item())
            denom   += 1
    if denom == 0:
        return float("nan"), float("nan")
    mse = mse_sum / denom
    rmse = float(np.sqrt(mse))
    return mse, rmse


# ===========================
# ACT Baseline Trainer
# ===========================
class ACTBaselineTrainer:
    """
    - 모델 출력: dict('act_preds'| 'logits' | 'out') 또는 tensor
    - 손실: MSE + -100 마스크
    - 저장: weights/best_baseline_act.pt (기본)
    """
    def __init__(self, model, cfg, train_loader, val_loader, save_dir: str = "outputs"):
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = cfg.Project.device
        self.model.to(self.device)
        self.save_dir = _ensure_dir(getattr(cfg.Project, "save_dir", save_dir))

        # ignore_index: ACT 설정에 없으면 TOT 설정->기본 -100
        self.ignore_index = getattr(getattr(cfg, "ACT", object()), "ignore_index",
                             getattr(getattr(cfg, "TOT", object()), "ignore_index", -100))

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.BaselineTask.lr,
            weight_decay=cfg.BaselineTask.weight_decay
        )
        self.history = {'epoch': [], 'train': [], 'val': []}

    def _run_epoch(self, loader, is_train: bool) -> float:
        self.model.train(is_train)
        total_loss, n_used = 0.0, 0
        pbar = tqdm(loader, desc=f"[{'Train' if is_train else 'Valid'}] ACT-Baseline")
        for batch in pbar:
            for k, v in batch.items():
                if torch.is_tensor(v): batch[k] = v.to(self.device)

            out   = self.model(batch)
            preds = _pick_preds_act(out)
            tgt   = _get_act_targets(batch)

            with torch.set_grad_enabled(is_train):
                loss = _mse_masked(preds, tgt, ignore_index=self.ignore_index)

            if not torch.isfinite(loss): 
                continue
            if is_train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += float(loss.item()); n_used += 1
            pbar.set_postfix(loss=float(loss.item()))
        return (total_loss / n_used) if n_used else float('nan')

    def train(self, save_path="weights/best_baseline_act.pt"):
        print(f"--- Start Baseline Training (ACT) ---")
        _ensure_dir(os.path.dirname(save_path))
        best_val = float('inf')
        for epoch in range(1, self.cfg.BaselineTask.epochs + 1):
            tr = self._run_epoch(self.train_loader, True)
            va = self._run_epoch(self.val_loader,   False)
            val_mse, val_rmse = evaluate_act_metrics(self.model, self.val_loader, self.device, desc="[ValEval] ACT")

            self.history['epoch'].append(epoch)
            self.history['train'].append(tr if tr==tr else float('nan'))
            self.history['val'].append(va if va==va else float('nan'))
            print(f"Epoch {epoch:02d} | Train {tr:.4f} | Val {va:.4f} | Val MSE {val_mse:.4f} | Val RMSE {val_rmse:.4f}")

            if val_mse < best_val:
                best_val = val_mse
                torch.save(self.model.state_dict(), save_path)
                print(f"  -> [Baseline] Best ACT model saved @ {save_path}")

        print(f"--- Finished Baseline Training (ACT) ---")


# ===========================
# ACT Enhancer Trainer (스태킹)
#  - 컨텍스트(모션/감정 등) 모델을 feature로 쓰는 enhancer 모델 학습
#  - 모델에 fusion_head가 있으면 그 부분만 학습
# ===========================
class ACTEnhancerTrainer:
    def __init__(self, model, cfg, train_loader, val_loader, save_dir: str = "outputs"):
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = cfg.Project.device
        self.model.to(self.device)
        self.save_dir = _ensure_dir(getattr(cfg.Project, "save_dir", save_dir))

        self.ignore_index = getattr(getattr(cfg, "ACT", object()), "ignore_index",
                             getattr(getattr(cfg, "TOT", object()), "ignore_index", -100))

        params = self.model.fusion_head.parameters() if hasattr(self.model, "fusion_head") else self.model.parameters()
        self.optimizer = torch.optim.Adam(params, lr=cfg.EnhancerTask.lr, weight_decay=cfg.EnhancerTask.weight_decay)

        self.history = {'epoch': [], 'train': [], 'val': []}

    def _run_epoch(self, loader, is_train: bool) -> float:
        # fusion_head만 있으면 그 부분만 train 모드
        if hasattr(self.model, "fusion_head"):
            self.model.fusion_head.train(is_train)
            self.model.train(False)  # 나머지는 동결 가정
        else:
            self.model.train(is_train)

        total_loss, n_used = 0.0, 0
        pbar = tqdm(loader, desc=f"[{'Train' if is_train else 'Valid'}] Enhancer-ACT")
        for batch in pbar:
            for k, v in batch.items():
                if torch.is_tensor(v): batch[k] = v.to(self.device)

            out   = self.model(batch)
            preds = _pick_preds_act(out)
            tgt   = _get_act_targets(batch)

            with torch.set_grad_enabled(is_train):
                loss = _mse_masked(preds, tgt, ignore_index=self.ignore_index)

            if not torch.isfinite(loss):
                continue
            if is_train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if hasattr(self.model, "fusion_head"):
                    torch.nn.utils.clip_grad_norm_(self.model.fusion_head.parameters(), max_norm=1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += float(loss.item()); n_used += 1
        return (total_loss / n_used) if n_used else float('nan')

    def train(self, save_path="weights/best_enhancer_act.pt"):
        print(f"--- Start Enhancer Training (ACT) ---")
        _ensure_dir(os.path.dirname(save_path))
        best_val = float('inf')

        for epoch in range(1, self.cfg.EnhancerTask.epochs + 1):
            tr = self._run_epoch(self.train_loader, True)
            va = self._run_epoch(self.val_loader,   False)
            val_mse, val_rmse = evaluate_act_metrics(self.model, self.val_loader, self.device, desc="[ValEval] Enhancer-ACT")

            self.history['epoch'].append(epoch)
            self.history['train'].append(tr if tr==tr else float('nan'))
            self.history['val'].append(va if va==va else float('nan'))
            print(f"Epoch {epoch:02d} | Train {tr:.4f} | Val {va:.4f} | Val MSE {val_mse:.4f} | Val RMSE {val_rmse:.4f}")

            if val_mse < best_val:
                best_val = val_mse
                torch.save(self.model.state_dict(), save_path)
                print(f"  -> [Enhancer] Best ACT model saved @ {save_path}")

        print(f"--- Finished Enhancer Training (ACT) ---")
