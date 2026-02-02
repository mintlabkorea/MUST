import os
import csv
import numpy as np
import torch
import warnings
warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings("ignore", message=".*use_reentrant.*")
warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad=True.*")

import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# --- 시각화 라이브러리 추가 ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- 모듈 임포트 ---
from config.config import Config
from trainers.base_trainer import dataProcessor

from trainers.motion_trainer import MotionTrainer
from trainers.emotion_trainer import EmotionTrainer
from trainers.fusion_trainer_v28 import FusionTrainer as ContextExpertTrainer
from trainers.totact_trainer import BaselineTrainer, EnhancerTrainer
from models.totact_models import TOT_BaselineAblation, ACT_BaselineAblation, EnhancedTOTModel, EnhancedACTModel
from data.code.pkl_dataloader_totact import PKLMultiModalDatasetBaseline
from data.loader import make_multitask_loader


# ============================================================
# 유틸
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def ensure_dirs():
    os.makedirs("weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)

@torch.no_grad()
def audit_tot_split(model, loader, device, pooling: str = "mean", desc: str = "[AUDIT] TOT"):
    """
    TOT 평가에 실제로 몇 개 샘플이 들어가는지, 어디서 버려지는지, 클래스 분포가 어떤지 요약.
    - 라벨 우선순위: 'label' -> 'label_act'
    - 시퀀스→원일 때 pooling('mean'|'last'|'max') 적용
    - -100 마스킹 적용
    반환: (stats_dict, true_hist(np), pred_hist(np))
    """
    def pick_logits(out):
        if isinstance(out, dict):
            for k in ("tot_logits","logits","out"):
                if k in out: return out[k]
            raise ValueError("No tot logits in dict")
        return out

    def pool_seq_logits(x, mode):
        if x.dim()!=3: return x
        if mode=="mean": return x.mean(dim=1)
        if mode=="last": return x[:, -1, :]
        if mode=="max":  return x.max(dim=1).values
        raise ValueError(mode)

    model.eval()
    stats = dict(
        batches=0,         # 처리한 배치 수
        elems=0,           # 라벨 원소 수(마스크 전)
        valid=0,           # 마스크 통과한 유효 표본 수
        bc_batches=0,      # (B,C) 모양 배치 수
        btc_batches=0,     # (B,T,C) 모양 배치 수
        seq2one_batches=0, # (B,T,C)+(B,) / (B,1) → 풀링 적용된 배치 수
        seq2seq_batches=0, # (B,T,C)+(B,T) 그대로 평가된 배치 수
        dropped_shape=0,   # 모양 미스매치로 스킵된 배치 수
    )
    true_hist = None
    pred_hist = None
    C_seen = None

    for batch in tqdm(loader, desc=desc):
        for k,v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        out   = model(batch)
        logits = pick_logits(out)  # (B,C) or (B,T,C)

        tgt = batch.get("label", batch.get("label_act", None))
        if tgt is None:
            print("[WARN] batch without label/label_act — skipped")
            stats["dropped_shape"] += 1
            continue

        # case A: (B,T,C) + (B,) or (B,1)  => seq2one
        if logits.dim()==3 and (tgt.dim()==1 or (tgt.dim()==2 and tgt.size(1)==1)):
            logits_eval = pool_seq_logits(logits, pooling) # (B,C)
            targets_eval = tgt.long() if tgt.dim()==1 else tgt.squeeze(1).long()
            pred = logits_eval.argmax(-1)
            true = targets_eval
            mask = (true != -100)
            n_valid = int(mask.sum().item())
            stats["seq2one_batches"] += 1
            stats["bc_batches"] += 1
            C_cur = logits_eval.size(-1)

        # case B: (B,T,C) + (B,T)          => seq2seq
        elif logits.dim()==3 and tgt.dim()==2 and logits.size(1)==tgt.size(1):
            pred = logits.argmax(-1).reshape(-1)   # (B*T,)
            true = tgt.reshape(-1)
            mask = (true != -100)
            n_valid = int(mask.sum().item())
            stats["seq2seq_batches"] += 1
            stats["btc_batches"] += 1
            C_cur = logits.size(-1)

        # case C: (B,C) + (B,)
        elif logits.dim()==2 and tgt.dim()==1:
            pred = logits.argmax(-1)               # (B,)
            true = tgt
            mask = (true != -100)
            n_valid = int(mask.sum().item())
            stats["bc_batches"] += 1
            C_cur = logits.size(-1)

        else:
            print(f"[WARN] shape mismatch skipped: logits={tuple(logits.shape)} tgt={tuple(tgt.shape)}")
            stats["dropped_shape"] += 1
            continue

        # 누적
        stats["batches"] += 1
        stats["elems"]   += int(true.numel())
        stats["valid"]   += n_valid

        C_seen = C_cur if C_seen is None else max(C_seen, C_cur)
        if n_valid > 0:
            if true_hist is None:
                true_hist = torch.zeros(C_cur, dtype=torch.long)
                pred_hist = torch.zeros(C_cur, dtype=torch.long)
            # 사이즈 변동 방지
            if true_hist.numel() < C_cur:
                true_hist = torch.nn.functional.pad(true_hist, (0, C_cur-true_hist.numel()))
                pred_hist = torch.nn.functional.pad(pred_hist, (0, C_cur-pred_hist.numel()))
            true_hist += torch.bincount(true[mask].detach().cpu(), minlength=C_cur)
            pred_hist += torch.bincount(pred[mask].detach().cpu(), minlength=C_cur)

    # 요약 출력
    print("---- TOT AUDIT SUMMARY ----")
    for k,v in stats.items():
        print(f"{k:>18}: {v}")
    if true_hist is not None:
        print(" label histogram:", true_hist.tolist())
        print(" pred  histogram:", pred_hist.tolist())
    else:
        print(" (no valid samples)")

    return stats, (None if true_hist is None else true_hist.numpy()), (None if pred_hist is None else pred_hist.numpy())

# ============================================================
# Collate 함수 (TOT/ACT 공용)
# ============================================================
def collate_fn_baseline(batch):
    keys = set().union(*[b.keys() for b in batch])
    out = {}
    for k in keys:
        vals = [b[k] for b in batch if k in b]
        if not vals:
            continue
        if isinstance(vals[0], torch.Tensor):
            continue
        out[k] = vals

    for k in keys:
        vals = [b[k] for b in batch if k in b and isinstance(b[k], torch.Tensor)]
        if not vals:
            continue
        if all(v.ndim == 0 for v in vals):
            out[k] = torch.stack(vals).long() if k == "label" else torch.stack(vals)
            continue

        padding_value = -100.0 if k == "label" else 0.0
        if all(v.ndim == 1 for v in vals):
            out[k] = pad_sequence(vals, batch_first=True, padding_value=padding_value)
        elif all(v.ndim == 2 for v in vals):
            out[k] = pad_sequence(vals, batch_first=True, padding_value=padding_value)
        else:
            fixed, max_T = [], max(v.shape[0] for v in vals)
            for v in vals:
                v = v[:, None] if v.ndim == 1 else v
                T, D = v.shape[:2]
                if T < max_T:
                    pad = torch.full((max_T - T, D), padding_value, dtype=v.dtype, device=v.device)
                    v = torch.cat([v, pad], dim=0)
                fixed.append(v)
            out[k] = torch.stack(fixed, dim=0)
    return out


# ============================================================
# 로더 생성
# ============================================================
def make_totact_loaders(cfg, dp, task_name: str, batch_size: int = None):
    ds_kwargs = {
        "data_map": dp.data_map,
        "veh_cols": dp.veh_cols,
        "fs": cfg.Data.fs,
        "mode": task_name,
        "window_sec": cfg.Data.window_sec_mot,
        "window_stride": cfg.Data.window_stride_mot,
    }
    train_ds = PKLMultiModalDatasetBaseline(participant_ids=dp.train_keys, **ds_kwargs)
    val_ds   = PKLMultiModalDatasetBaseline(participant_ids=dp.val_keys, **ds_kwargs)
    test_ds  = PKLMultiModalDatasetBaseline(participant_ids=dp.test_keys, **ds_kwargs)

    bs = batch_size or cfg.Data.batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  collate_fn=collate_fn_baseline)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, collate_fn=collate_fn_baseline)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, collate_fn=collate_fn_baseline)
    return train_loader, val_loader, test_loader


# ============================================================
# 평가/시각화
# ============================================================
def save_confusion(y_true, y_pred, save_path: str, title: str = "Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def save_act_line_plot(true_values, pred_values, save_path: str, title: str = "ACT Prediction vs True"):
    plt.figure(figsize=(10, 4))
    plt.plot(true_values, label="True")
    plt.plot(pred_values, label="Pred")
    plt.legend(); plt.title(title); plt.xlabel("Time"); plt.ylabel("ACT (S)")
    plt.tight_layout(); plt.savefig(save_path); plt.close()

@torch.no_grad()
def evaluate_enhancer_tot(model, loader, device, pooling: str = "mean"):
    """
    Final step 전용 TOT 평가 (Enhancer)
    - 출력: 텐서 또는 dict({'tot_logits'|'logits'|'out'})
    - 라벨: 'label' 우선 (없으면 'label_act')
    - 시퀀스→원 과제 시 시간축 풀링(mode: mean/last/max)
    - ignore_index = -100 마스킹
    반환: acc(float), (y_true, y_pred)  # 필요하면 시각화에 사용
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
            return logits.mean(dim=1)
        elif mode == "last":
            return logits[:, -1, :]
        elif mode == "max":
            return logits.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")

    model.eval()
    all_pred, all_true = [], []

    for batch in tqdm(loader, desc="[TEST] Enhanced-TOT"):
        # move to device
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        out = model(batch)
        logits = _pick_logits(out)  # (B,C) or (B,T,C)
        tgt = batch.get("label", batch.get("label_act", None))
        if tgt is None:
            continue

        # 시퀀스→원: logits (B,T,C) & tgt (B,) or (B,1)
        if logits.dim() == 3 and (tgt.dim() == 1 or (tgt.dim() == 2 and tgt.size(1) == 1)):
            logits_eval = _pool_seq_logits(logits, pooling)    # (B,C)
            targets_eval = tgt.long() if tgt.dim() == 1 else tgt.squeeze(1).long()
            pred = logits_eval.argmax(-1)                      # (B,)
            true = targets_eval                                # (B,)
            mask = (true != -100)
            if mask.any():
                all_pred.append(pred[mask].detach().cpu())
                all_true.append(true[mask].detach().cpu())
            continue

        # 시퀀스→시퀀스: (B,T,C) vs (B,T)
        if logits.dim() == 3 and tgt.dim() == 2 and logits.size(1) == tgt.size(1):
            pred = logits.argmax(-1).reshape(-1)               # (B*T,)
            true = tgt.reshape(-1)                             # (B*T,)
            mask = (true != -100)
            if mask.any():
                all_pred.append(pred[mask].detach().cpu())
                all_true.append(true[mask].detach().cpu())
            continue

        # 배치 단일: (B,C) vs (B,)
        if logits.dim() == 2 and tgt.dim() == 1:
            pred = logits.argmax(-1)                           # (B,)
            true = tgt                                         # (B,)
            mask = (true != -100)
            if mask.any():
                all_pred.append(pred[mask].detach().cpu())
                all_true.append(true[mask].detach().cpu())
            continue

        print(f"[WARN] evaluate_enhancer_tot shape mismatch: logits={tuple(logits.shape)}, tgt={tuple(tgt.shape)}")

    if len(all_true) == 0:
        return 0.0, (None, None)

    import numpy as np
    from sklearn.metrics import accuracy_score
    y_pred = torch.cat(all_pred).numpy()
    y_true = torch.cat(all_true).numpy()
    acc = accuracy_score(y_true, y_pred)
    return float(acc), (y_true, y_pred)



@torch.no_grad()
def evaluate_enhancer_act(model, loader, device):
    """
    Enhanced ACT evaluator
    - Supports both legacy (B,T,3) and current (B,T,1) outputs
    - Aligns time length with labels and masks -100.0
    """
    model.eval()
    all_pred_min, all_true = [], []

    for batch in tqdm(loader, desc="[TEST] Enhanced-ACT"):
        # move tensors to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        # forward (tensor or dict)
        out = model(batch)
        pred = out
        if isinstance(out, dict):
            for k in ("act_preds", "preds", "act_logits", "logits", "out"):
                if k in out:
                    pred = out[k]
                    break

        # unify prediction shape
        # - (B,T,1) -> (B,T)
        # - (B,T,3) legacy -> use channel 1 (min_ACT) if available; else first channel
        if pred.dim() == 3:
            C = pred.size(-1)
            if C == 1:
                pred_use = pred.squeeze(-1)           # (B,T)
            elif C >= 2:
                pred_use = pred[..., 1]               # min_ACT channel
            else:
                raise ValueError(f"Unexpected pred channels: {C}")
        else:
            pred_use = pred                           # (B,T) or (B,)

        # labels
        true = batch["label"]                         # (B,T) or (B,) or (B,T,1)
        if true.dim() == 3 and true.size(-1) == 1:
            true = true.squeeze(-1)
        if true.dim() == 1 and pred_use.dim() == 2 and pred_use.size(0) == true.size(0):
            true = true.unsqueeze(1)

        # time alignment (crop to min length)
        if pred_use.dim() == 2 and true.dim() == 2 and pred_use.size(1) != true.size(1):
            T = min(pred_use.size(1), true.size(1))
            pred_use = pred_use[:, :T]
            true     = true[:, :T]

        # mask & gather
        mask = (true != -100.0) & torch.isfinite(true) & torch.isfinite(pred_use)
        if not mask.any():
            continue
        all_pred_min.append(pred_use[mask].detach().cpu())
        all_true.append(true[mask].detach().cpu())

    if not all_true:
        return float("inf"), float("inf"), None, None

    y_pred = torch.cat(all_pred_min).numpy()
    y_true = torch.cat(all_true).numpy()
    mse = ((y_pred - y_true) ** 2).mean()
    rmse = float(np.sqrt(mse))
    return mse, rmse, y_true, y_pred

@torch.no_grad()
def _eval_act_raw_series(model, loader, device):
    """
    (true, pred_min) 시계열을 하나로 이어 붙여 반환.
    - 모델 출력: tensor 또는 dict({'act_preds'|'preds'|'logits'|'out'})
    - 라벨: batch['label'] 또는 batch['label_act']
    - (B,T,1)/(B,T) 모두 처리
    """
    def _pick_preds(out):
        if isinstance(out, dict):
            for k in ('act_preds','preds','act_logits','logits','out'):
                if k in out: return out[k]
            raise ValueError("No ACT outputs in dict")
        return out

    model.eval()
    all_true, all_pred = [], []
    for batch in tqdm(loader, desc="[EVAL] ACT (raw)"):
        for k,v in batch.items():
            if torch.is_tensor(v): batch[k] = v.to(device)

        out   = model(batch)
        preds = _pick_preds(out)           # (B,T,1) or (B,T)
        true  = batch.get('label_act', batch.get('label'))

        # preds -> (B,T)
        if preds.dim()==3 and preds.size(-1)==1: preds = preds.squeeze(-1)
        elif preds.dim()==3:                      preds = preds[..., 0]  # 구버전 가드

        # true -> (B,T)
        if true.dim()==3 and true.size(-1)==1: true = true.squeeze(-1)
        if true.dim()==1 and preds.dim()==2 and preds.size(0)==true.size(0):
            true = true.unsqueeze(1)

        # 길이 보정
        if preds.dim()==2 and true.dim()==2 and preds.size(1)!=true.size(1):
            T = min(preds.size(1), true.size(1))
            preds, true = preds[:, :T], true[:, :T]

        mask = (true != -100.0) & torch.isfinite(true) & torch.isfinite(preds)
        if mask.any():
            all_true.append(true[mask].detach().cpu())
            all_pred.append(preds[mask].detach().cpu())

    if not all_true:
        return None, None

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    return y_true, y_pred


def plot_act_for_subject(model, cfg, dp, participant_id, split="test", out_dir="results"):
    """
    특정 참가자 1명만 대상으로 ACT 라인플롯 저장.
    split: 'train' | 'val' | 'test'
    파일명: results/act_{split}_subj-{ID}.png
    """
    os.makedirs(out_dir, exist_ok=True)
    # 로더 생성
    ds_kwargs = {
        'data_map': dp.data_map,
        'veh_cols': dp.veh_cols,
        'fs': cfg.Data.fs,
        'mode': 'act',
        'window_sec': cfg.Data.window_sec_mot,
        'window_stride': cfg.Data.window_stride_mot,
    }
    from torch.utils.data import DataLoader
    from data.code.pkl_dataloader_totact import PKLMultiModalDatasetBaseline

    if split == "train":
        keys = [k for k in dp.train_keys if k == participant_id]
    elif split == "val":
        keys = [k for k in dp.val_keys if k == participant_id]
    else:
        keys = [k for k in dp.test_keys if k == participant_id]

    if not keys:
        print(f"[WARN] subject {participant_id} not found in {split} split")
        return None

    ds = PKLMultiModalDatasetBaseline(participant_ids=keys, **ds_kwargs)
    loader = DataLoader(ds, batch_size=cfg.Data.batch_size, shuffle=False,
                        collate_fn=collate_fn_baseline)

    y_true, y_pred = _eval_act_raw_series(model, loader, cfg.Project.device)
    if y_true is None:
        print(f"[WARN] no valid ACT labels for subject {participant_id} ({split})")
        return None

    # 저장
    fig_path = os.path.join(out_dir, f"act_{split}_subj-{participant_id}.png")
    plt.figure(figsize=(10,5))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Predicted")
    plt.title(f"ACT Prediction vs. True (subj={participant_id}, split={split})")
    plt.xlabel("Test Time Index"); plt.ylabel("ACT (s)")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()
    print(f"[SAVE] {fig_path}")
    return fig_path


@torch.no_grad()
def plot_act_by_scenario(model, loader, device, out_dir="results", prefix="act_by_scn"):
    """
    로더를 그대로 받아, 배치의 'scenario' 키가 있으면 시나리오별로 분리해서 플롯 저장.
    - 없으면 전체를 'all'로 저장.
    파일명: results/{prefix}_{scenario}.png
    """
    os.makedirs(out_dir, exist_ok=True)

    def _pick_preds(out):
        if isinstance(out, dict):
            for k in ('act_preds','preds','act_logits','logits','out'):
                if k in out: return out[k]
            raise ValueError("No ACT outputs in dict")
        return out

    buckets = {}  # scenario -> {'true': [tensor...], 'pred':[tensor...]}

    model.eval()
    for batch in tqdm(loader, desc="[EVAL] ACT by scenario"):
        for k,v in batch.items():
            if torch.is_tensor(v): batch[k] = v.to(device)

        out   = model(batch)
        preds = _pick_preds(out)
        true  = batch.get('label_act', batch.get('label'))

        # preds -> (B,T)
        if preds.dim()==3 and preds.size(-1)==1: preds = preds.squeeze(-1)
        elif preds.dim()==3:                      preds = preds[..., 0]
        if true.dim()==3 and true.size(-1)==1:    true  = true.squeeze(-1)
        if true.dim()==1 and preds.dim()==2 and preds.size(0)==true.size(0):
            true = true.unsqueeze(1)

        if preds.dim()==2 and true.dim()==2 and preds.size(1)!=true.size(1):
            T = min(preds.size(1), true.size(1))
            preds, true = preds[:, :T], true[:, :T]

        mask = (true != -100.0) & torch.isfinite(true) & torch.isfinite(preds)
        if not mask.any(): 
            continue

        # 시나리오 추출 (없으면 'all')
        scn = batch.get('scenario', None)
        if scn is None:
            key = "all"
        else:
            # 텐서 / 문자열 / 숫자 모두 대응
            if torch.is_tensor(scn):
                key = str(scn.detach().cpu().tolist())
            elif isinstance(scn, (list, tuple)):
                key = str(list(scn))  # 배치 단위 시나리오가 여러개일 수 있음
            else:
                key = str(scn)

        entry = buckets.setdefault(key, {'true': [], 'pred': []})
        entry['true'].append(true[mask].detach().cpu())
        entry['pred'].append(preds[mask].detach().cpu())

    # 저장
    saved = []
    for scn, d in buckets.items():
        if not d['true']: 
            continue
        y_true = torch.cat(d['true']).numpy()
        y_pred = torch.cat(d['pred']).numpy()

        out_path = os.path.join(out_dir, f"{prefix}_{scn}.png")
        plt.figure(figsize=(10,5))
        plt.plot(y_true, label="True")
        plt.plot(y_pred, label="Predicted")
        plt.title(f"ACT Prediction vs. True (scenario={scn})")
        plt.xlabel("Time Index"); plt.ylabel("ACT (s)")
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
        print(f"[SAVE] {out_path}")
        saved.append(out_path)

    if not saved:
        print("[WARN] No ACT plots saved (no valid labels or buckets)")
    return saved

def _smooth(x: np.ndarray, fs: int, smooth_sec: float):
    if smooth_sec is None or smooth_sec <= 0:
        return x
    k = max(1, int(round(fs * smooth_sec)))
    if k == 1: return x
    w = np.ones(k) / k
    return np.convolve(x, w, mode="same")

def _rolling_mae(y_true, y_pred, fs, window_sec=8.0, step_sec=2.0):
    W = max(1, int(round(fs*window_sec)))
    S = max(1, int(round(fs*step_sec)))
    n = len(y_true)
    starts = list(range(0, max(1, n - W + 1), S))
    maes = []
    for s in starts:
        e = np.abs(y_true[s:s+W] - y_pred[s:s+W])
        if len(e) < W: break
        maes.append(e.mean())
    return np.array(starts), np.array(maes), W

def _detect_spikes(y_true, fs, thr=14.0, min_gap_sec=4.0):
    """
    간단 스파이크 탐지: true >= thr인 구간의 시작점들만 추려서 이벤트로 봄.
    이벤트 간 최소 간격 min_gap_sec 보장.
    """
    idx = np.where(y_true >= thr)[0]
    if len(idx) == 0: return []
    events = [idx[0]]
    min_gap = int(round(fs*min_gap_sec))
    for i in idx[1:]:
        if i - events[-1] >= min_gap:
            events.append(i)
    return events

def _save_case_plot(y_true, y_pred, fs, s, e, out_path, title):
    x = np.arange(s, e)
    plt.figure(figsize=(10,4))
    plt.plot(x, y_true[s:e], label="True")
    plt.plot(x, y_pred[s:e], label="Predicted")
    plt.title(title)
    plt.xlabel("Test Time Index"); plt.ylabel("ACT (s)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

@torch.no_grad()
def make_act_case_studies(model,
                          cfg,
                          dp,
                          participant_id: str,
                          split: str = "val",
                          window_sec: float = 8.0,
                          step_sec: float = 2.0,
                          top_k: int = 5,
                          smooth_sec: float = 0.25,
                          spike_thr: float = 14.0,
                          spike_pad_sec: float = 3.0,
                          out_root: str = "results/case_studies"):
    """
    1) 전구간 롤링 MAE 기준 best/worst
    2) 급정거 스파이크(thr 이상) 주변 윈도우 기준 best/worst
    를 각각 top_k씩 PNG로 저장 + summary.csv 저장
    """
    # --- 1) 해당 subject 로더 만들기 ---
    from torch.utils.data import DataLoader
    from data.code.pkl_dataloader_totact import PKLMultiModalDatasetBaseline
    def _keys_for(split):
        if split == "train": pool = dp.train_keys
        elif split == "val": pool = dp.val_keys
        else: pool = dp.test_keys
        return [k for k in pool if str(k) == str(participant_id)]

    keys = _keys_for(split)
    if not keys:
        print(f"[WARN] subject {participant_id} not found in {split} split")
        return None

    ds_kwargs = {
        'data_map': dp.data_map,
        'veh_cols': dp.veh_cols,
        'fs': cfg.Data.fs,
        'mode': 'act',
        'window_sec': cfg.Data.window_sec_mot,
        'window_stride': cfg.Data.window_stride_mot,
    }
    ds = PKLMultiModalDatasetBaseline(participant_ids=keys, **ds_kwargs)
    loader = DataLoader(ds, batch_size=cfg.Data.batch_size, shuffle=False, collate_fn=collate_fn_baseline)
    fs = int(cfg.Data.fs)

    # --- 2) subject의 전체 시계열 얻기 (true, pred) ---
    # 당신이 이미 쓰는 _eval_act_raw_series가 있다면 그걸 써도 좋고,
    # 아래는 그 로직을 inline으로 작성:
    def _pick_preds(out):
        if isinstance(out, dict):
            for k in ('act_preds','preds','act_logits','logits','out'):
                if k in out: return out[k]
            raise ValueError("No ACT outputs")
        return out

    model.eval()
    y_true_all, y_pred_all = [], []
    for batch in tqdm(loader, desc=f"[EVAL] ACT subj={participant_id} ({split})"):
        for k,v in batch.items():
            if torch.is_tensor(v): batch[k] = v.to(cfg.Project.device)
        out = model(batch)
        preds = _pick_preds(out)  # (B,T,1) or (B,T) or (B,T,3)
        if preds.dim()==3 and preds.size(-1)==1: preds = preds.squeeze(-1)
        elif preds.dim()==3: preds = preds[..., 0]
        true = batch.get('label_act', batch.get('label'))
        if true.dim()==3 and true.size(-1)==1: true = true.squeeze(-1)
        if true.dim()==1 and preds.dim()==2 and preds.size(0)==true.size(0):
            true = true.unsqueeze(1)
        if preds.dim()==2 and true.dim()==2 and preds.size(1)!=true.size(1):
            T = min(preds.size(1), true.size(1))
            preds, true = preds[:, :T], true[:, :T]
        mask = (true != -100.0) & torch.isfinite(true) & torch.isfinite(preds)
        if mask.any():
            y_true_all.append(true[mask].detach().cpu())
            y_pred_all.append(preds[mask].detach().cpu())

    if not y_true_all:
        print("[WARN] no valid labels")
        return None

    y_true = torch.cat(y_true_all).numpy()
    y_pred = torch.cat(y_pred_all).numpy()

    # --- 3) smoothing (optional) ---
    y_true_s = _smooth(y_true, fs, smooth_sec)
    y_pred_s = _smooth(y_pred, fs, smooth_sec)

    # --- 4) GLOBAL: 롤링 MAE 기준 top-k best/worst ---
    starts, maes, W = _rolling_mae(y_true_s, y_pred_s, fs, window_sec, step_sec)
    order = np.argsort(maes)
    best_idxs  = order[:top_k]
    worst_idxs = order[::-1][:top_k]

    subj_dir = os.path.join(out_root, f"subj-{participant_id}", split)
    os.makedirs(subj_dir, exist_ok=True)
    summary_rows = []

    for rank, idx in enumerate(best_idxs, 1):
        s, e = int(starts[idx]), int(starts[idx] + W)
        outp = os.path.join(subj_dir, f"global_best_{rank:02d}_{s}-{e}.png")
        _save_case_plot(y_true_s, y_pred_s, fs, s, e, outp,
                        title=f"GLOBAL BEST #{rank} (MAE={maes[idx]:.3f})")
        summary_rows.append(["global_best", rank, s, e, float(maes[idx])])

    for rank, idx in enumerate(worst_idxs, 1):
        s, e = int(starts[idx]), int(starts[idx] + W)
        outp = os.path.join(subj_dir, f"global_worst_{rank:02d}_{s}-{e}.png")
        _save_case_plot(y_true_s, y_pred_s, fs, s, e, outp,
                        title=f"GLOBAL WORST #{rank} (MAE={maes[idx]:.3f})")
        summary_rows.append(["global_worst", rank, s, e, float(maes[idx])])

    # --- 5) SPIKE: 급정거 스파이크 중심 top-k best/worst ---
    centers = _detect_spikes(y_true_s, fs, thr=spike_thr, min_gap_sec=4.0)
    pad = int(round(fs*spike_pad_sec))
    spike_mae = []
    spike_ranges = []
    for c in centers:
        s, e = max(0, c - pad), min(len(y_true_s), c + pad)
        m = np.abs(y_true_s[s:e] - y_pred_s[s:e]).mean()
        spike_mae.append(m)
        spike_ranges.append((s, e))
    if spike_ranges:
        spike_order = np.argsort(spike_mae)
        s_best  = spike_order[:top_k]
        s_worst = spike_order[::-1][:top_k]
        for rank, k in enumerate(s_best, 1):
            s, e = spike_ranges[k]
            outp = os.path.join(subj_dir, f"spike_best_{rank:02d}_{s}-{e}.png")
            _save_case_plot(y_true_s, y_pred_s, fs, s, e, outp,
                            title=f"SPIKE BEST #{rank} (MAE={spike_mae[k]:.3f})")
            summary_rows.append(["spike_best", rank, int(s), int(e), float(spike_mae[k])])
        for rank, k in enumerate(s_worst, 1):
            s, e = spike_ranges[k]
            outp = os.path.join(subj_dir, f"spike_worst_{rank:02d}_{s}-{e}.png")
            _save_case_plot(y_true_s, y_pred_s, fs, s, e, outp,
                            title=f"SPIKE WORST #{rank} (MAE={spike_mae[k]:.3f})")
            summary_rows.append(["spike_worst", rank, int(s), int(e), float(spike_mae[k])])

    # --- 6) 요약 CSV 저장 ---
    csv_path = os.path.join(subj_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket","rank","start","end","mae"])
        w.writerows(summary_rows)
    print(f"[SAVE] summary -> {os.path.abspath(csv_path)}")

    return subj_dir

# ============================================================
# 메인 파이프라인 (사전학습부터 시작)
# ============================================================
def main():
    ensure_dirs()
    cfg = Config()
    set_seed(cfg.Project.seed)
    dp = dataProcessor(cfg); 
    if hasattr(dp, "prepare"): dp.prepare()
    device = cfg.Project.device

    # --------------------------------------------------------
    # STEP 0) 모션/감정 사전학습 (반드시 실행)
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 0) Pretraining Motion / Emotion (start here)")
    print("="*60)

    mot_ckpt = "weights/best_pretrain_motion_imu_veh.pt"
    emo_ckpt = "weights/best_pretrain_emotion_ppg_sc_survey.pt"

    motion_tr = MotionTrainer(cfg, train_keys=dp.train_keys, val_keys=dp.val_keys, test_keys=dp.test_keys)
    emotion_tr = EmotionTrainer(cfg, dp)
    
    # 학습 & 저장
    # try:
    #     motion_tr.train(save_path=mot_ckpt)
    # except TypeError:
    #     motion_tr.train()
    # try:
    #     emotion_tr.train(save_path=emo_ckpt)
    # except TypeError:
    #     emotion_tr.train()

    # 베스트 로드 및 검증
    if os.path.exists(mot_ckpt):
        state = torch.load(mot_ckpt, map_location="cpu")
        motion_tr.load(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
    if os.path.exists(emo_ckpt):
        state = torch.load(emo_ckpt, map_location="cpu")
        emotion_tr.load(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)

    if hasattr(motion_tr, "evaluate"):
        try: motion_tr.evaluate(split="val")
        except TypeError: motion_tr.evaluate()
    if hasattr(emotion_tr, "evaluate"):
        try: emotion_tr.evaluate(split="val")
        except TypeError: emotion_tr.evaluate()

    # --------------------------------------------------------
    # STEP 1) 컨텍스트 전문가 (v28 퓨전: 감정/모션)
    #   - 가능하면 사전학습 가중치를 주입
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1) Train Context Expert (Fusion v28)")
    print("="*60)

    context_expert = ContextExpertTrainer(cfg, dp.train_keys, dp.val_keys, dp.test_keys)

    # 사전학습 가중치 전달: 트레이너가 해당 API를 제공할 경우 자동 주입
    injected = False
    if hasattr(context_expert, "load_from_pretrained"):
        try:
            context_expert.load_from_pretrained(motion_ckpt=mot_ckpt, emotion_ckpt=emo_ckpt)
            injected = True
            print("[Fusion] loaded pretrain via load_from_pretrained")
        except Exception as e:
            print(f"[Fusion] pretrain injection skipped: {e}")
    elif all(hasattr(context_expert, attr) for attr in ["motion_backbone", "emotion_backbone"]):
        try:
            if os.path.exists(mot_ckpt):
                state = torch.load(mot_ckpt, map_location="cpu")
                msd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
                context_expert.motion_backbone.load_state_dict(msd, strict=False)
            if os.path.exists(emo_ckpt):
                state = torch.load(emo_ckpt, map_location="cpu")
                esd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
                context_expert.emotion_backbone.load_state_dict(esd, strict=False)
            injected = True
            print("[Fusion] loaded pretrain into motion_backbone/emotion_backbone")
        except Exception as e:
            print(f"[Fusion] backbone load skipped: {e}")

    fusion_ckpt = "weights/best_fusion_v28.pt"
    #context_expert.fusion_train(save_path=fusion_ckpt)

    if os.path.exists(fusion_ckpt):
        state = torch.load(fusion_ckpt, map_location="cpu")
        context_expert.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)

    # 검증 성능
    va_loader = make_multitask_loader(cfg, dp.val_keys, shuffle=False, dp=dp)
    try:
        va_mot, va_v, va_a = context_expert.evaluate(va_loader)
        print(f"[Fusion v28] VAL -> Motion: {va_mot:.4f} | Valence: {va_v:.4f} | Arousal: {va_a:.4f}")
    except Exception:
        _ = context_expert.evaluate(va_loader)

    # --------------------------------------------------------
    # STEP 2-1) TOT/ACT 베이스라인 학습
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2-1) Train TOT/ACT Baselines")
    print("="*60)

    tot_train_loader, tot_val_loader, tot_test_loader = make_totact_loaders(cfg, dp, "tot", batch_size=cfg.Data.batch_size)
    act_train_loader, act_val_loader, act_test_loader = make_totact_loaders(cfg, dp, "act", batch_size=cfg.Data.batch_size)

    tot_baseline = TOT_BaselineAblation(cfg).to(device)
    print(f"[TOT] using modalities: {cfg.TOT.use_modalities} (feat_dim={cfg.TOT.feat_dim}, attn_pool={cfg.TOT.attn_pool})")
    act_baseline = ACT_BaselineAblation(cfg).to(device)
    
    BaselineTrainer(tot_baseline, cfg, tot_train_loader, tot_val_loader, "tot", num_classes=3).train(test_loader=tot_test_loader)  # saves weights/best_baseline_tot.pt
    #BaselineTrainer(act_baseline, cfg, act_train_loader, act_val_loader, "act", num_classes=3).train(test_loader=act_test_loader)   # saves weights/best_baseline_act.pt

    # --------------------------------------------------------
    # STEP 2-2) 엔핸서 파인튜닝 (베이스라인 동결 + 퓨전 특성 추가 입력)
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2-2) Fine-tune with Fused Context (freeze baselines)")
    print("="*60)

    tot_baseline_ckpt = "weights/best_baseline_tot.pt"
    act_baseline_ckpt = "weights/best_baseline_act.pt"
    if os.path.exists(tot_baseline_ckpt):
        tot_baseline.load_state_dict(torch.load(tot_baseline_ckpt, map_location="cpu"))
    if os.path.exists(act_baseline_ckpt):
        act_baseline.load_state_dict(torch.load(act_baseline_ckpt, map_location="cpu"))
    tot_baseline.eval(); act_baseline.eval()

    enhancer_tot = EnhancedTOTModel(cfg, tot_baseline, context_expert).to(device)
    enhancer_act = EnhancedACTModel(cfg, act_baseline, context_expert).to(device)

    enh_tot_ckpt = "weights/best_enhancer_tot.pt"
    enh_act_ckpt = "weights/best_enhancer_act.pt"
    EnhancerTrainer(enhancer_tot, cfg, tot_train_loader, tot_val_loader, "tot", num_classes=3).train(save_path=enh_tot_ckpt)
    EnhancerTrainer(enhancer_act, cfg, act_train_loader, act_val_loader, "act", num_classes=3).train(save_path=enh_act_ckpt)

    if os.path.exists(enh_tot_ckpt):
        enhancer_tot.load_state_dict(torch.load(enh_tot_ckpt, map_location="cpu"))
    if os.path.exists(enh_act_ckpt):
        enhancer_act.load_state_dict(torch.load(enh_act_ckpt, map_location="cpu"))

    # --------------------------------------------------------
    # STEP 3) 최종 평가 + 시각화
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 3) Final Evaluation & Visualization")
    print("="*60)

    # Baseline or Enhancer 아무거나 모델/로더 넣어 감사 가능
    stats_val, ytrue_hist_val, ypred_hist_val = audit_tot_split(
        model=enhancer_tot,   # baseline_tot 도 가능
        loader=tot_val_loader,
        device=cfg.Project.device,
        pooling="mean",
        desc="[AUDIT] TOT Val"
    )

    stats_test, ytrue_hist_test, ypred_hist_test = audit_tot_split(
        model=enhancer_tot,
        loader=tot_test_loader,
        device=cfg.Project.device,
        pooling="mean",
        desc="[AUDIT] TOT Test"
    )


    te_loader = make_multitask_loader(cfg, dp.test_keys, shuffle=False, dp=dp)
    try:
        test_mot, test_v, test_a = context_expert.evaluate(te_loader)
    except Exception:
        out = context_expert.evaluate(te_loader)
        test_mot, test_v, test_a = (out + (float("nan"),) * 3)[:3]

    print(f"[Fusion v28] TEST -> Motion: {test_mot:.4f} | Valence: {test_v:.4f} | Arousal: {test_a:.4f}")

    test_acc_tot, (tot_true, tot_pred) = evaluate_enhancer_tot(enhancer_tot, tot_test_loader, device)

    test_mse_min_act, test_rmse_mean_act, act_true, act_pred = evaluate_enhancer_act(enhancer_act, act_test_loader, device)

    if tot_true is not None and tot_pred is not None:
        save_confusion(tot_true, tot_pred, "results/tot_confusion.png", title="TOT Confusion Matrix")
    if act_true is not None and act_pred is not None:
        save_act_line_plot(act_true, act_pred, "results/act_prediction_plot.png")

    print("\n" + "="*40)
    print("FINAL TEST RESULTS")
    print("="*40)
    print(f"Emotion/Motion -> Motion Acc: {test_mot:.4f}, Valence Acc: {test_v:.4f}, Arousal Acc: {test_a:.4f}")
    print(f"Enhanced TOT   -> TOT Acc: {test_acc_tot:.4f}")
    print(f"Enhanced ACT   -> min ACT MSE: {test_mse_min_act:.4f}")
    print(f"Enhanced ACT   -> mean ACT RMSE: {test_rmse_mean_act:.4f}")
    print("="*40)
    print("📊 Plots saved under ./results")

    plot_act_for_subject(enhancer_act, cfg, dp, participant_id="10", split="test", out_dir="results")
    # 예: test split 전체 로더로 시나리오별 플롯 저장
    saved_paths = plot_act_by_scenario(enhancer_act, act_test_loader, device=cfg.Project.device,
                                   out_dir="results", prefix="act_test_scn")
    plot_act_for_subject(enhancer_act, cfg, dp, participant_id="16", split="val", out_dir="results")

    # 예: 검증셋, subject '16'에 대해 케이스 스터디 저장
    case_dir = make_act_case_studies(
        model=enhancer_act, cfg=cfg, dp=dp, participant_id="16", split="val",
        window_sec=8.0, step_sec=2.0, top_k=5,
        smooth_sec=0.25,          # 0~0.5s 정도 부드럽게 (옵션)
        spike_thr=14.0,           # 급정거 ACT 상한 근처
        spike_pad_sec=3.0,        # 스파이크 주변 ±3초
        out_root="results/case_studies"
    )
    print("saved to:", case_dir)

if __name__ == "__main__":
    main()
