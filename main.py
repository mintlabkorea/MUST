
import argparse
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/must_matplotlib")
import torch
import warnings
warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings("ignore", message=".*use_reentrant.*")
warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad=True.*")
from sklearn.preprocessing import label_binarize

import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# --- 시각화 라이브러리 추가 ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- 모듈 임포트 ---
from config.config import Config
from trainers.base_trainer import dataProcessor

# 필요한 모든 모델과 트레이너를 임포트합니다.
from trainers.fusion_trainer_v28_assym import FusionTrainer as ContextExpertTrainer
from trainers.totact_trainer import BaselineTrainer, EnhancerTrainer
from models.totact_models import TOT_Baseline, ACT_Baseline, EnhancedTOTModel, EnhancedACTModel
from data.code.pkl_dataloader_totact import PKLMultiModalDatasetBaseline
from data.loader import make_motion_loader, make_multitask_loader

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collate_fn_baseline(batch):
    """베이스라인 태스크를 위한 간단한 Collate 함수"""
    keys = batch[0].keys()
    collated = {key: [d[key] for d in batch] for key in keys}

    for key, values in collated.items():
        if isinstance(values[0], torch.Tensor):
            # 0차원 텐서(스칼라)는 stack, 그 외는 pad_sequence 사용
            if values[0].dim() == 0:
                collated[key] = torch.stack(values)
            else:
                collated[key] = pad_sequence(values, batch_first=True, padding_value=-100.0)
    
    return collated


def run_baseline_task(cfg, dp, task_name):
    print(f"\n{'='*50}\n🚀 Running Baseline for Task: {task_name.upper()}\n{'='*50}")
    
    # 1. 데이터 로더 생성
    ds_kwargs = {
        'data_map': dp.data_map, # dataProcessor가 로드한 데이터 사용
        'veh_cols': dp.veh_cols, 
        'fs': cfg.Data.fs,
        'mode': task_name, 
        'window_sec': cfg.Data.window_sec_mot, 
        'window_stride': cfg.Data.window_stride_mot
    }
    
    train_ds = PKLMultiModalDatasetBaseline(participant_ids=dp.train_keys, **ds_kwargs)
    val_ds = PKLMultiModalDatasetBaseline(participant_ids=dp.val_keys, **ds_kwargs)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.Data.batch_size, shuffle=True, collate_fn=collate_fn_baseline, num_workers=cfg.Data.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.Data.batch_size, shuffle=False, collate_fn=collate_fn_baseline, num_workers=cfg.Data.num_workers)

    # 2. 모델 선택 및 초기화
    if task_name == 'tot':
        model = TOT_Baseline(cfg)
    elif task_name == 'act':
        model = ACT_Baseline(cfg)
    else:
        raise ValueError(f"Unknown task: {task_name}")

    # 3. 트레이너 실행
    trainer = BaselineTrainer(model, cfg, train_loader, val_loader, task_name)
    trainer.train()

def save_tot_confusion_matrix(true_labels, pred_labels, save_path):
    """TOT 결과에 대한 Confusion Matrix를 생성하고 저장합니다."""
    cm = confusion_matrix(true_labels, pred_labels)
    class_names = ['Emergency', 'Caution', 'Safe'] # 클래스 이름 (긴급, 주의, 안전)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('TOT Classification Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None  # scipy 미설치 대비

def _make_odd(n: int) -> int:
    n = int(max(3, n))
    return n if n % 2 == 1 else n + 1

def _moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    w = _make_odd(w)
    pad = w // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(x_pad, kernel, mode="same")[pad:-pad]

from typing import Optional, Tuple

def save_act_line_plot(
    true_values,
    pred_values,
    save_path=None,
    *,
    smooth: Optional[str] = "savgol",  # None | "ma" | "savgol"
    window: int = 101,
    polyorder: int = 3,
    segment: Optional[Tuple[int, int]] = None,
    dpi: int = 300,
    show: bool = True
):
    """ACT 실제-예측 라인 플롯 (스무딩/세그먼트/저장/보기 제어)"""
    true_values = np.asarray(true_values, dtype=float)
    pred_values = np.asarray(pred_values, dtype=float)

    # 구간 잘라내기
    if segment is not None:
        s, e = segment
        true_values = true_values[s:e]
        pred_values = pred_values[s:e]

    # 창 길이 자동 보정
    if len(true_values) < window:
        window = _make_odd(max(3, len(true_values)//5*2 + 1))

    # 스무딩
    if smooth == "ma":
        true_s = _moving_avg(true_values, window)
        pred_s = _moving_avg(pred_values, window)
        smooth_str = f"(MA, w={window})"
    elif smooth == "savgol":
        if savgol_filter is None:
            raise ImportError("Savitzky–Golay 사용하려면 scipy가 필요합니다: pip install scipy")
        window = _make_odd(min(window, len(true_values) - (1 - len(true_values) % 2)))
        window = max(3, window)
        polyorder = min(polyorder, window - 1)
        true_s = savgol_filter(true_values, window_length=window, polyorder=polyorder, mode="interp")
        pred_s = savgol_filter(pred_values, window_length=window, polyorder=polyorder, mode="interp")
        smooth_str = f"(S-G, w={window}, p={polyorder})"
    else:
        true_s, pred_s = true_values, pred_values
        smooth_str = ""

    # 플롯
    plt.figure(figsize=(10, 5))
    plt.plot(true_s, color='red', label='True')
    plt.plot(pred_s, color='blue', label='Predicted')
    title = 'ACT Prediction vs. True Values ' + (smooth_str if smooth_str else "")
    plt.title(title)
    plt.xlabel('Test Time Index')
    plt.ylabel('ACT (s)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 저장/보기
    if save_path:
        plt.savefig(save_path, dpi=dpi)
        print(f"ACT prediction plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

def save_act_plots_bulk(
    true_values,
    pred_values,
    *,
    out_dir: str = "results/act_plots",
    smooth: str = "ma",        # None | "ma" | "savgol"
    window: int = 81,
    dpi: int = 300,
    # 세그먼트 자동 분할 설정
    segment_len: int = 2000,   # 세그먼트 길이
    stride: int = 2000,        # 세그먼트 간 이동 크기(슬라이딩 윈도우 원하면 < segment_len으로)
    prefix: str = "act",       # 파일명 접두어
):
    """
    ACT 라인 플롯을 전체/세그먼트로 일괄 저장합니다.
    - 전체 플롯: {out_dir}/{prefix}_full.png
    - 세그먼트 플롯: {out_dir}/{prefix}_seg_{index:03d}_{start}_{end}.png
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) 전체 플롯 저장
    full_path = os.path.join(out_dir, f"{prefix}_full.png")
    save_act_line_plot(
        true_values, pred_values,
        save_path=full_path,
        smooth=smooth, window=window,
        segment=None, dpi=dpi, show=False
    )

    # 2) 세그먼트 리스트 생성
    n = len(true_values)
    segments = []
    start = 0
    while start < n:
        end = min(start + segment_len, n)
        # 너무 짧은 마지막 세그먼트는 생략(원하면 조건 제거)
        if end - start >= 3:
            segments.append((start, end))
        if end == n:
            break
        start += stride

    # 3) 각 세그먼트 플롯 저장
    for i, (s, e) in enumerate(segments):
        seg_path = os.path.join(out_dir, f"{prefix}_seg_{i:03d}_{s}_{e}.png")
        save_act_line_plot(
            true_values, pred_values,
            save_path=seg_path,
            smooth=smooth, window=window,
            segment=(s, e), dpi=dpi, show=False
        )

def save_motion_line_plot(
    true_values,
    pred_values,
    save_path=None,
    *,
    segment: Optional[Tuple[int, int]] = None,
    dpi: int = 300,
    show: bool = True
):
    """Motion 시계열 라인 플롯 (구간 확대/저장/보기 제어)"""
    true_values = np.asarray(true_values, dtype=float)
    pred_values = np.asarray(pred_values, dtype=float)

    if segment is not None:
        s, e = segment
        true_values = true_values[s:e]
        pred_values = pred_values[s:e]

    plt.figure(figsize=(10, 4))
    plt.plot(true_values, label="Motion True")
    plt.plot(pred_values, label="Motion Pred", alpha=0.9)
    plt.xlabel("Time Index")
    plt.ylabel("Motion (arb. unit)")
    plt.title("Motion Prediction vs. True")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi)
        print(f"Motion line plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

def save_motion_plots_bulk(
    true_values,
    pred_values,
    *,
    out_dir: str = "results/motion_plots",
    dpi: int = 300,
    segment_len: int = 2000,   # 세그먼트 길이
    stride: int = 2000,        # 오버랩 X면 segment_len과 동일
    prefix: str = "motion"
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) 전체 플롯
    full_path = os.path.join(out_dir, f"{prefix}_full.png")
    save_motion_line_plot(true_values, pred_values, save_path=full_path, dpi=dpi, show=False)

    # 2) 세그먼트들
    n = len(true_values)
    i = 0
    for start in range(0, n, stride):
        end = min(start + segment_len, n)
        if end - start < 3:
            break
        seg_path = os.path.join(out_dir, f"{prefix}_seg_{i:03d}_{start}_{end}.png")
        save_motion_line_plot(
            true_values, pred_values,
            save_path=seg_path, segment=(start, end), dpi=dpi, show=False
        )
        i += 1

def save_motion_plots_from_pt(
    pt_path: str,
    *,
    out_dir: str = "results/motion_plots",
    dpi: int = 300,
    segment_len: int = 2000,
    stride: int = 2000,
    prefix: str = "motion",
    true_key: str = None,
    pred_key: str = None,
):
    """
    pt 파일은 (가능하면) {'true': ..., 'pred': ...} 형태를 추천.
    키가 다르면 자동으로 유사 키를 탐색합니다.
    """
    import torch, numpy as np

    bundle = torch.load(pt_path, map_location="cpu")

    # 1) 키 자동 탐색
    if true_key is None:
        for k in ["true", "y_true", "gt", "target", "targets", "label", "labels"]:
            if k in bundle: 
                true_key = k; break
    if pred_key is None:
        for k in ["pred", "y_pred", "preds", "prediction", "predictions", "output", "outputs", "logits"]:
            if k in bundle: 
                pred_key = k; break

    # 2) 실패 시, 사용 가능한 키 안내
    if true_key is None or pred_key is None:
        available = ", ".join(sorted(map(str, bundle.keys())))
        raise KeyError(
            f"Could not find true/pred keys in '{pt_path}'. "
            f"Available keys: {available}. "
            f"Pass true_key=..., pred_key=... explicitly."
        )

    mot_true = bundle[true_key]
    mot_pred = bundle[pred_key]

    # 3) torch.Tensor -> numpy
    if hasattr(mot_true, "detach"): mot_true = mot_true.detach().cpu().numpy()
    if hasattr(mot_pred, "detach"): mot_pred = mot_pred.detach().cpu().numpy()
    mot_true = np.asarray(mot_true)
    mot_pred = np.asarray(mot_pred)

    # 4) 모양 보정/검증
    mot_true = mot_true.squeeze()
    mot_pred = mot_pred.squeeze()
    if mot_true.ndim > 1: mot_true = mot_true.reshape(-1)
    if mot_pred.ndim > 1: mot_pred = mot_pred.reshape(-1)
    if mot_true.shape[0] != mot_pred.shape[0]:
        raise ValueError(f"Length mismatch: true={mot_true.shape}, pred={mot_pred.shape}")

    # 5) 기존 일괄 저장 함수 재사용
    save_motion_plots_bulk(
        mot_true, mot_pred,
        out_dir=out_dir, dpi=dpi, segment_len=segment_len, stride=stride, prefix=prefix
    )


@torch.no_grad()
def evaluate_enhancer_model(model, loader, task_name, device):
    """
    모델 평가 후, 성능 지표와 함께 시각화에 필요한 raw 예측/정답 값을 반환합니다.
    """
    model.eval()
    all_preds, all_trues = [], []
    all_preds_min = []

    for batch in tqdm(loader, desc=f"[Evaluate] Enhancer-{task_name.upper()}"):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        preds = model(batch)
        
        mask = batch['label'] != -100
        if not mask.any(): continue

        if task_name == 'act':
            all_preds.append(preds[mask][:, 0].cpu())
            all_preds_min.append(preds[mask][:, 1].cpu())
        else: # 'tot'
            all_preds.append(preds[mask].cpu())

        all_trues.append(batch['label'][mask].cpu())
    
    if not all_trues: 
        if task_name == 'act': return (0.0, 0.0), (None, None)
        else: return 0.0, (None, None)

    all_preds = torch.cat(all_preds).numpy()
    all_trues = torch.cat(all_trues).numpy()
    
    if task_name == 'tot':
        predicted_labels = np.argmax(all_preds, axis=1)
        score = accuracy_score(all_trues, predicted_labels)
        return score, (all_trues, predicted_labels) # 점수와 함께 (실제값, 예측값) 반환
    else: # ACT
        all_preds_min = torch.cat(all_preds_min).numpy()
        main_mse = mean_squared_error(all_trues, all_preds_min)
        mean_act_rmse = mean_squared_error(all_trues, all_preds, squared=False)
        
        # 실제값은 하나이므로 all_trues를 사용하고, 예측값은 min_act를 사용
        return (main_mse, mean_act_rmse), (all_trues, all_preds_min)

def plot_multiclass_roc(y_true, y_proba, class_names, title, out_path):
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    n_classes = y_proba.shape[1]
    Y = label_binarize(y_true, classes=np.arange(n_classes))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(7,5))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=1.5, label=f"{class_names[i]} (AUC={roc_auc[i]:.2f})")
    plt.plot(fpr["micro"], tpr["micro"], lw=2.5, linestyle="--", label=f"micro (AUC={roc_auc['micro']:.2f})")
    plt.plot([0,1],[0,1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()

def plot_multiclass_pr(y_true, y_proba, class_names, title, out_path):
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    n_classes = y_proba.shape[1]
    Y = label_binarize(y_true, classes=np.arange(n_classes))
    precision, recall, ap = {}, {}, {}
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y[:, i], y_proba[:, i])
        ap[i] = average_precision_score(Y[:, i], y_proba[:, i])

    plt.figure(figsize=(7,5))
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], lw=1.5, label=f"{class_names[i]} (AP={ap[i]:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()

def run_smoke(cfg):
    print("--- Smoke Check: sample data pipeline ---")
    dp = dataProcessor(cfg)
    dp.prepare()

    motion_loader = make_motion_loader(cfg, dp.train_keys, shuffle=False, dp=dp)
    multitask_loader = make_multitask_loader(cfg, dp.train_keys, shuffle=False, dp=dp)
    motion_batch = next(iter(motion_loader))
    multitask_batch = next(iter(multitask_loader))

    print(f"Profile: {cfg.Project.profile}")
    print(f"PKL: {cfg.Project.pkl_all}")
    print(f"Survey CSV: {cfg.Project.survey_csv}")
    print(f"Train/Val/Test keys: {len(dp.train_keys)}/{len(dp.val_keys)}/{len(dp.test_keys)}")
    print(f"Motion batch imu: {tuple(motion_batch['imu_motion'].shape)}")
    print(f"Motion batch veh: {tuple(motion_batch['veh_motion'].shape)}")
    print(f"Motion labels: {tuple(motion_batch['label_motion'].shape)}")
    print(f"Emotion batch ppg: {tuple(multitask_batch['ppg_emotion'].shape)}")
    print(f"Survey batch: {tuple(multitask_batch['survey_e'].shape)}")
    print("Smoke check passed.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run MUST training or a sample-data smoke check.")
    parser.add_argument(
        "--profile",
        choices=["auto", "full", "sample"],
        default=os.environ.get("MUST_PROFILE", "auto"),
        help="Data profile. auto uses the private full dataset when present, otherwise sample.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Only validate config, sample data loading, and dataloader batch shapes.",
    )
    return parser.parse_args()


def main(profile="auto", smoke=False):
    cfg = Config().apply_profile(profile)
    set_seed(cfg.Project.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.Project.device = device
    print(f"Running on device: {device}")

    if smoke or cfg.Project.profile == "sample":
        run_smoke(cfg)
        return

    print("--- Data Preparation ---")
    dp = dataProcessor(cfg)
    dp.prepare()

    weights_dir = cfg.Project.weights_dir
    results_dir = cfg.Project.results_dir
    path_context_expert = os.path.join(weights_dir, "best_context_expert.pt")
    path_tot_baseline = os.path.join(weights_dir, "best_baseline_tot.pt")
    path_act_baseline = os.path.join(weights_dir, "best_baseline_act.pt")
    path_enhancer_tot = os.path.join(weights_dir, "best_enhancer_tot.pt")
    path_enhancer_act = os.path.join(weights_dir, "best_enhancer_act.pt")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True) 

    # ============================================================================
    #  1단계: 전문가 모델 훈련 (주석 처리된 부분은 필요 시 활성화)
    # ============================================================================
    print("\n\n--- STEP 1: 전문가 모델 훈련 ---")
    print("\n▶ (1-1) Training Context Expert (Emotion/Motion)...")
    context_trainer = ContextExpertTrainer(cfg, dp.train_keys, dp.val_keys, dp.test_keys)
    context_trainer.fusion_train(save_path=path_context_expert)
    print("\n▶ (1-2) Training Task Experts (TOT/ACT Baselines)...")
    run_baseline_task(cfg, dp, 'tot')
    run_baseline_task(cfg, dp, 'act')

    # ============================================================================
    #  2단계: Enhancer 모델 훈련 (주석 처리된 부분은 필요 시 활성화)
    # ============================================================================
    print("\n\n--- STEP 2: Enhancer 모델 훈련 ---")
    
    print("\n▶ (2-1) Loading Pre-trained Expert Models...")
    context_expert = ContextExpertTrainer(cfg, dp.train_keys, dp.val_keys, dp.test_keys)
    context_expert.load_state_dict(torch.load(path_context_expert)['model_state_dict'])
    
    tot_baseline_expert = TOT_Baseline(cfg)
    tot_baseline_expert.load_state_dict(torch.load(path_tot_baseline))
    
    act_baseline_expert = ACT_Baseline(cfg)
    act_baseline_expert.load_state_dict(torch.load(path_act_baseline))

    # 2-2. TOT Enhancer 훈련
    enhancer_tot = EnhancedTOTModel(cfg, tot_baseline_expert, context_expert)
    ds_kwargs_tot = {'data_map': dp.data_map, 'veh_cols': dp.veh_cols, 'fs': cfg.Data.fs, 'mode': 'tot', 'window_sec': cfg.Data.window_sec_mot, 'window_stride': cfg.Data.window_stride_mot}
    tot_train_ds = PKLMultiModalDatasetBaseline(participant_ids=dp.train_keys, **ds_kwargs_tot)
    tot_val_ds = PKLMultiModalDatasetBaseline(participant_ids=dp.val_keys, **ds_kwargs_tot)
    tot_train_loader = DataLoader(tot_train_ds, batch_size=cfg.Data.batch_size, shuffle=True, collate_fn=collate_fn_baseline)
    tot_val_loader = DataLoader(tot_val_ds, batch_size=cfg.Data.batch_size, shuffle=False, collate_fn=collate_fn_baseline)
    tot_enhancer_trainer = EnhancerTrainer(enhancer_tot, cfg, tot_train_loader, tot_val_loader, 'tot')
    tot_enhancer_trainer.train(save_path=path_enhancer_tot)
    
    # 2-3. ACT Enhancer 훈련
    enhancer_act = EnhancedACTModel(cfg, act_baseline_expert, context_expert)
    ds_kwargs_act = {'data_map': dp.data_map, 'veh_cols': dp.veh_cols, 'fs': cfg.Data.fs, 'mode': 'act', 'window_sec': cfg.Data.window_sec_mot, 'window_stride': cfg.Data.window_stride_mot}
    act_train_ds = PKLMultiModalDatasetBaseline(participant_ids=dp.train_keys, **ds_kwargs_act)
    act_val_ds = PKLMultiModalDatasetBaseline(participant_ids=dp.val_keys, **ds_kwargs_act)
    act_train_loader = DataLoader(act_train_ds, batch_size=cfg.Data.batch_size, shuffle=True, collate_fn=collate_fn_baseline)
    act_val_loader = DataLoader(act_val_ds, batch_size=cfg.Data.batch_size, shuffle=False, collate_fn=collate_fn_baseline)
    act_enhancer_trainer = EnhancerTrainer(enhancer_act, cfg, act_train_loader, act_val_loader, 'act')
    act_enhancer_trainer.train(save_path=path_enhancer_act)


    # ============================================================================
    #  3단계: 최종 성능 평가 및 시각화
    # ============================================================================
    print("\n\n--- STEP 3: 최종 성능 평가 및 시각화 ---")
    
    # 3-1. 감정/행동 성능 평가
    print("\n▶ (3-1) Evaluating Context Expert (Emotion/Motion)...")
    context_expert = ContextExpertTrainer(cfg, dp.train_keys, dp.val_keys, dp.test_keys)
    context_expert.load_state_dict(torch.load(path_context_expert, map_location=device)['model_state_dict'])
    context_expert.eval()
    test_loader_emo_mot = make_multitask_loader(cfg, dp.test_keys, shuffle=False, dp=dp)
    test_acc_mot, test_acc_v, test_acc_a = context_expert.evaluate(test_loader_emo_mot)

    # 3-2. 최종 TOT/ACT 성능 평가
    print("\n▶ (3-2) Evaluating Enhanced TOT/ACT Models...")
    # TOT 모델 로드 및 평가
    tot_baseline_expert = TOT_Baseline(cfg) # Enhancer 모델 초기화를 위해 필요
    enhancer_tot = EnhancedTOTModel(cfg, tot_baseline_expert, context_expert)
    enhancer_tot.load_state_dict(torch.load(path_enhancer_tot, map_location=device))
    enhancer_tot.to(device)
    ds_kwargs_tot = {'data_map': dp.data_map, 'veh_cols': dp.veh_cols, 'fs': cfg.Data.fs, 'mode': 'tot', 'window_sec': cfg.Data.window_sec_mot, 'window_stride': cfg.Data.window_stride_mot}
    tot_test_ds = PKLMultiModalDatasetBaseline(participant_ids=dp.test_keys, **ds_kwargs_tot)
    tot_test_loader = DataLoader(tot_test_ds, batch_size=cfg.Data.batch_size, shuffle=False, collate_fn=collate_fn_baseline)
    
    # 평가 함수로부터 성능과 시각화용 데이터를 함께 받음
    test_acc_tot, (tot_true, tot_pred) = evaluate_enhancer_model(enhancer_tot, tot_test_loader, 'tot', device)
    if tot_true is not None:
        save_tot_confusion_matrix(tot_true, tot_pred, os.path.join(results_dir, 'tot_confusion_matrix.png'))

    # ACT 모델 로드 및 평가
    act_baseline_expert = ACT_Baseline(cfg) # Enhancer 모델 초기화를 위해 필요
    enhancer_act = EnhancedACTModel(cfg, act_baseline_expert, context_expert)
    enhancer_act.load_state_dict(torch.load(path_enhancer_act, map_location=device))
    enhancer_act.to(device)
    ds_kwargs_act = {'data_map': dp.data_map, 'veh_cols': dp.veh_cols, 'fs': cfg.Data.fs, 'mode': 'act', 'window_sec': cfg.Data.window_sec_mot, 'window_stride': cfg.Data.window_stride_mot}
    act_test_ds = PKLMultiModalDatasetBaseline(participant_ids=dp.test_keys, **ds_kwargs_act)
    act_test_loader = DataLoader(act_test_ds, batch_size=cfg.Data.batch_size, shuffle=False, collate_fn=collate_fn_baseline)
    
    # 평가 함수로부터 성능과 시각화용 데이터를 함께 받음
    (test_mse_min_act, test_rmse_mean_act), (act_true, act_pred) = evaluate_enhancer_model(enhancer_act, act_test_loader, 'act', device)
    (main_mse_min_act, mean_act_rmse), (act_true, act_pred) = evaluate_enhancer_model(enhancer_act, act_test_loader, 'act', device)
    if act_true is not None:
        # 전체 + 세그먼트 일괄 저장
        save_act_plots_bulk(
            act_true, act_pred,
            out_dir=os.path.join(results_dir, "act_plots"),
            smooth="ma", window=81, dpi=300,
            segment_len=2000,  # 세그먼트 길이
            stride=2000,       # 오버랩 없이 자르려면 segment_len과 동일
            prefix="act"
        )

    # 컨텍스트 평가 직후
    motion_pt = os.path.join(weights_dir, "best_fusion_v28_assym.pt")  # 네가 가진 pt 경로로 변경 가능
    
    if os.path.exists(motion_pt):
        save_motion_plots_from_pt(
            motion_pt,
            out_dir=os.path.join(results_dir, "motion_plots"),
            segment_len=2000, stride=2000, dpi=300, prefix="motion"
        )
    else:
        print(f"{motion_pt} 가 없어서 스킵했어. (dump_motion_preds_to_pt로 먼저 생성 가능)")

    print("\n\n" + "="*30)
    print("      FINAL TEST RESULTS")
    print("="*30)
    print(f"Emotion/Motion -> Motion Acc: {test_acc_mot:.4f}, Valence Acc: {test_acc_v:.4f}, Arousal Acc: {test_acc_a:.4f}")
    print(f"Enhanced TOT   -> TOT Acc: {test_acc_tot:.4f}")
    print(f"Enhanced ACT   -> min ACT MSE: {test_mse_min_act:.4f}")
    print(f"Enhanced ACT   -> mean ACT RMSE: {test_rmse_mean_act:.4f}")
    print("="*30)
    print("\n📊 Evaluation plots saved in 'results' directory.")

if __name__ == "__main__":
    args = parse_args()
    main(profile=args.profile, smoke=args.smoke)
