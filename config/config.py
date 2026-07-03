import os
from pathlib import Path
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


@dataclass
class ProjectConfig:
    """프로젝트 전반에 걸친 기본 설정 (경로, 시드 등)"""
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42
    profile: str = "auto"
    root_dir: str = field(default_factory=lambda: project_path())
    weights_dir: str = field(default_factory=lambda: project_path("weights"))
    results_dir: str = field(default_factory=lambda: project_path("results"))
    # 데이터 경로
    pkl_all: str = field(default_factory=lambda: project_path("data", "data", "train", "train_ver2.pkl"))
    survey_csv: str = field(default_factory=lambda: project_path("data", "sample", "sample_survey.csv"))


@dataclass
class DataConfig:
    """데이터 분할, 샘플링, 로더 관련 설정"""
    # 데이터 분할 (피실험자 ID)
    test_subjects: List[str] = field(default_factory=lambda: ['22', '107', '112', '24', '30'])
    val_subjects: List[str] = field(default_factory=lambda: ['10', '111', '101', '26', '31'])
    # 데이터로더
    num_workers: int = 0
    batch_size: int = 32 # 확정
    # 샘플링
    fs: int = 100
    seq_len: int = 1000
    # 윈도우 크기 (초)
    emo_mode: str = "timeline"               
    window_sec_emo: float = 3.0 # 확정      
    window_stride_emo: float = 1.5 # 확정

    mot_mode: str = "motion"
    window_sec_mot: float = 3.0 # 확정           
    window_stride_mot: float = 3.0 # 확정

@dataclass
class PretrainTaskConfig:
    """사전학습 태스크 공통 설정"""
    epochs: int = 25
    patience: int = 5
    weight_decay: float = 5e-5 # 확정
    hidden_dim: int = 128 #확정

@dataclass
class PretrainEmotionConfig(PretrainTaskConfig):
    """감정 인식 사전학습 관련 설정"""
    lr: float = 0.0006
    dropout: float = 0.2
    num_valence: int = 9
    num_arousal: int = 9
    
    valence_weights: List[float] = field(default_factory=lambda: [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,])
    # [0.95, 1.15, 0.62, 1.09, 0.51, 1.1, 0.96, 2.11, 6.56], [2.11, 1.58, 0.9, 1.06, 0.62, 0.8, 0.72, 1.1, 1.51]
    arousal_weights: List[float] = field(default_factory=lambda: [1.0, 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,])
    # valence_weights: List[float] = field(default_factory=lambda: [1.6, 1.1]) # 1.4, 1.1
    # arousal_weights: List[float] = field(default_factory=lambda: [1.6, 1.0]) # 1.45, 1.0
    lambda_valence: float = 0.5 # 확정
    lambda_arousal: float = 0.7 # 확정
    ignore_index: int = -100
    modalities_to_use: List[str] = field(default_factory=lambda: ['ppg', 'sc', 'survey'])
    ckpt_path: str = field(default_factory=lambda: project_path("weights", "best_emotion_ppg_sc_survey.pt"))
    

@dataclass
class PretrainMotionConfig(PretrainTaskConfig):
    """모션 인식 사전학습 관련 설정"""
    lr: float = 0.0005
    num_motion: int = 3
    class_labels: List[str] = field(default_factory=lambda: ['Phone', 'Drink', 'Watch'])
    class_weights: List[float] = field(default_factory=lambda: [1.0, 1.1, 1.5]) # 1.0, 1.0, 2.0 (확정)
    modalities_to_use: List[str] = field(default_factory=lambda: ['imu', 'veh'])
    ignore_index: int = -1
    ckpt_path: str = field(default_factory=lambda: project_path("weights", "best_motion_imu_veh.pt"))


@dataclass
class MainTaskConfig:
    """메인 융합 모델 학습 관련 설정"""
    epochs: int = 50
    patience: int = 4
    lr: float = 1e-5
    weight_decay: float = 5e-4

    # v22
    lr_asym_ratio: float = 0.1      # Cautious 그룹의 학습률 비율
    wd_asym_ratio: float = 2.0      # Cautious 그룹의 가중치 감쇠 비율
    use_uncertainty_loss: bool = True # 불확실성 기반 손실 함수 사용 여부
    
    # Cross-modal Loss 관련 설정
    cross_modal_lambda: float = 0.2   # 상호 정렬 손실의 가중치

    lambda_emotion: float = 2.5
    lambda_tot: float = 1.0
    lambda_act: float = 1.0

    # PMF (Progressive Modality Freezing) 관련 설정
    use_pmf: bool = True
    pmf_start_epoch: int = 2          # PMF를 시작할 에폭
    pmf_growing_rate: float = 1.05    # 에폭마다 임계값이 증가하는 비율
    pmf_max_theta: float = 0.8        # 임계값의 최대치

@dataclass
class FusionModelConfig:
    """메인 융합 모델의 구조 관련 하이퍼파라미터"""
    hidden_dim: int = 128
    static_dim: int = 8  
    max_seq_len: int = 100 # 모델이 처리할 최대 시퀀스 길이
    dropout: float = 0.3
    
    num_heads: int = 2
    n_task_tokens: int = 2
    emo_delay_steps: int = 50      # Δ
    emo_ctx_window: int = 200      # 과거 윈도우 길이
    emo_use_mot_logits: bool = True
    interaction_mode: str = 'cross_attention_film' # 'cross_attention_only', 'film_only', 'cross_attention_film'

@dataclass
class EncoderParams:
    """각 모달리티 인코더의 하이퍼파라미터"""
    imu: Dict = field(default_factory=lambda: {
        'input_dim': 14, 'encoder_dim': 128, 'num_layers': 3, 'num_heads': 2,
        'ff_expansion': 4, 'conv_expansion': 2, 'input_dropout': 0.2,
        'ff_dropout': 0.2, 'attn_dropout': 0.2, 'conv_dropout': 0.2,
        'conv_kernel': 7, 'half_step_residual': True
    })
    veh: Dict = field(default_factory=lambda: {
        'input_dim': 12, 'embed_dim': 64, 'num_channels': 64, 'num_layers': 5,
        'kernel_size': 3, 'dropout': 0.1, 'padding': 1,
    })
    ppg: Dict = field(default_factory=lambda: {
        'embed_dim': 64, 'cnn_channels': [16, 32, 64], 'lstm_hidden': 128,
        'lstm_layers': 2, 'attn1_dim': 128, 'rr_dim': 32, 'float_dim': 16,
        'dropout': 0.3,
    })
    sc: Dict = field(default_factory=lambda: {
        'vocab_size': 52, 'embed_dim': 16, 'dropout': 0.1,'max_scenario_type': 46,
        'max_phase_id': 3 
    })
    survey: Dict = field(default_factory=lambda: {
        'input_dim': 31, 'hidden_dims': [32, 16], 'embed_dim': 8, 'dropout': 0.1
    })


@dataclass
class BaselineTaskConfig:
    """TOT/ACT 베이스라인 모델 학습 설정"""
    epochs: int = 30
    patience: int = 5
    lr: float = 0.0005  # 조금 더 높은 학습률로 시작
    weight_decay: float = 1e-5
    # 베이스라인 모델은 VEH, SC 데이터만 사용
    modalities_to_use: List[str] = field(default_factory=lambda: ['veh', 'sc'])

@dataclass
class EnhancerTaskConfig:
    """TOT/ACT 베이스라인 모델 학습 설정"""
    epochs: int = 50
    patience: int = 5
    lr: float = 0.0001  # 조금 더 높은 학습률로 시작
    weight_decay: float = 1e-5
    # 베이스라인 모델은 VEH, SC 데이터만 사용
    patience: float = 8


@dataclass
class TOTConfig:
    """TOT classifier settings used by models.totact_models.TOT_Baseline."""
    num_classes: int = 3
    ignore_index: int = -100
    use_modalities: List[str] = field(default_factory=lambda: ["veh", "sc"])
    feat_dim: int = 32
    attn_pool: bool = True
    gru_layers: int = 1
    hidden: Optional[int] = None

    # Enhancer options. The model has defaults, but keeping them explicit avoids
    # cfg.TOT attribute errors when running older/newer model variants.
    enh_hid: int = 128
    enh_use_residual: bool = True
    enh_use_prob: bool = True
    enh_norm_each: bool = True
    enh_gate_init: float = 0.10
    enh_time_pool: str = "mean"
    enh_fuse_mode: str = "logit_add"
    enh_ctx_source: str = "prob"


@dataclass
class ACTConfig:
    """ACT regressor settings used by models.totact_models.ACT_Baseline."""
    use_modalities: List[str] = field(default_factory=lambda: ["veh", "sc"])
    feat_dim: int = 64
    gru_layers: int = 1
    hidden: Optional[int] = None

@dataclass
class Config:
    """
    프로젝트의 모든 설정을 관리하는 통합 클래스.
    의미 단위로 그룹화된 하위 설정 객체들을 포함합니다.
    """
    # --- 기본 및 데이터 설정 ---
    Project: ProjectConfig = field(default_factory=ProjectConfig)
    Data: DataConfig = field(default_factory=DataConfig)
    
    # --- 사전학습 단계별 설정 ---
    PretrainMotion: PretrainMotionConfig = field(default_factory=PretrainMotionConfig)
    PretrainEmotion: PretrainEmotionConfig = field(default_factory=PretrainEmotionConfig)
    
    # --- 메인 학습 단계 설정 ---
    MainTask: MainTaskConfig = field(default_factory=MainTaskConfig)

    # --- 모델 구조 관련 설정 ---
    Encoders: EncoderParams = field(default_factory=EncoderParams)
    FusionModel: FusionModelConfig = field(default_factory=FusionModelConfig)

    BaselineTask: BaselineTaskConfig = field(default_factory=BaselineTaskConfig)
    EnhancerTask: EnhancerTaskConfig = field(default_factory=EnhancerTaskConfig)
    TOT: TOTConfig = field(default_factory=TOTConfig)
    ACT: ACTConfig = field(default_factory=ACTConfig)

    def __post_init__(self):
        self.apply_profile(os.environ.get("MUST_PROFILE", self.Project.profile))

    def apply_profile(self, profile: str = "auto"):
        """Resolve runtime paths relative to the repository root.

        ``auto`` keeps the local full dataset when it exists, otherwise falls
        back to the synthetic sample data committed with the repository.
        """
        full_pkl = project_path("data", "data", "train", "train_ver2.pkl")
        sample_pkl = project_path("data", "sample", "sample_train.pkl")
        sample_survey = project_path("data", "sample", "sample_survey.csv")

        if profile == "auto":
            profile = "full" if Path(full_pkl).exists() else "sample"

        self.Project.profile = profile
        self.Project.root_dir = project_path()
        self.Project.weights_dir = project_path("weights")
        self.Project.results_dir = project_path("results")

        if profile == "sample":
            self.Project.pkl_all = sample_pkl
            self.Project.survey_csv = sample_survey
            self.Data.test_subjects = ["7", "8"]
            self.Data.val_subjects = ["5", "6"]
            self.Data.batch_size = 2
            self.Data.num_workers = 0
            self.Data.window_sec_emo = 1.0
            self.Data.window_stride_emo = 1.0
            self.Data.window_sec_mot = 1.0
            self.Data.window_stride_mot = 1.0
        elif profile == "full":
            self.Project.pkl_all = full_pkl
            if not Path(self.Project.survey_csv).exists():
                self.Project.survey_csv = sample_survey
        else:
            raise ValueError(f"Unknown MUST profile: {profile}")

        self.Project.pkl_all = os.environ.get("MUST_PKL_ALL", self.Project.pkl_all)
        self.Project.survey_csv = os.environ.get("MUST_SURVEY_CSV", self.Project.survey_csv)
        self.PretrainEmotion.ckpt_path = os.environ.get(
            "MUST_EMOTION_CKPT",
            project_path("weights", "best_emotion_ppg_sc_survey.pt"),
        )
        self.PretrainMotion.ckpt_path = os.environ.get(
            "MUST_MOTION_CKPT",
            project_path("weights", "best_motion_imu_veh.pt"),
        )
        return self
