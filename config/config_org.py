@dataclass
class Config:
    # Device setup
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data paths
    pkl_all: str = os.path.expanduser("/home/mintlab01/TOR/data/data/train/train.pkl")
    
    # Survey
    survey_csv: str = "/home/mintlab01/TOR/data/data/train/survey/pre_survey.csv"

    # Exclusions
    exclude: set = field(default_factory=lambda: set())

    # dataset split
    val_subjects: List[str]  = field(default_factory=lambda: ["21", "1", "9", "18", "31"])
    test_subjects: List[str] = field(default_factory=lambda: ["13", "16", "10", "30", "26"])
    # val_subjects: List[str]  = field(default_factory=lambda: ["3", "7", "22"])
    # test_subjects: List[str] = field(default_factory=lambda: ["4","12","19", "24", "28", "32"])
    seed: int = 42

    # Dataset mode & 윈도우 설정 (이제 여기서 모두 제어)
    emo_mode: str = "phase"               # "timeline" or "phase" or "motion"
    window_sec_emo: float = 10.0            
    window_stride_emo: float = 5.0          

    mot_mode: str = "motion"
    window_sec_mot: float = 5.0            
    window_stride_mot: float = 5.0  

    unified_win_sec:float = 10.0

    # Sampling & DataLoader
    fs: int = 100
    num_workers: int = 4
    
    # Task별 모달리티 바이어스 (초기값 예시)
    modality_bias: dict = field(default_factory=lambda: {
        'motion':  {'imu': 1.0, 'vehicle': 0.01, 'ppg': 0.01, 'scenario': 0.01, 'survey': 0.01},
        'emotion': {'imu': 1.0, 'vehicle': 1.0, 'ppg': 1.5, 'scenario': 1.0, 'survey': 1.0},
        'tot':     {'imu': 1.0, 'vehicle': 1.2, 'ppg': 1.0, 'scenario': 1.0, 'survey': 1.0},
        'act':     {'imu': 1.0, 'vehicle': 1.2, 'ppg': 1.0, 'scenario': 1.0, 'survey': 1.0},
    })

    # Sequence & batching
    seq_len:     int = 256
    pe_max_len:  int = 120
    batch_size:  int = 64
    apply_pe_before_lstm: bool = True

    lr: float=0.0001
    # Model hyperparameters
    hidden: int = 128
    static_dim: int = 8
    lr_motion: float  =0.0005
    weight_decay: float = 0.0001 
    lr_emotion: float = 0.0008
    epochs: int = 25
    patience: int = 10

    # ── Task & Model Dimensions ──────────────────────────────────────────
    num_motion: int = 4          # motion_weights 리스트의 길이와 일치 (phone, drink, watch, drive)
    num_valence: int = 2         # valence_weights 리스트의 길이와 일치
    num_arousal: int = 2         # arousal_weights 리스트의 길이와 일치

    # -- Auxiliary Task Dimensions --
    num_behavior_groups: int = 3 # 예: [정상, 위험, 안정] 등 행동 그룹 수
    aux_weight: float = 0.2        # 보조 손실(Auxiliary loss)의 가중치

    # Survey encoder
    survey_input_dim: int = 12

    # Fusion/TFT
    lr_fusion: float  = 1e-4
    lstm_layers: int        = 2
    dropout: float          = 0.2
    max_seq_len: int        = 100
    num_decoder_layers: int = 3
    num_heads: int          = 8
    num_motion: int         = 4
    num_valence: int        = 2   # classification bins (if used)
    num_arousal: int        = 2

    # 레이블 인덱스 0→Phone, 1→Drink, 2→Watch, 3→Drive
    motion_labels: List[str] = field(default_factory=lambda: ['Phone', 'Drink', 'Watch', 'Drive'])

    # task-weight scheduling
    lambda_motion:  float = 1.0  # 모션 손실의 기본 가중치
    lambda_emotion: float = 2.0  # 마지막 에폭에서 감정 손실 가중치

    # ── Fusion Experiment Params ───────────────────────────────────
    fusion_params = {
        "motion_modalities": ["imu", "veh", "sc"],
        "emotion_modalities": ["imu", "ppg", "veh", "sc", "survey"],
        
        # 학습시킬 모듈 제어
        "train_sm_fuse": True,
        "train_gm_fuse": True,
        "train_motion_predictor": True,
        "train_emotion_predictor": True
    }

    # Params for each encoder, Do not modify
    imu_params: dict = field(default_factory=lambda: {
        'input_dim': 14,
        'encoder_dim': 32,
        'num_layers': 2,
        'num_heads': 2,
        'ff_expansion': 4,
        'conv_expansion': 2,
        'input_dropout': 0.2,
        'ff_dropout': 0.2,
        'attn_dropout': 0.2,
        'conv_dropout': 0.2,
        'conv_kernel': 7,
        'half_step_residual': True
    })
    veh_params: dict = field(default_factory=lambda: {
        'input_dim': 12,
        'embed_dim': 64,
        'num_channels': 64,
        'num_layers': 4,
        'kernel_size': 3,
        'dropout': 0.1,
        'padding': 1,
    })
    ppg_params: dict = field(default_factory=lambda: {
        'embed_dim': 64,
        'cnn_channels': [16,32,64],
        'lstm_hidden': 128,
        'lstm_layers': 2,
        'attn1_dim': 128,
        'rr_dim': 32,
        'float_dim': 16,
        'dropout': 0.1,
    })
    sc_params: dict = field(default_factory=lambda: {
        'max_scenario_id': 50,
        'max_scenario_type': 5,
        'max_phase_id': 3,
        'vocab_size': 52,
        'embed_dim': 16,
        'dropout': 0.1,
    })
    survey_params: dict = field(default_factory=lambda: {
        'input_dim': 12,
        'hidden_dims': [32,16],
        'embed_dim': 8,
        'dropout': 0.1
    })

    # for variable‐selection network
    tv_sizes: dict = field(init=False)

    # Loss weights & ignore indices
    motion_weights: list    = field(default_factory=lambda: [1.25,1.2,1.7,1.15])
    # valence_weights: list   = field(default_factory=lambda: [1.3,1.1,1.5])
    # arousal_weights: list   = field(default_factory=lambda: [2.3,1.3,1.35])
    valence_weights: list   = field(default_factory=lambda: [1.0, 1.6])
    arousal_weights: list   = field(default_factory=lambda: [1.6, 1.0])

    lambda_val: float = 1.0
    lambda_aro: float = 1.0
    
    ign_tot: float = -100.0
    ign_mot: int   = -1
    ign_emo: int   = -100

    def __post_init__(self):
        # every modality project into hidden_size before fusion
        self.tv_sizes = {
            "imu": self.hidden,
            "veh": self.hidden,
            "ppg": self.hidden,
            "sc": self.hidden
        }

    @classmethod
    def from_yaml(cls, path):
        params = yaml.safe_load(open(path)) or {}
        default = cls().veh_params
        if 'veh_params' in params:
            # 기존 default 는 보존, YAML 에 있는 것만 덮어쓰기
            merged = {**default, **params['veh_params']}
            params['veh_params'] = merged
        return cls(**params)