"""
V11 기반, addictive Fusion + PMF (Progressive Modality Freezing) 적용
- Hard Freeze 해제, 차등 학습률을 사용한 Fine-tuning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from sklearn.metrics import accuracy_score
from config.config import Config
from copy import deepcopy
from tqdm import tqdm

from trainers.base_trainer import dataProcessor
from data.loader import make_multitask_loader # 멀티태스크 로더를 그대로 사용
from models.encoder.imu_encoder import IMUFeatureEncoder
from models.encoder.ppg_TCN_encoder import PPGEncoder
from models.encoder.veh_encoder import VehicleTCNEncoder
from models.encoder.sc_encoder import ScenarioEmbedding
from models.encoder.survey_encoder import PreSurveyEncoder

from models.fusion.predictors import EmotionPredictor
from models.head.motion_head import MotionHead

class FusionTrainer(nn.Module, dataProcessor):
    def __init__(self, cfg: Config, train_keys, val_keys, test_keys):
        super().__init__()
        dataProcessor.__init__(self, cfg)
        self.prepare()

        self.train_keys, self.val_keys, self.test_keys = train_keys, val_keys, test_keys
        self.cfg = cfg

        self._build_model()
        self._load_pretrained_weights()
        
        if self.cfg.MainTask.use_pmf:
            self._init_pmf()

        self._create_optimizer() 

        self.m2raw = {'imu':'imu', 'veh':'veh_m', 'sc':'sc', 'ppg':'ppg', 'survey':'survey'}
        self.e2raw = {'imu':'imu', 'veh':'veh_e', 'sc':'sc', 'ppg':'ppg', 'survey':'survey'}


    def _build_model(self):
        print("\n--- Building SIMPLIFIED model architecture based on config ---")
        hidden_dim = self.cfg.FusionModel.hidden_dim
        dropout_p = getattr(self.cfg.FusionModel, 'dropout', 0.5)

        mot_mods = set(getattr(self.cfg.PretrainMotion, 'modalities_to_use', []))
        emo_mods = set(getattr(self.cfg.PretrainEmotion, 'modalities_to_use', []))
        all_unique_modalities = mot_mods.union(emo_mods)
        print(f"Required modalities: {list(all_unique_modalities)}")

        self.nets = nn.ModuleDict()
        self.projs = nn.ModuleDict()

        if 'imu' in all_unique_modalities:
            self.nets['imu'] = IMUFeatureEncoder(self.cfg.Encoders.imu)
            self.projs['imu'] = nn.Linear(self.cfg.Encoders.imu['encoder_dim'], hidden_dim)
        if 'veh' in all_unique_modalities:
            self.nets['veh'] = VehicleTCNEncoder(self.cfg.Encoders.veh)
            self.projs['veh'] = nn.Linear(self.cfg.Encoders.veh['embed_dim'], hidden_dim)
        if 'ppg' in all_unique_modalities:
            self.nets['ppg'] = PPGEncoder(self.cfg.Encoders.ppg)
            self.projs['ppg'] = nn.Linear(self.cfg.Encoders.ppg['embed_dim'] + 6, hidden_dim)
        if 'sc' in all_unique_modalities:
            self.nets['sc'] = ScenarioEmbedding(self.cfg.Encoders.sc)
            self.projs['sc'] = nn.Linear(self.cfg.Encoders.sc['embed_dim'], hidden_dim)
        if 'survey' in all_unique_modalities:
            self.nets['survey'] = PreSurveyEncoder(self.cfg.Encoders.survey)
            self.projs['survey'] = nn.Linear(self.cfg.Encoders.survey['embed_dim'], hidden_dim)
            
        if mot_mods:
            self.motion_modalities = sorted(list(mot_mods))
            self.motion_feature_fusion = nn.Conv1d(hidden_dim * len(self.motion_modalities), hidden_dim, 1)
            self.motion_head = MotionHead(hidden_dim, self.cfg.PretrainMotion.num_motion)
            self.motion_joint_projection = nn.Sequential(
                nn.Conv1d(hidden_dim * 2, hidden_dim, 1),
                nn.Dropout(dropout_p)
            )

        if emo_mods:
            self.emotion_modalities = sorted([m for m in emo_mods if m != 'survey'])
            predictor_input_dim = hidden_dim * len(self.emotion_modalities)
            self.emotion_feature_fusion = nn.Sequential(
                nn.Linear(predictor_input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_p)
            )
            self.emotion_valence_predictor = EmotionPredictor(predictor_input_dim, hidden_dim, self.cfg.PretrainEmotion.num_valence)
            self.emotion_arousal_predictor = EmotionPredictor(predictor_input_dim, hidden_dim, self.cfg.PretrainEmotion.num_arousal)

        init_log_vars = torch.tensor([0.0, 0.0, 0.0, 0.0])
        self.log_vars = nn.Parameter(init_log_vars)
        self.scaler = torch.cuda.amp.GradScaler()
        self.to(self.cfg.Project.device)

    def _load_pretrained_weights(self):
        device = self.cfg.Project.device

        def _get_enc(ckpt):
            return ckpt.get('encoder', ckpt)

        def _safe_load(module, state_dict, prefix, tag):
            # state_dict가 None이거나 비어있으면 로드 시도 안 함
            if state_dict is None or not state_dict:
                print(f"[Load Skip] {tag}: State dictionary is empty or None.")
                return

            # prefix가 있으면 해당 부분만 잘라내고, 없으면 전체 사용
            if prefix:
                sub_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            else:
                sub_dict = state_dict

            if not sub_dict:
                print(f"[Load Skip] {tag}: No keys found with prefix '{prefix if prefix else 'N/A'}'")
                return
                
            msg = module.load_state_dict(sub_dict, strict=False)
            missing = getattr(msg, 'missing_keys', [])
            unexpected = getattr(msg, 'unexpected_keys', [])
            print(f"[Load OK] {tag}: {len(sub_dict)} keys loaded. (missing: {len(missing)}, unexpected: {len(unexpected)})")

        # ===========================
        # 1. 감정 모델 가중치 로드
        # ===========================
        emo_cfg = self.cfg.PretrainEmotion
        if getattr(emo_cfg, 'modalities_to_use', []):
            print("\n--- Loading Emotion Pre-trained Weights ---")
            emo_ckpt = torch.load(getattr(emo_cfg, 'ckpt_path', 'weights/best_emotion_tri.pt'), map_location=device)
            emo_enc_states = _get_enc(emo_ckpt)

            emo_map = { 'ppg': ('ppg', 'nets.ppg.', 'projs.ppg.'),
                        'sc':  ('sc',  'nets.sc.',  'projs.sc.'),
                        'veh': ('veh', 'nets.veh.', 'projs.veh.'),     # veh_e -> veh (생성된 키와 일치)
                        'survey': ('survey', 'nets.survey.', 'projs.survey.') }

            for m in emo_cfg.modalities_to_use:
                if m in emo_map:
                    attr, enc_prefix, proj_prefix = emo_map[m]
                    if hasattr(self.nets, attr): _safe_load(getattr(self.nets, attr), emo_enc_states, enc_prefix, f"Emotion Encoder[{m}]")
                    if attr in self.projs: _safe_load(self.projs[attr], emo_enc_states, proj_prefix, f"Emotion Projection[{m}]")
            
            # 감정 예측기(Valence/Arousal) 가중치 로드
            if hasattr(self, 'emotion_valence_predictor'):
                _safe_load(self.emotion_valence_predictor, emo_ckpt.get('valence_predictor'), None, "Emotion Valence Predictor")
            if hasattr(self, 'emotion_arousal_predictor'):
                _safe_load(self.emotion_arousal_predictor, emo_ckpt.get('arousal_predictor'), None, "Emotion Arousal Predictor")

        # ===========================
        # 2. 행동 모델 가중치 로드
        # ===========================
        mot_cfg = self.cfg.PretrainMotion
        if getattr(mot_cfg, 'modalities_to_use', []):
            print("\n--- Loading Motion Pre-trained Weights ---")
            mot_ckpt = torch.load(getattr(mot_cfg, 'ckpt_path', 'weights/best_motion.pt'), map_location=device)
            mot_enc_states = _get_enc(mot_ckpt)

            mot_map = { 'imu': ('imu', 'imu.', 'p_imu.'),
                        'veh': ('veh', 'veh.', 'p_veh_m.'),  # veh_m -> veh (생성된 키와 일치)
                        'sc' : ('sc',  'sc.',  'p_sc.') }
            
            for m in mot_cfg.modalities_to_use:
                if m in mot_map:
                    attr, enc_prefix, proj_prefix = mot_map[m]
                    if hasattr(self.nets, attr): _safe_load(getattr(self.nets, attr), mot_enc_states, enc_prefix, f"Motion Encoder[{m}]")
                    if attr in self.projs: _safe_load(self.projs[attr], mot_enc_states, proj_prefix, f"Motion Projection[{m}]")

            # 행동 헤드 가중치 로드
            if getattr(self.cfg, 'load_motion_head', True) and hasattr(self, 'motion_head'):
                _safe_load(self.motion_head, mot_ckpt.get('head'), None, 'Motion Head')
    
    def _process_hrv(self, batch, device):
        # Helper to process HRV features
        hrv_keys = ['ppg_rmssd_emotion', 'ppg_sdnn_emotion']
        rr = batch['ppg_rr_emotion'].to(device)
        hrv_list = [rr.mean(1, keepdim=True), rr.std(1, keepdim=True), rr.min(1, keepdim=True).values, rr.max(1, keepdim=True).values]
        hrv_list.extend([batch[k].to(device).unsqueeze(1) for k in hrv_keys])
        return torch.cat(hrv_list, dim=1)

    def _create_optimizer(self):
        base_lr = self.cfg.MainTask.lr
        
        # 사전 훈련된 인코더/프로젝터 파라미터 그룹 (작은 학습률 적용)
        pre_trained_params = list(self.nets.parameters()) + list(self.projs.parameters())

        # 새로 추가된 융합/예측 레이어 파라미터 그룹 (기본 학습률 적용)
        new_params = list(self.motion_feature_fusion.parameters()) + \
                     list(self.motion_joint_projection.parameters()) + \
                     list(self.emotion_feature_fusion.parameters()) + \
                     list(self.motion_head.parameters()) + \
                     list(self.emotion_valence_predictor.parameters()) + \
                     list(self.emotion_arousal_predictor.parameters())
        
        param_groups = [
            {'params': pre_trained_params, 'lr': base_lr * 0.1, 'weight_decay': self.cfg.MainTask.weight_decay},
            {'params': new_params, 'lr': base_lr, 'weight_decay': self.cfg.MainTask.weight_decay},
            {'params': [self.log_vars], 'lr': base_lr}
        ]
        self.optim = torch.optim.Adam(param_groups)
        print("Optimizer created with differential learning rates for adaptive fine-tuning.")

    # [PMF] PMF 관련 메서드 추가
    def _init_pmf(self):
        self.pmf_modalities = ['imu', 'ppg', 'veh_m', 'veh_e', 'sc', 'survey']
        self.loss_mask = {}
        self.memory_mask = {}
        print("PMF module initialized.")

    def setup_pmf_masks(self, num_samples):
        device = self.cfg.Project.device
        for key in self.pmf_modalities:
            self.loss_mask[key] = torch.ones(num_samples, dtype=torch.float, device=device)
            self.memory_mask[key] = torch.ones(num_samples, dtype=torch.float, device=device)
        print(f"PMF masks created for {num_samples} samples.")

    @torch.no_grad()
    def modal_freezing(self, loader):
        print("\n--- Updating PMF Masks for All Training Data ---")
        self.eval()
        all_relevance_scores = {m: [] for m in self.pmf_modalities}
        all_indices = {m: [] for m in self.pmf_modalities}

        for batch in tqdm(loader, desc="PMF Relevance Scoring"):
            indices = batch['indices'].to(self.cfg.Project.device)
            out = self.forward(batch, return_raw_features=True)
            raw_features = out['raw_features']
            
            for mod in self.pmf_modalities:
                if raw_features.get(mod) is not None:
                    relevance = torch.norm(raw_features[mod], p=2, dim=1)
                    all_relevance_scores[mod].append(relevance)
                    all_indices[mod].append(indices)
        
        print("  - Updating masks...")
        for m in self.pmf_modalities:
            if not all_relevance_scores[m]: continue
            
            scores = torch.cat(all_relevance_scores[m])
            indices = torch.cat(all_indices[m])
            
            # Scatter-add to handle multiple scores for the same sample index
            unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
            summed_scores = torch.zeros_like(unique_indices, dtype=torch.float).scatter_add_(0, inverse_indices, scores)
            
            q = 0.05 # 하위 5%를 동결
            threshold = torch.quantile(summed_scores, q)
            current_mask = (summed_scores >= threshold).float()
            
            # 한번 얼린 모달리티는 계속 얼린 상태로 유지 (memory_mask)
            final_mask = current_mask * self.memory_mask[m][unique_indices]
            self.loss_mask[m][unique_indices] = final_mask
            self.memory_mask[m][unique_indices] = final_mask
            
            frozen_ratio = 1 - self.loss_mask[m].mean().item()
            print(f"  - Modality [{m}]: Threshold={threshold:.4f}, Frozen Ratio={frozen_ratio:.2%}")

    def forward(self, batch, return_raw_features: bool = False):
        """
        Additive(+)-기반 + PMF 마스킹 forward
        - Motion Path: (B, T, H) 시퀀스 특징 + 감정 컨텍스트(B, H)를 시간축으로 확장한 후 더해 사용
        - Emotion Path: (B, H) 풀링 특징들을 concat해서 사용
        - PMF: 훈련 시, 모달리티별 풀링 특징(raw_features[mod])에 per-sample 마스크 곱함
        기대 batch 키:
        - 모션 쪽: imu_motion, veh_motion, sc_motion_evt/type/phase/time
        - 감정 쪽: ppg_emotion, veh_emotion, scenario_evt_e/type_e/phase_e/time_e, survey_e
        - 공통: indices (PMF용)
        """
        device = self.cfg.Project.device
        features = {'motion': {}, 'emotion': {}}
        raw_features = {}  # PMF를 위한 모달리티별 풀링 벡터 보관

        # --- 0) 길이/인덱스 준비 ---
        T_mot = batch['imu_motion'].shape[1] if 'imu_motion' in batch else 1000
        batch_indices = batch.get('indices', None)
        if batch_indices is not None:
            batch_indices = batch_indices.to(device)

        # --- 1) 모달리티별 특징 추출 (시퀀스/풀링 분리) ---
        # IMU: 모션 경로에선 시퀀스로 사용, PMF는 평균 풀링 벡터에만 적용
        if 'imu_motion' in batch and 'imu' in self.nets and 'imu' in self.projs:
            imu_len = (batch['imu_motion'].abs().sum(-1) > 0).sum(1)
            imu_seq = self.nets['imu'](batch['imu_motion'].to(device), imu_len)          # (B, T, E_im)
            imu_seq = self.projs['imu'](imu_seq)                                         # (B, T, H)
            raw_features['imu'] = imu_seq.mean(dim=1)                                    # (B, H)
            features['motion']['imu'] = imu_seq                                          # (B, T, H)

        # VEH (모션): 풀링 → expand
        #   - 구현에 따라 self.nets에 'veh' 하나만 있거나 'veh_m'/'veh_e'로 나눠질 수 있음
        veh_m_enc = 'veh_m' if 'veh_m' in self.nets else ('veh' if 'veh' in self.nets else None)
        veh_m_prj = 'veh_m' if 'veh_m' in self.projs else ('veh' if 'veh' in self.projs else None)
        if veh_m_enc and veh_m_prj and 'veh_motion' in batch:
            # 보통 veh 인코더는 (B, C, T) 또는 pooled 반환. 여기선 pooled를 받아 (B,H)로 투영했다고 가정
            veh_m_pooled = self.nets[veh_m_enc](batch['veh_motion'].to(device).permute(0, 2, 1), return_pooled=True)
            veh_m_vec = self.projs[veh_m_prj](veh_m_pooled)                                          # (B, H)
            raw_features['veh_m'] = veh_m_vec
            features['motion']['veh'] = veh_m_vec.unsqueeze(1).expand(-1, T_mot, -1)                # (B, T, H)

        # SC (모션): 풀링 → expand (시나리오 키는 _motion_ 접두)
        if 'sc' in self.nets and 'sc' in self.projs and \
        all(k in batch for k in ['sc_motion_evt', 'sc_motion_type', 'sc_motion_phase', 'sc_motion_time']):
            sc_pooled = self.nets['sc'](
                batch['sc_motion_evt'].to(device),
                batch['sc_motion_type'].to(device),
                batch['sc_motion_phase'].to(device),
                batch['sc_motion_time'].to(device)
            )  # (B, E_sc)
            sc_vec = self.projs['sc'](sc_pooled)                                          # (B, H)
            raw_features['sc'] = sc_vec
            features['motion']['sc'] = sc_vec.unsqueeze(1).expand(-1, T_mot, -1)         # (B, T, H)

        # PPG (감정): 시퀀스 → 프로젝션 → (B,T,H). PMF는 풀링 벡터에 적용.
        if 'ppg_emotion' in batch and 'ppg' in self.nets and 'ppg' in self.projs:
            ppg = batch['ppg_emotion'].to(device)
            if ppg.dim() == 2:  # (B, T) -> (B, T, 1)
                ppg = ppg.unsqueeze(-1)
            # TCN은 보통 (B, C, T) 반환 → (B, T, C)로 변환
            ppg_seq = self.nets['ppg'](ppg.permute(0, 2, 1), return_pooled=False).permute(0, 2, 1)  # (B, T, C)
            # HRV 확장해 채널 concat
            hrv = self._process_hrv(batch, device).unsqueeze(1).expand(-1, ppg_seq.shape[1], -1)    # (B, T, 6)
            ppg_seq = torch.cat([ppg_seq, hrv], dim=-1)                                             # (B, T, C+6)
            ppg_seq = self.projs['ppg'](ppg_seq)                                                    # (B, T, H)
            raw_features['ppg'] = ppg_seq.mean(dim=1)                                               # (B, H)
            # 모션 경로에 PPG를 보조로 쓸 경우: 풀링 벡터를 expand
            features['motion']['ppg'] = raw_features['ppg'].unsqueeze(1).expand(-1, T_mot, -1)      # (B, T, H)
            # 감정 경로에서는 풀링 벡터만 사용
            features['emotion']['ppg'] = raw_features['ppg']                                         # (B, H)

        # VEH (감정): 풀링 벡터
        veh_e_enc = 'veh_e' if 'veh_e' in self.nets else ('veh' if 'veh' in self.nets else None)
        veh_e_prj = 'veh_e' if 'veh_e' in self.projs else ('veh' if 'veh' in self.projs else None)
        if veh_e_enc and veh_e_prj and 'veh_emotion' in batch:
            veh_e_pooled = self.nets[veh_e_enc](batch['veh_emotion'].to(device).permute(0, 2, 1), return_pooled=True)  # (B, C)
            veh_e_vec = self.projs[veh_e_prj](veh_e_pooled)                                                             # (B, H)
            raw_features['veh_e'] = veh_e_vec
            features['emotion']['veh'] = veh_e_vec                                                                      # (B, H)

        # SC (감정): 풀링 벡터 (시나리오 키는 _e 접두)
        if 'sc' in self.nets and 'sc' in self.projs and \
        all(k in batch for k in ['scenario_evt_e', 'scenario_type_e', 'phase_evt_e', 'scenario_time_e']):
            sc_e = self.nets['sc'](
                batch['scenario_evt_e'].to(device),
                batch['scenario_type_e'].to(device),
                batch['phase_evt_e'].to(device),
                batch['scenario_time_e'].to(device)
            )  # (B, E_sc)
            sc_e_vec = self.projs['sc'](sc_e)                 # (B, H)
            # 모션 경로에서 이미 sc를 썼다면 덮어쓰지 않음
            if 'sc' not in raw_features:
                raw_features['sc'] = sc_e_vec
            features['emotion']['sc'] = sc_e_vec              # (B, H)

        # SURVEY (감정): 풀링 벡터
        if 'survey_e' in batch and 'survey' in self.nets and 'survey' in self.projs:
            survey_vec = self.projs['survey'](self.nets['survey'](batch['survey_e'].to(device)))  # (B, H)
            raw_features['survey'] = survey_vec
            features['emotion']['survey'] = survey_vec

        # 감정 경로에 IMU(평균) 추가 원하면:
        if 'imu_motion' in batch and 'imu' in self.nets and 'imu' in self.projs:
            features['emotion']['imu'] = raw_features.get('imu', imu_seq.mean(dim=1))  # (B, H)

        # --- 2) PMF 마스킹 (훈련시에만) ---
        if self.cfg.MainTask.use_pmf and self.training and batch_indices is not None:
            # 2-1) raw_features에 존재하고, loss_mask에도 있는 모달만 마스킹
            mot_raw_mods = [self.m2raw[m] for m in getattr(self, 'motion_modalities', []) if self.m2raw.get(m)]
            emo_raw_mods = [self.e2raw[m] for m in getattr(self, 'emotion_modalities', []) if self.e2raw.get(m)]

            for mod in mot_raw_mods:
                if mod in raw_features and mod in self.loss_mask:
                    raw_features[mod] = raw_features[mod] * self.loss_mask[mod][batch_indices].unsqueeze(1)

            for mod in emo_raw_mods:
                if mod in raw_features and mod in self.loss_mask:
                    raw_features[mod] = raw_features[mod] * self.loss_mask[mod][batch_indices].unsqueeze(1)

            # 2-2) 마스크 적용된 벡터를 다시 모션/감정 경로 텐서에 반영
            # IMU는 (B,T,H) 시퀀스라 직접 곱해줌
            if 'imu' in features['motion'] and 'imu' in self.loss_mask:
                m = self.loss_mask['imu'][batch_indices].view(-1, 1, 1)
                features['motion']['imu'] = features['motion']['imu'] * m

            # 나머지 모션 모달은 (B,H) 벡터 → (B,T,H)로 expand
            for m in getattr(self, 'motion_modalities', []):
                raw_k = self.m2raw.get(m)
                if m != 'imu' and m in features['motion'] and raw_k in raw_features:
                    features['motion'][m] = raw_features[raw_k].unsqueeze(1).expand(-1, T_mot, -1)

            # 감정 경로는 (B,H) 벡터 그대로 갱신
            for m in getattr(self, 'emotion_modalities', []):
                raw_k = self.e2raw.get(m)
                if m in features['emotion'] and raw_k in raw_features:
                    features['emotion'][m] = raw_features[raw_k]

        if return_raw_features:
            return {'raw_features': raw_features}

        # --- 3) Task-Specific Fusion ---
        # Motion: 각 모달(B,T,H)을 고정 순서(self.motion_modalities)로 정렬하고
        #         누락된 모달은 0으로 채워 입력 채널 수를 항상 hidden_dim*len(self.motion_modalities)로 맞춤
        B = None
        for v in features['motion'].values():
            B = v.shape[0]; break
        if B is None:
            # 배치 크기 추정 (어떤 키든 하나 사용)
            for k in ['imu_motion','veh_motion','ppg_emotion','veh_emotion','survey_e','scenario_evt_e','sc_motion_evt']:
                if k in batch:
                    B = batch[k].shape[0]; break
        if B is None:
            B = len(batch_indices) if batch_indices is not None else 1

        motion_tensors = []
        for m in getattr(self, 'motion_modalities', []):
            if m in features['motion']:
                motion_tensors.append(features['motion'][m])                  # (B,T,H)
            else:
                motion_tensors.append(torch.zeros(B, T_mot, self.cfg.FusionModel.hidden_dim,
                                                 device=device))             # 0-pad (B,T,H)
        if motion_tensors:
            motion_cat = torch.cat(motion_tensors, dim=2).permute(0, 2, 1)   # (B, H*, T)
            fused_motion_seq = self.motion_feature_fusion(motion_cat)        # (B, H, T)
        else:
            fused_motion_seq = torch.zeros((B, self.cfg.FusionModel.hidden_dim, max(T_mot,1)),
                                           device=device)

        # Emotion: (B,H)들을 고정 순서(self.emotion_modalities)로 정렬하고
        #          누락 모달은 0-패딩해서 Linear 입력 차원(H*len(emotion_modalities)) 유지
        emo_tensors = []
        for m in getattr(self, 'emotion_modalities', []):
            if m in features['emotion']:
                emo_tensors.append(features['emotion'][m])                     # (B,H)
            else:
                emo_tensors.append(torch.zeros(B, self.cfg.FusionModel.hidden_dim,
                                               device=device))                 # 0-pad (B,H)
        if emo_tensors:
            emotion_cat = torch.cat(emo_tensors, dim=1)                        # (B, H*)
        else:
            emotion_cat = torch.zeros(B, self.cfg.FusionModel.hidden_dim,
                                      device=device)
        fused_emotion_vector = self.emotion_feature_fusion(emotion_cat)        # (B, H)

        # --- 4) Cross-Task Prediction (Additive) ---
        # 감정 컨텍스트(B,H)를 시간축으로 확장해 모션 특징(B,H,T)과 더함
        emotion_context_for_mot = fused_emotion_vector.unsqueeze(2).expand_as(fused_motion_seq)  # (B, H, T)
        motion_input_for_head = fused_motion_seq + emotion_context_for_mot
        mot_logits = self.motion_head(motion_input_for_head)  # (B, C_mot)

        # 감정 예측: 모션 컨텍스트 평균을 제공(+ survey 정적 컨텍스트 선택적 추가)
        motion_context_for_emo = fused_motion_seq.mean(dim=2).detach()  # (B, H)
        emotion_input_dict = {'fused': emotion_cat.unsqueeze(1)}        # (B, 1, H*)
        if 'survey' in features['emotion']:
            emotion_input_dict['static'] = features['emotion']['survey']  # (B, H)

        valence_logits = self.emotion_valence_predictor(emotion_input_dict, context=motion_context_for_emo)
        arousal_logits = self.emotion_arousal_predictor(emotion_input_dict, context=motion_context_for_emo)

        return {
            'motion_logits': mot_logits,
            'valence_logits': valence_logits,
            'arousal_logits': arousal_logits,
            'fused_motion': fused_motion_seq.mean(dim=2),   # (B, H)
            'fused_emotion': fused_emotion_vector          # (B, H)
        }
    
    def fusion_train(self):
        # [PMF] 데이터 로더에 include_indices=True 추가
        tr_loader = make_multitask_loader(self.cfg, self.train_keys, shuffle=True, dp=self, include_indices=True)
        va_loader = make_multitask_loader(self.cfg, self.val_keys, shuffle=False, dp=self)
        
        # [PMF] 데이터 로더 생성 후 마스크 초기화
        if self.cfg.MainTask.use_pmf:
            self.setup_pmf_masks(len(tr_loader.dataset))
        
        best_loss, patience_counter, best_performance = float('inf'), 0, {}

        for epoch in range(1, self.cfg.MainTask.epochs + 1):
            # [PMF] 에포크 시작 전 마스크 업데이트
            if self.cfg.MainTask.use_pmf and epoch >= self.cfg.MainTask.pmf_start_epoch:
                self.modal_freezing(tr_loader)

            tr_losses = self.run_epoch(tr_loader, train=True)
            va_losses = self.run_epoch(va_loader, train=False)
            
            va_acc_mot, va_acc_v, va_acc_a = self.evaluate(va_loader)
            
            tr_loss_str = (f"Tr Loss: {tr_losses['total']:.3f} (M:{tr_losses['motion']:.2f}, V:{tr_losses['valence']:.2f}, A:{tr_losses['arousal']:.2f}, C:{tr_losses['cross']:.2f})")
            va_loss_str = f"Val Loss: {va_losses['total']:.3f}"
            print(f"Epoch {epoch:02d} | {tr_loss_str} | {va_loss_str} | Acc(M/V/A): {va_acc_mot:.3f}/{va_acc_v:.3f}/{va_acc_a:.3f}")

            current_val_loss = va_losses['total']
            if current_val_loss < best_loss:
                best_loss, patience_counter = current_val_loss, 0
                best_performance = {'mot_acc': va_acc_mot, 'val_acc': va_acc_v, 'aro_acc': va_acc_a, 'best_loss': best_loss}
                
                torch.save({'model_state_dict': self.state_dict(), 'optimizer_state_dict': self.optim.state_dict()}, "weights/best_fusion_v15.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.MainTask.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        print(f"\n--- Best Validation Performance ---")
        print(f"Loss: {best_performance.get('best_loss', 'N/A'):.4f} | Acc(M/V/A): {best_performance.get('mot_acc', 0):.3f}/{best_performance.get('val_acc', 0):.3f}/{best_performance.get('aro_acc', 0):.3f}")
        return best_performance

    def run_epoch(self, loader, train: bool):
        self.train(train)
        epoch_losses = {'total': 0.0, 'motion': 0.0, 'valence': 0.0, 'arousal': 0.0, 'cross': 0.0}
        num_batches = 0
        device = self.cfg.Project.device

        for batch in loader:
            with torch.set_grad_enabled(train):
                out = self.forward(batch) 

                # --- 1) Main Task Losses ---
                mot_logits, mot_labels = out['motion_logits'], batch['label_motion'].to(device)
                
                valid_mask = (mot_labels.reshape(-1) > 0) & (mot_labels.reshape(-1) != 4)
                loss_mot = F.cross_entropy(mot_logits.reshape(-1, mot_logits.shape[-1])[valid_mask], (mot_labels.reshape(-1)[valid_mask] - 1).long()) if valid_mask.any() else torch.tensor(0., device=device)

                val_logits = out['valence_logits']
                if val_logits.dim() == 3:  # (B,1,C) 방어
                    val_logits = val_logits.squeeze(1)
                raw_v = batch['valence_reg_emotion'].to(device).view(-1)
                tgt_v = torch.full_like(raw_v, -100, dtype=torch.long); tgt_v[raw_v < 4]=0; tgt_v[(raw_v>=4)&(raw_v<7)]=1; tgt_v[raw_v>=7]=2
                loss_v = F.cross_entropy(val_logits, tgt_v, ignore_index=-100)
                
                aro_logits = out['arousal_logits']
                if aro_logits.dim() == 3:
                    aro_logits = aro_logits.squeeze(1)
                raw_a = batch['arousal_reg_emotion'].to(device).view(-1)
                tgt_a = torch.full_like(raw_a, -100, dtype=torch.long); tgt_a[raw_a < 4]=0; tgt_a[(raw_a>=4)&(raw_a<7)]=1; tgt_a[raw_a>=7]=2
                loss_a = F.cross_entropy(aro_logits, tgt_a, ignore_index=-100)
                
                cos_sim = F.cosine_similarity(out['fused_motion'], out['fused_emotion'], dim=1)
                loss_cross = (1 - cos_sim).mean()

                # Total Loss with Uncertainty Weighting
                loss = (torch.exp(-self.log_vars[0]) * loss_mot + 0.5 * self.log_vars[0]) + \
                       (torch.exp(-self.log_vars[1]) * loss_v + 0.5 * self.log_vars[1]) + \
                       (torch.exp(-self.log_vars[2]) * loss_a + 0.5 * self.log_vars[2]) + \
                       (torch.exp(-self.log_vars[3]) * loss_cross * self.cfg.MainTask.cross_modal_lambda + 0.5 * self.log_vars[3])

            epoch_losses['total'] += loss.item()
            epoch_losses['motion'] += loss_mot.item()
            epoch_losses['valence'] += loss_v.item()
            epoch_losses['arousal'] += loss_a.item()
            epoch_losses['cross'] += loss_cross.item()
            num_batches += 1

            if train:
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()

        
        return {k: v / num_batches for k, v in epoch_losses.items()} if num_batches > 0 else epoch_losses
    
    @torch.no_grad()
    def evaluate(self, loader, return_preds=False):
        # This method is simplified for brevity but logic remains the same
        self.eval()
        all_preds = {'motion_preds': [], 'motion_trues': [], 'valence_preds': [], 'valence_trues': [], 'arousal_preds': [], 'arousal_trues': []}

        for batch in loader:
            out = self.forward(batch)
            
            p_mot = out['motion_logits'].argmax(-1).cpu(); t_mot_raw = batch['label_motion'].cpu()
            mask_mot = (t_mot_raw > 0) & (t_mot_raw != 4)
            all_preds['motion_preds'].append(p_mot[mask_mot]); all_preds['motion_trues'].append(t_mot_raw[mask_mot] - 1)
            
            p_v = out['valence_logits'].argmax(-1).cpu(); raw_v = batch['valence_reg_emotion'].view(-1).cpu()
            t_v = torch.full_like(raw_v, -1, dtype=torch.long); t_v[raw_v<4]=0; t_v[(raw_v>=4)&(raw_v<7)]=1; t_v[raw_v>=7]=2
            m_v = t_v != -1
            all_preds['valence_preds'].append(p_v[m_v]); all_preds['valence_trues'].append(t_v[m_v])
            
            p_a = out['arousal_logits'].argmax(-1).cpu(); raw_a = batch['arousal_reg_emotion'].view(-1).cpu()
            t_a = torch.full_like(raw_a, -1, dtype=torch.long); t_a[raw_a<4]=0; t_a[(raw_a>=4)&(raw_a<7)]=1; t_a[raw_a>=7]=2
            m_a = t_a != -1
            all_preds['arousal_preds'].append(p_a[m_a]); all_preds['arousal_trues'].append(t_a[m_a])

        for key in all_preds: all_preds[key] = torch.cat(all_preds[key])
        
        if return_preds:
            return all_preds
        
        acc_mot = accuracy_score(all_preds['motion_trues'], all_preds['motion_preds'])
        acc_v = accuracy_score(all_preds['valence_trues'], all_preds['valence_preds'])
        acc_a = accuracy_score(all_preds['arousal_trues'], all_preds['arousal_preds'])
        return acc_mot, acc_v, acc_a
