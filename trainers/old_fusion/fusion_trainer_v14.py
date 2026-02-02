"""
V13 -> 데이터로더 분리 (모션 로스가 이상해서 수정 필요)
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
    def __init__(self, cfg, mot_train_loader, emo_train_loader, 
                 mot_val_loader, emo_val_loader, 
                 mot_test_loader, emo_test_loader):
        super().__init__()
        
        self.cfg = cfg
        self.device = cfg.Project.device
        
        # [핵심 수정] 분리된 훈련 로더를 각각 저장합니다.
        self.mot_train_loader, self.emo_train_loader = mot_train_loader, emo_train_loader
        self.mot_val_loader, self.mot_test_loader = mot_val_loader, mot_test_loader
        self.emo_val_loader, self.emo_test_loader = emo_val_loader, emo_test_loader
                
        self._build_model()
        self._load_pretrained_weights()
            
        if self.cfg.MainTask.use_pmf:
            self._init_pmf()
            
            # [수정] 이제 self.mot_train_loader가 존재하므로 이 코드가 정상 동작합니다.
            num_samples = max(len(self.mot_train_loader.dataset), len(self.emo_train_loader.dataset))
            self.setup_pmf_masks(num_samples)

        self._create_optimizer()

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

            emo_map = { 'ppg': ('ppg', 'nets.ppg.', 'projs.ppg.'), 'sc': ('sc', 'nets.sc.', 'projs.sc.'), 'veh': ('veh', 'nets.veh.', 'projs.veh.'), 'survey': ('survey', 'nets.survey.', 'projs.survey.') }

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

            mot_map = { 'imu': ('imu', 'imu.', 'p_imu.'), 'veh': ('veh', 'veh.', 'p_veh.'), 'sc': ('sc', 'sc.', 'p_sc.') }
            
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
        pre_trained_params = list(self.nets.parameters()) + list(self.projs.parameters())
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

    def _init_pmf(self):
        self.pmf_modalities = ['imu', 'ppg', 'veh', 'sc', 'survey']
        self.loss_mask, self.memory_mask = {}, {}
        print("PMF module initialized.")

    def setup_pmf_masks(self, num_samples):
        for key in self.pmf_modalities:
            self.loss_mask[key] = torch.ones(num_samples, dtype=torch.float, device=self.device)
            self.memory_mask[key] = torch.ones(num_samples, dtype=torch.float, device=self.device)
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

    def forward(self, mot_batch=None, emo_batch=None, return_raw_features=False):
        device = self.device
        raw_features = {}
        hidden_dim = self.cfg.FusionModel.hidden_dim
        # --- 1. 특징 추출 ---
        if mot_batch is not None:
            imu_len = (mot_batch['imu_motion'].abs().sum(-1) > 0).sum(1)
            raw_features['imu'] = self.projs['imu'](self.nets['imu'](mot_batch['imu_motion'].to(device), imu_len).mean(dim=1))
            raw_features['veh'] = self.projs['veh'](self.nets['veh'](mot_batch['veh_motion'].to(device), return_pooled=True))
            raw_features['sc'] = self.projs['sc'](self.nets['sc'](mot_batch['sc_motion_evt'].to(device), mot_batch['sc_motion_type'].to(device), mot_batch['sc_motion_phase'].to(device), mot_batch['sc_motion_time'].to(device)))

        if emo_batch is not None:
            ppg_tcn = self.nets['ppg'](emo_batch['ppg_emotion'].to(device).permute(0, 2, 1))
            combined = torch.cat([ppg_tcn, self._process_hrv(emo_batch, device)], dim=1)
            raw_features['ppg'] = self.projs['ppg'](combined)
            raw_features['survey'] = self.projs['survey'](self.nets['survey'](emo_batch['survey_e'].to(device)))
            if 'veh' not in raw_features:
                raw_features['veh'] = self.projs['veh'](self.nets['veh'](emo_batch['veh_emotion'].to(device).permute(0, 2, 1), return_pooled=True))
            if 'sc' not in raw_features:
                raw_features['sc'] = self.projs['sc'](self.nets['sc'](emo_batch['scenario_evt_e'].to(device), emo_batch['scenario_type_e'].to(device), emo_batch['phase_evt_e'].to(device), emo_batch['scenario_time_e'].to(device)))
            if 'imu' not in raw_features:
                imu_len = (emo_batch['imu_emotion'].abs().sum(-1) > 0).sum(1)
                raw_features['imu'] = self.projs['imu'](self.nets['imu'](emo_batch['imu_emotion'].to(device), imu_len).mean(dim=1))

        if return_raw_features: return {'raw_features': raw_features}
        
        batch_indices = (emo_batch or mot_batch).get('indices')
        if batch_indices is not None: batch_indices = batch_indices.to(device)

        if self.cfg.MainTask.use_pmf and self.training and batch_indices is not None:
            for mod in self.pmf_modalities:
                if mod in raw_features:
                    raw_features[mod] = raw_features[mod] * self.loss_mask[mod][batch_indices].unsqueeze(1)
        
        mot_logits, val_logits, aro_logits = None, None, None
        fused_motion_final, fused_emotion_final = None, None

        if mot_batch is not None and emo_batch is not None:
            T_mot = mot_batch['imu_motion'].shape[1]
            # [수정] KeyError 방지를 위해 raw_features에 키가 있는지 확인
            features_mot = {m: raw_features[m].unsqueeze(1).expand(-1, T_mot, -1) for m in self.motion_modalities if m in raw_features}
            motion_cat = torch.cat([features_mot[m] for m in self.motion_modalities if m in features_mot], dim=2).permute(0, 2, 1)
            fused_motion_seq = self.motion_feature_fusion(motion_cat)
            
            features_emo = {m: raw_features[m] for m in self.emotion_modalities if m in raw_features}
            emotion_cat = torch.cat([features_emo[m] for m in self.emotion_modalities if m in features_emo], dim=1)
            fused_emotion_vector = self.emotion_feature_fusion(emotion_cat)

            emotion_context = fused_emotion_vector.unsqueeze(2).expand_as(fused_motion_seq)
            motion_input = self.motion_joint_projection(torch.cat([fused_motion_seq, emotion_context], dim=1))
            mot_logits = self.motion_head(motion_input)
            
            motion_context = fused_motion_seq.mean(dim=2).detach()
            emo_input_dict = {'fused': emotion_cat.unsqueeze(1), 'static': raw_features.get('survey')}
            val_logits = self.emotion_valence_predictor(emo_input_dict, context=motion_context)
            aro_logits = self.emotion_arousal_predictor(emo_input_dict, context=motion_context)
            
            fused_motion_final = fused_motion_seq.mean(dim=2)
            fused_emotion_final = fused_emotion_vector

        elif mot_batch is not None:
             B = mot_batch['imu_motion'].shape[0]
             T_mot = mot_batch['imu_motion'].shape[1]
             
             # 모달리티가 없으면 0으로 채운 텐서를 생성하여 리스트에 추가
             feature_list_for_concat = []
             for mod in self.motion_modalities:
                 if mod in raw_features:
                     feature_list_for_concat.append(raw_features[mod].unsqueeze(1).expand(-1, T_mot, -1))
                 else:
                     zero_tensor = torch.zeros(B, T_mot, hidden_dim, device=device)
                     feature_list_for_concat.append(zero_tensor)

             motion_cat = torch.cat(feature_list_for_concat, dim=2).permute(0, 2, 1)
             fused_motion_seq = self.motion_feature_fusion(motion_cat)
             mot_logits = self.motion_head(fused_motion_seq)

        elif emo_batch is not None:
            B = emo_batch['imu_emotion'].shape[0]

            # 모달리티가 없으면 0으로 채운 텐서를 생성하여 리스트에 추가
            feature_list_for_concat = []
            for mod in self.emotion_modalities:
                 if mod in raw_features:
                     feature_list_for_concat.append(raw_features[mod])
                 else:
                     zero_tensor = torch.zeros(B, hidden_dim, device=device)
                     feature_list_for_concat.append(zero_tensor)

            emotion_cat = torch.cat(feature_list_for_concat, dim=1)
            emo_input_dict = {'fused': emotion_cat.unsqueeze(1), 'static': raw_features.get('survey')}
            val_logits = self.emotion_valence_predictor(emo_input_dict, context=None)
            aro_logits = self.emotion_arousal_predictor(emo_input_dict, context=None)

        return {
            'motion_logits': mot_logits, 'valence_logits': val_logits, 'arousal_logits': aro_logits,
            'fused_motion': fused_motion_final, 'fused_emotion': fused_emotion_final
        }
    
    def run_epoch(self, train: bool):
        self.train(train)
        epoch_losses = {'total': 0.0, 'motion': 0.0, 'valence': 0.0, 'arousal': 0.0, 'cross': 0.0}
        
        mot_loader = self.mot_train_loader if train else self.mot_val_loader
        emo_loader = self.emo_train_loader if train else self.emo_val_loader
        
        if len(mot_loader) >= len(emo_loader):
            iterator = tqdm(zip(mot_loader, itertools.cycle(emo_loader)), total=len(mot_loader), desc=f"Epoch ({'Train' if train else 'Val'})")
        else:
            iterator = tqdm(zip(itertools.cycle(mot_loader), emo_loader), total=len(emo_loader), desc=f"Epoch ({'Train' if train else 'Val'})")

        for mot_batch, emo_batch in iterator:
            # [핵심 수정] 마지막 배치의 크기 불일치 문제를 해결하기 위한 로직
            b_mot = mot_batch['imu_motion'].shape[0]
            b_emo = emo_batch['imu_emotion'].shape[0]
            if b_mot != b_emo:
                min_b = min(b_mot, b_emo)
                # 더 큰 배치의 모든 텐서를 작은 쪽에 맞춰 잘라냅니다.
                if b_mot > min_b:
                    mot_batch = {k: v[:min_b] for k, v in mot_batch.items()}
                if b_emo > min_b:
                    emo_batch = {k: v[:min_b] for k, v in emo_batch.items()}
            
            with torch.set_grad_enabled(train):
                out = self.forward(mot_batch=mot_batch, emo_batch=emo_batch) 

                # --- 1) Main Task Losses ---
                loss_mot, loss_v, loss_a, loss_cross = [torch.tensor(0., device=self.device) for _ in range(4)]

                if out.get('motion_logits') is not None:
                    mot_logits, mot_labels = out['motion_logits'], mot_batch['label_motion'].to(self.device)
                    
                    mot_logits_last_frame = mot_logits[:, -1, :]
                    mot_labels_last_frame = mot_labels[:, -1]
                    
                    # [핵심 수정] 마스크를 더 견고하게 만듭니다.
                    # 1. 모델의 출력에서 클래스 개수를 가져옵니다.
                    num_motion_classes = mot_logits_last_frame.shape[-1]
                    
                    # 2. 레이블이 유효한 범위(1부터 클래스 개수까지)에 있는지 확인합니다.
                    valid_mask = (mot_labels_last_frame > 0) & (mot_labels_last_frame <= num_motion_classes)
                    
                    if valid_mask.any():
                        # 유효한 레이블만 사용하여 손실을 계산합니다.
                        loss_mot = F.cross_entropy(mot_logits_last_frame[valid_mask], (mot_labels_last_frame[valid_mask] - 1).long())

                if out.get('valence_logits') is not None:
                    val_logits, raw_v = out['valence_logits'], emo_batch['valence_reg_emotion'].to(self.device).view(-1)
                    tgt_v = torch.full_like(raw_v, -100, dtype=torch.long); tgt_v[raw_v < 4]=0; tgt_v[(raw_v>=4)&(raw_v<7)]=1; tgt_v[raw_v>=7]=2
                    loss_v = F.cross_entropy(val_logits, tgt_v, ignore_index=-100)
                
                if out.get('arousal_logits') is not None:
                    aro_logits, raw_a = out['arousal_logits'], emo_batch['arousal_reg_emotion'].to(self.device).view(-1)
                    tgt_a = torch.full_like(raw_a, -100, dtype=torch.long); tgt_a[raw_a < 4]=0; tgt_a[(raw_a>=4)&(raw_a<7)]=1; tgt_a[raw_a>=7]=2
                    loss_a = F.cross_entropy(aro_logits, tgt_a, ignore_index=-100)
                
                if out.get('fused_motion') is not None and out.get('fused_emotion') is not None:
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

            if train:
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
        
        return {k: v / len(iterator) for k, v in epoch_losses.items()} if len(iterator) > 0 else epoch_losses
    
    @torch.no_grad()
    def evaluate_motion(self, loader):
        self.eval()
        preds_mot, trues_mot = [], []
        for mot_batch in tqdm(loader, desc="Evaluating Motion"):
            out = self.forward(mot_batch=mot_batch)
            if out['motion_logits'] is None: continue
            
            # Predictions for the whole sequence, shape: (B, T)
            p_mot_seq = out['motion_logits'].argmax(-1).cpu()
            # True labels for the whole sequence, shape: (B, T)
            t_mot_seq = mot_batch['label_motion'].cpu()
            
            # [핵심 수정] 예측(p_mot)과 정답(t_mot) 모두 마지막 프레임의 값만 선택합니다.
            p_mot_target = p_mot_seq[:, -1]               # Shape: (B,)
            t_mot_target = t_mot_seq[:, -1]               # Shape: (B,)
            
            # 유효한 정답 레이블에 대한 마스크 생성
            mask_mot = t_mot_target > 0
            
            # 마스크를 적용하여 유효한 1D 텐서들만 리스트에 추가
            if mask_mot.any():
                preds_mot.append(p_mot_target[mask_mot])
                trues_mot.append((t_mot_target[mask_mot] - 1)) # Labels are 1-based
        
        # [수정] 리스트가 비어있는 경우를 더 안전하게 처리
        if not trues_mot:
            return 0.0
            
        # 이제 두 텐서 모두 1D이므로 에러가 발생하지 않습니다.
        return accuracy_score(torch.cat(trues_mot), torch.cat(preds_mot))

    @torch.no_grad()
    def evaluate_emotion(self, loader):
        self.eval()
        
        # [FIX] Provide 4 empty lists to match the 4 variables
        preds_v, trues_v, preds_a, trues_a = [], [], [], []
        
        for emo_batch in tqdm(loader, desc="Evaluating Emotion"):
            out = self.forward(emo_batch=emo_batch)
            if out['valence_logits'] is None: continue
            raw_v = emo_batch['valence_reg_emotion'].reshape(-1).cpu()
            t_v = torch.full_like(raw_v, -100); t_v[raw_v < 4]=0; t_v[(raw_v >= 4)&(raw_v < 7)]=1; t_v[raw_v >= 7]=2
            m_v = t_v != -100
            preds_v.append(out['valence_logits'].argmax(-1).cpu()[m_v]); trues_v.append(t_v[m_v])
            
            raw_a = emo_batch['arousal_reg_emotion'].reshape(-1).cpu()
            t_a = torch.full_like(raw_a, -100); t_a[raw_a < 4]=0; t_a[(raw_a >= 4)&(raw_a < 7)]=1; t_a[raw_a >= 7]=2
            m_a = t_a != -100
            preds_a.append(out['arousal_logits'].argmax(-1).cpu()[m_a]); trues_a.append(t_a[m_a])
            
        acc_v = accuracy_score(torch.cat(trues_v), torch.cat(preds_v)) if trues_v and len(trues_v[0]) > 0 else 0.0
        acc_a = accuracy_score(torch.cat(trues_a), torch.cat(preds_a)) if trues_a and len(trues_a[0]) > 0 else 0.0
        return acc_v, acc_a
    
    def fusion_train(self):
        best_loss, patience_counter, best_performance = float('inf'), 0, {}

        for epoch in range(1, self.cfg.MainTask.epochs + 1):
            if self.cfg.MainTask.use_pmf and epoch >= self.cfg.MainTask.pmf_start_epoch:
                # PMF는 훈련 데이터셋 기준이므로 두 로더를 모두 사용해야 함
                # 간단하게 하기 위해 더 큰 로더를 기준으로 modal_freezing 수행
                loader_for_pmf = self.mot_train_loader if len(self.mot_train_loader) >= len(self.emo_train_loader) else self.emo_train_loader
                self.modal_freezing(loader_for_pmf)

            tr_losses = self.run_epoch(train=True)
            va_losses = self.run_epoch(train=False) # [추가] 검증 손실 계산
            
            va_acc_mot = self.evaluate_motion(self.mot_val_loader)
            va_acc_v, va_acc_a = self.evaluate_emotion(self.emo_val_loader)
            
            tr_loss_str = (f"Tr Loss: {tr_losses['total']:.3f} (M:{tr_losses['motion']:.2f}, V:{tr_losses['valence']:.2f}, A:{tr_losses['arousal']:.2f}, C:{tr_losses['cross']:.2f})")
            va_loss_str = f"Val Loss: {va_losses['total']:.3f}"
            print(f"Epoch {epoch:02d} | {tr_loss_str} | {va_loss_str} | Acc(M/V/A): {va_acc_mot:.3f}/{va_acc_v:.3f}/{va_acc_a:.3f}")

            current_val_loss = va_losses['total']
            if current_val_loss < best_loss:
                best_loss, patience_counter = current_val_loss, 0
                best_performance = {'mot_acc': va_acc_mot, 'val_acc': va_acc_v, 'aro_acc': va_acc_a, 'best_loss': best_loss}
                
                torch.save({'model_state_dict': self.state_dict(), 'optimizer_state_dict': self.optim.state_dict()}, "weights/best_fusion_v14.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.MainTask.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        print(f"\n--- Best Validation Performance ---")
        print(f"Loss: {best_performance.get('best_loss', 'N/A'):.4f} | Acc(M/V/A): {best_performance.get('mot_acc', 0):.3f}/{best_performance.get('val_acc', 0):.3f}/{best_performance.get('aro_acc', 0):.3f}")
        return best_performance