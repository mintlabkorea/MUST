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

from models.encoder.ppg_lstm_encoder import PPGEncoder
from models.encoder.imu_encoder import IMUFeatureEncoder
from models.encoder.veh_encoder import VehicleTCNEncoder
from models.encoder.sc_encoder import ScenarioEmbedding
from models.encoder.survey_encoder import PreSurveyEncoder

from data.code.pkl_dataloader import PKLMultiModalDataset
from data.code.collate_fn_mod import collate_fn_unified

from models.fusion.mot_predictor_mumu import MotionPredictor
from models.fusion.emo_predictor_mumu   import EmotionPredictor, EmotionTemporalClassifier
from models.fusion.fusion_block   import SMFusionBlock,GMFusionBlock

from utils.losses import compute_losses, masked_mse
from utils.visualization import visualize

from trainers.base_trainer import TrainerBase, dataProcessor

class TrainerMultiTask(TrainerBase, dataProcessor):
    def __init__(self, cfg):
        self.cfg = cfg
        self._prepare_data()
        self._debug_dataset() 

        self._build_models()
        self._move_all_to_device()
        self._create_optimizer()

        os.makedirs('weights', exist_ok=True)
        os.makedirs('results', exist_ok=True)


    def make_motion_loader(self, keys, shuffle):
        return DataLoader(
            PKLMultiModalDataset(
                participant_ids=keys,
                mode=self.cfg.mot_mode,
                **self.dataset_kwargs  # window_sec=window_sec_mot, window_stride=window_stride_mot
            ),
            batch_size=self.cfg.batch_size, shuffle=shuffle,
            num_workers=self.cfg.num_workers, pin_memory=True,
            collate_fn=lambda b: collate_fn_unified(
                b,
                veh_dim=len(self.veh_cols),
                seq_len=self.cfg.seq_len,
                win_samp=int(self.cfg.window_sec_mot*self.cfg.fs)
            )
        )

    def make_emotion_loader(self, keys, shuffle):
        # dataset_kwargs 복사 후 emo 전용 window_sec/stride 덮어쓰기
        kw = dict(self.dataset_kwargs)
        kw['window_sec']    = self.cfg.window_sec_emo
        kw['window_stride'] = self.cfg.window_stride_emo
        return DataLoader(
            PKLMultiModalDataset(
                participant_ids=keys,
                mode=self.cfg.emo_mode,
                **kw
            ),
            batch_size=self.cfg.batch_size, shuffle=shuffle,
            num_workers=self.cfg.num_workers, pin_memory=True,
            collate_fn=lambda b: collate_fn_unified(
                b,
                veh_dim=len(self.veh_cols),
                seq_len=self.cfg.seq_len,
                win_samp=int(self.cfg.window_sec_emo * self.cfg.fs)
            )
        )

    def _build_models(self):
        self.nets = nn.ModuleDict({
            'imu': IMUFeatureEncoder(self.cfg),
            'ppg': PPGEncoder(self.cfg),
            'veh': VehicleTCNEncoder(self.cfg),
            'sc' : ScenarioEmbedding(self.cfg),
            'survey' : PreSurveyEncoder(self.cfg)
        })

        # ─── 2) projection layers ────────────────────────────────────
        self.projs = nn.ModuleDict({
            'imu'   : nn.Linear(self.cfg.imu_params['encoder_dim'], self.cfg.hidden),
            'ppg'   : nn.Linear(self.cfg.ppg_params['embed_dim'],    self.cfg.hidden),
            'veh'   : nn.Linear(self.cfg.veh_params['embed_dim'],    self.cfg.hidden),
            'sc'    : nn.Linear(self.cfg.sc_params['embed_dim'],     self.cfg.hidden),
            'survey': nn.Linear(self.cfg.survey_dim,                 self.cfg.hidden)
        })

        self.modalities      = ['imu', 'ppg', 'veh', 'sc', 'survey']

        self.modality_logits = nn.Parameter(torch.zeros(len(self.modalities)), requires_grad=True)
        self.modality_bias   = nn.Parameter(torch.zeros(len(self.modalities)), requires_grad=True)
        self.beta            = nn.Parameter(torch.tensor(0.5),          requires_grad=True)

        # (self.nets, self.projs 는 _prepare_data에서 이미 정의되어 있다고 가정)
        self.sm_fuse          = SMFusionBlock(hidden_dim=self.cfg.hidden,
                                            modalities=self.modalities)
        self.gm_fuse          = GMFusionBlock(modalities = self.modalities,
                                              hidden_dim = self.cfg.hidden,
                                              num_heads = self.cfg.num_heads)
        self.bg_head          = nn.Linear(self.cfg.hidden, self.cfg.num_motion)
        self.eg_head          = nn.Linear(self.cfg.hidden, self.cfg.num_valence)
        self.motion_predictor = MotionPredictor(feature_dim=self.cfg.hidden,
                                                output_dim=self.cfg.num_motion)
        self.emotion_predictor= EmotionPredictor(feature_dim=self.cfg.hidden,
                                                hidden_dim=self.cfg.hidden,
                                                num_valence=self.cfg.num_valence,
                                                num_arousal=self.cfg.num_arousal)
        
        device = self.cfg.device
        # ─── 4) scaler & loss 정의 ─────────────────────────────────────
        self.scaler = GradScaler()
        self.ce_v   = nn.CrossEntropyLoss(
                        weight=torch.tensor(self.cfg.valence_weights,
                                            device=device),
                        ignore_index=self.cfg.ign_emo)
        self.ce_a   = nn.CrossEntropyLoss(
                        weight=torch.tensor(self.cfg.arousal_weights,
                                            device=device),
                        ignore_index=self.cfg.ign_emo)
        self.ce_m   = nn.CrossEntropyLoss(
                        weight=torch.tensor(self.cfg.motion_weights,
                                            device=device),
                        ignore_index=self.cfg.ign_mot)
    
        
    def _create_optimizer(self):
        # params 모으고 옵티마이저 생성
        import itertools
        net_p  = itertools.chain(*(m.parameters() for m in self.nets.values()))
        proj_p = itertools.chain(*(p.parameters() for p in self.projs.values()))
        self.optim = torch.optim.Adam(
            list(net_p) + list(proj_p)
            + list(self.motion_predictor.parameters())
            + list(self.emotion_predictor.parameters())
            + list(self.bg_head.parameters())
            + list(self.eg_head.parameters())
            + [self.modality_logits, self.modality_bias, self.beta],
            lr=self.cfg.lr
        )

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
        self.beta.to(device)
        # 2) 만약 dict 형식으로 관리하는 모듈(nets, projs)도
        for d in ('nets', 'projs'):
            if hasattr(self, d):
                container = getattr(self, d)
                for k, module in container.items():
                    if isinstance(module, nn.Module):
                        container[k] = module.to(device)
        

    def forward_simple(self, batch):
        device = self.cfg.device
        # 은닉 차원
        H = self.cfg.hidden
        # IMU 또는 imu_emotion 중 있는 쪽을 안전하게 가져옴
        if 'imu_motion' in batch:
            imu_tensor = batch.get('imu_motion')
        else:
            imu_tensor = batch.get('imu_emotion')
        B = imu_tensor.size(0)

        # 1) IMU (always present)
        imu = imu_tensor.to(device)
        imu_len = (imu.abs().sum(-1) > 0).sum(1)
        imu_out    = self.nets['imu'](imu, imu_len)     # → (B, T, D_imu)
        imu_pooled = imu_out.mean(dim=1)                # → (B, D_imu)
        imu_emb    = self.projs['imu'](imu_pooled)      # → (B, H)

        # 2) PPG
        if 'ppg_motion' in batch or 'ppg_emotion' in batch:
            if 'ppg_motion' in batch:
                ppg    = batch.get('ppg_motion').to(device)
                rr     = batch.get('ppg_rr_motion').to(device)
                rmssd  = batch.get('ppg_rmssd_motion').to(device)
                sdnn   = batch.get('ppg_sdnn_motion').to(device)
            else:
                ppg    = batch.get('ppg_emotion').to(device)
                rr     = batch.get('ppg_rr_emotion').to(device)
                rmssd  = batch.get('ppg_rmssd_emotion').to(device)
                sdnn   = batch.get('ppg_sdnn_emotion').to(device)
            # rr     = rr.to(device)
            # rmssd  = rmssd.to(device)
            # sdnn   = sdnn.to(device)
            ppg_emb = self.projs['ppg'](
                self.nets['ppg'](ppg, rr, rmssd, sdnn)
            )
        else:
            ppg_emb = torch.zeros(B, H, device=device)

        # 3) VEHICLE (should always be there)
        # 모션 모드에선 'veh', 감정 모드에선 'veh_emotion' 키 사용
        if 'veh_motion' in batch:
            veh_tensor = batch.get('veh_motion')
            if veh_tensor is not None:
                veh = veh_tensor.to(device)
                veh_mask = batch.get('veh_mask_motion').to(device).bool()
            else:
                veh_emb = torch.zeros(B, H, device=device)   
        else:
            veh_tensor = batch.get('veh_emotion')
            if veh_tensor is not None:
                veh = veh_tensor.to(device)
                veh_mask = batch.get('veh_mask_emotion').to(device).bool()
            else:
                veh_emb = torch.zeros(B, H, device = device)

        veh_emb  = self.projs['veh'](
            self.nets['veh'](veh.transpose(1,2), None,
                            return_pooled=True, mask=veh_mask)
            )


        # 4) SCENARIO
        if 'scenario_evt_m' in batch or 'scenario_evt_e' in batch:
            evt = batch.get('scenario_evt_m', batch.get('scenario_evt_e')).to(device)
            typ = batch.get('scenario_type_m', batch.get('scenario_type_e')).to(device)
            ph  = batch.get('phase_evt_m', batch.get('phase_evt_e')).to(device)
            tm  = batch.get('scenario_time_m', batch.get('scenario_time_e')).to(device)
            sc_emb = self.projs['sc'](self.nets['sc'](evt, typ, ph, tm))
        else:
            sc_emb = torch.zeros(B, H, device=device)

        # 5) SURVEY
        if 'survey_m' in batch or 'survey_e' in batch:
            surv = batch.get('survey_m', batch.get('survey_e')).to(device)
            surv_emb = self.projs['survey'](surv)
        else:
            surv_emb = torch.zeros(B, self.cfg.hidden, device=device)

        # 6) fuse & heads
        # 1) collect raw embeddings
        raw_embs = {
            'imu':      imu_emb,
            'ppg':      ppg_emb,
            'veh':      veh_emb,
            'sc':       sc_emb,
            'survey':   surv_emb
        }

        # 2) compute bias-adjusted logits
        #    αₘ = softmax(modality_logits + modality_bias)
        g = self.modality_logits + self.modality_bias    # shape (5,)
        alpha = F.softmax(g, dim=0)                      # normalized weights

        # 3) modality gating
        gated_embs = {
            m: raw_embs[m] * alpha[i]
            for i, m in enumerate(self.modalities)
        }

        # 4) simple gated sum
        fused = sum(gated_embs.values())

        # Simple predictors 사용 (fusion + MLP) 대신 Predictor 클래스 활용
        # Motion
        feat_m = {
            'imu':      fused.unsqueeze(1),
            'veh':      fused.unsqueeze(1),
            'sc':       fused.unsqueeze(1),
            'survey':   fused.unsqueeze(1),
        }
        mot_repr, mot_logits = self.motion_predictor(feat_m, return_feature=True)
        mot_repr = mot_repr.unsqueeze(1).detach()  # (B,1,feature_dim)
        

        # Emotion
        feat_e = {
            'ppg':      fused.unsqueeze(1),
            'sc':       fused.unsqueeze(1),
            'survey':   fused.unsqueeze(1),
        }
        emo_dict = self.emotion_predictor(feat_e, mot_repr)
        val_logits = emo_dict['valence_logits']
        aro_logits = emo_dict['arousal_logits']

        return {
            'motion_logits':  mot_logits,
            'valence_logits': val_logits,
            'arousal_logits': aro_logits,
        }
    
    def forward_batch(self, batch):
        device = self.cfg.device

        B = next(iter(batch.values())).shape[0]
        H = self.cfg.hidden

        # 1) 각 모달리티 인코딩 → projection
        imu_tensor = batch.get('imu_motion')
        imu      = imu_tensor.to(device)
        imu_len  = (imu.abs().sum(-1) > 0).sum(1)
        imu_emb  = self.projs['imu'](
                    self.nets['imu'](imu, imu_len).mean(1)
                )  # (B, H)

        if 'ppg_motion' in batch:
            ppg     = batch['ppg_motion'].to(device)
            rr      = batch.get('ppg_rr_motion', torch.zeros(B, device=device))
            rmssd   = batch.get('ppg_rmssd_motion', torch.zeros(B, device=device))
            sdnn    = batch.get('ppg_sdnn_motion', torch.zeros(B, device=device))
            ppg_emb = self.projs['ppg'](
                        self.nets['ppg'](ppg, rr.to(device), rmssd.to(device), sdnn.to(device))
                    )

        else:
            # print(list(batch.keys()))
            print('no ppg in batch')
            ppg_emb = torch.zeros(B, H, device=device)

        # if 'veh' in batch or 'veh_motion' in batch:
        if 'veh_motion' in batch:
            veh      = batch['veh_motion'].to(device)
            veh_mask = batch['veh_mask_motion'].to(device).bool()
            veh_emb  = self.projs['veh'](
                        self.nets['veh'](veh.transpose(1,2), None,
                                            return_pooled=True,
                                            mask=veh_mask)
                    )

        else:
            # print('no veh in batch')
            veh_emb = torch.zeros(B, H, device=device)

        if 'scenario_evt_m' in batch:
            evt = batch['scenario_evt_m'].to(device)
            typ = batch['scenario_type_m'].to(device)
            ph  = batch['phase_evt_m'].to(device)
            tm  = batch['scenario_time_m'].to(device)
            sc_emb = self.projs['sc'](
                        self.nets['sc'](evt, typ, ph, tm)
                    )
        else:
            sc_emb = torch.zeros(B, H, device=device)

        if 'survey_m' in batch:
            survey   = batch['survey_m'].to(device)
            surv_emb = self.projs['survey'](survey)
        else:
            surv_emb = torch.zeros(B, H, device=device)

        raw_embs = {
            'imu':      imu_emb,
            'ppg':      ppg_emb,
            'veh':      veh_emb,
            'sc':       sc_emb,
            'survey':   surv_emb
        }
        
        g     = self.modality_logits + self.modality_bias       # (M,)
        alpha = F.softmax(g, dim=0)                             # (M,)

        gated_embs = {
            m: raw_embs[m] * alpha[i]
            for i,m in enumerate(self.modalities)
        }

        S = torch.stack([gated_embs[m] for m in self.modalities], dim=0).sum(0)
        # 3) Cross-Attention 입력으로 gated_embs 를 줌
        feat_m = {
            m: gated_embs[m].unsqueeze(1)
            for m in ['imu','ppg', 'veh','sc','survey']
        }

        fused_repr = self.beta*self.motion_predictor.fusion(feat_m).squeeze(1) + (1. - self.beta) * S
        mot_logits = self.motion_predictor.head(fused_repr)  # (B, num_motion)

        return {
            'motion_logits':  mot_logits,
        }

    def forward_emotion(self, batch):
        device = self.cfg.device

        imu_e       = batch['imu_emotion'].to(device)
        imu_e_lens  = batch['imu_e_lens'].to(device)
        ppg_e       = batch['ppg_emotion'].to(device)
        ppg_rr      = batch['ppg_rr_emotion'].to(device)
        ppg_rmssd   = batch['ppg_rmssd_emotion'].to(device)
        ppg_sdnn    = batch['ppg_sdnn_emotion'].to(device)
        veh_e       = batch['veh_emotion'].to(device)
        veh_mask    = batch['veh_mask_emotion'].to(device).bool()
        sc_evt_e    = batch['scenario_evt_e'].to(device)
        phase_evt_e = batch['phase_evt_e'].to(device)
        sc_type_e   = batch['scenario_type_e'].to(device)
        sc_time_e   = batch['scenario_time_e'].to(device)
        survey_e    = batch['survey_e'].to(device)

        # imu encoder
        imu_out   = self.nets['imu'](imu_e, imu_e_lens)      # (B_e, T, D)
        imu_pooled= imu_out.mean(dim=1)                     # (B_e, D)
        imu_emb   = self.projs['imu'](imu_pooled)           # (B_e, H)

        # ppg encoder
        ppg_out   = self.nets['ppg'](ppg_e, ppg_rr, ppg_rmssd, ppg_sdnn)  # (B_e, D)
        ppg_emb   = self.projs['ppg'](ppg_out)                           # (B_e, H)

        # veh encoder
        veh_emb_raw = self.nets['veh'](
            veh_e.transpose(1,2), None, return_pooled=True, mask=veh_mask
        )  # (B_e, D)
        veh_emb     = self.projs['veh'](veh_emb_raw)  # (B_e, H)

        # sc encoder
        sc_emb_raw = self.nets['sc'](sc_evt_e, sc_type_e, phase_evt_e, sc_time_e)  # (B_e, D)
        sc_emb     = self.projs['sc'](sc_emb_raw)                                    # (B_e, H)
        surv_emb = self.projs['survey'](survey_e)

        raw_embs = {
            'imu':      imu_emb,
            'ppg':      ppg_emb,
            'veh':      veh_emb,
            'sc':       sc_emb,
            'survey':   surv_emb
        }
        
        g     = self.modality_logits + self.modality_bias       # (M,)
        alpha = F.softmax(g, dim=0)                             # (M,)

        gated_embs = {
            m: raw_embs[m] * alpha[i]
            for i,m in enumerate(self.modalities)
        }

        S = sum(gated_embs.values())  # (B,H)

        emo_feat = {
            'ppg':    ppg_emb.unsqueeze(1),
            'sc':     sc_emb.unsqueeze(1),
            'survey': surv_emb.unsqueeze(1),
        }
        X_aux = self.sm_fuse(raw_embs)
        cross_e = self.gm_fuse(X_aux, raw_embs)

        fused_repr = self.beta * cross_e + (1 - self.beta) * S 

        mot_repr = self.motion_predictor(
            {m: gated_embs[m].unsqueeze(1) for m in ['imu','veh','sc','survey']},
            return_feature=True)[0].unsqueeze(1)
        
        emo_out = self.emotion_predictor(
            emo_feat,
            motion_repr=mot_repr,
            guided_feature=fused_repr.unsqueeze(1)
        )

        val_logits = emo_out['valence_logits']
        aro_logits = emo_out['arousal_logits']

        return { 'valence_logits': val_logits,
                 'arousal_logits': aro_logits,
                 'eg_logits':      self.eg_head(fused_repr)
                }
    

    def run_motion_epoch(self, loader, train: bool):
        """Motion 전용 epoch 루프"""
        if train:
            self.motion_predictor.train()
            for m in [*self.nets.values(), *self.projs.values()]:
                m.train()
            iterator = tqdm(loader, desc="Train Motion")
        else:
            self.motion_predictor.eval()
            for m in [*self.nets.values(), *self.projs.values()]:
                m.eval()
            iterator = loader

        total_loss, count = 0.0, 0
        for batch in iterator:
            # 1) 필터링 (모든 라벨이 ignore인 윈도우 건너뛰기)
            raw_mot_seq = batch['labels']['label_motion']  # (B, T)
            # 각 윈도우의 첫 프레임 레이블을 대표값으로 사용
            raw_mot   = raw_mot_seq[:, 0].long()          # (B,)
            mask_win  = raw_mot != self.cfg.ign_mot        # (B,)

            if not mask_win.any():
                 continue
            
            # batch-level 필터링: B차원(Tensor)만 마스킹
            B = mask_win.shape[0]
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor) and v.shape[0] == B:
                    batch[k] = v[mask_win]

            # labels 딕셔너리도 별도 처리
            for k, v in list(batch['labels'].items()):
                if isinstance(v, torch.Tensor) and v.shape[0] == B:
                    batch['labels'][k] = v[mask_win]

            # 2) Forward + Loss via forward_batch
            out = self.forward_batch(batch)
            mot_logits = out['motion_logits']           # shape (B, num_motion)
            # 대표 레이블(0-based)
            tgt = (raw_mot - 1).clamp(0, self.cfg.num_motion - 1).to(self.cfg.device)
            valid = raw_mot != self.cfg.ign_mot
            if valid.any():
                loss_m = self.ce_m(mot_logits[valid], tgt[valid])
            else:
                loss_m = torch.tensor(0., device=self.cfg.device)
                
            # 3) Backward/Step
            if train:
                self.optim.zero_grad()
                self.scaler.scale(loss_m).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(
                        *[m.parameters() for m in [*self.nets.values(),
                                                   *self.projs.values(),
                                                   self.motion_predictor]],
                    [self.modality_bias, self.modality_logits, self.beta]
                    ), max_norm=1.0
                )
                self.scaler.step(self.optim)
                self.scaler.update()

            total_loss += loss_m.item()
            count += 1

        return total_loss / max(count,1)

    def run_emotion_epoch(self, loader, train: bool):
        """Emotion 전용 epoch 루프"""
        if train:
            self.emotion_predictor.train()
            for m in [*self.nets.values(), *self.projs.values()]:
                m.train()
            iterator = tqdm(loader, desc="Train Emotion")
        else:
            self.emotion_predictor.eval()
            for m in [*self.nets.values(), *self.projs.values()]:
                m.eval()
            iterator = loader

        total_loss, count = 0.0, 0
        for batch in iterator:
            # 감정 전용 키 사용
            raw_v = batch['valence_reg_emotion'].reshape(-1)
            raw_a = batch['arousal_reg_emotion'].reshape(-1)
            mask  = (raw_v != self.cfg.ign_emo) & (raw_a != self.cfg.ign_emo)
            if mask.sum() == 0:
                continue

            # 필터링: batch-level mask
            def _f(x):
                return x[mask] if isinstance(x, torch.Tensor) and x.shape[0] == mask.shape[0] else x
            for k in list(batch.keys()):
                batch[k] = _f(batch[k])

            # Predictor를 통한 감정 예측
            # forward_batch 혹은 forward_emotion 중 선택
            out = self.forward_emotion(batch)
            # binning & loss 계산
            raw_v = batch['valence_reg_emotion'].reshape(-1)
            raw_a = batch['arousal_reg_emotion'].reshape(-1)
            tgt_v = torch.full_like(raw_v, self.cfg.ign_emo, dtype=torch.long)
            tgt_a = torch.full_like(raw_a, self.cfg.ign_emo, dtype=torch.long)
            tgt_v[raw_v < 4] = 0
            tgt_v[(raw_v >= 4) & (raw_v < 7)] = 1
            tgt_v[raw_v >= 7] = 2
            tgt_a[raw_a < 4] = 0
            tgt_a[(raw_a >= 4) & (raw_a < 7)] = 1
            tgt_a[raw_a >= 7] = 2

            mask_v = tgt_v != self.cfg.ign_emo
            mask_a = tgt_a != self.cfg.ign_emo

            # Predictor 출력 사용
            val_logits = out['valence_logits']
            aro_logits = out['arousal_logits']
            l_v = (self.ce_v(val_logits[mask_v], tgt_v[mask_v].to(self.cfg.device))
                   if mask_v.any() else torch.tensor(0., device=self.cfg.device))
            l_a = (self.ce_a(aro_logits[mask_a], tgt_a[mask_a].to(self.cfg.device))
                   if mask_a.any() else torch.tensor(0., device=self.cfg.device))
            loss = l_v + l_a

            if train:
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                # 감정 predictor 포함하여 clip
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(
                        *[m.parameters() for m in [*self.nets.values(),
                                                   *self.projs.values(),
                                                   self.emotion_predictor]]
                    ), max_norm=1.0
                )
                self.scaler.step(self.optim)
                self.scaler.update()

            total_loss += loss.item()
            count += 1

        return total_loss / max(count,1)
        
    def evaluate_motion(self, loader):
        """Motion 전용 평가: accuracy 계산"""
        self.motion_predictor.eval()
        for m in [*self.nets.values(), *self.projs.values()]:
            m.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for batch in loader:
                out = self.forward_batch(batch)
                # 1) 윈도우별 예측 (batch size B)
                preds = out['motion_logits'].argmax(-1).cpu().numpy()       # shape: (B,)
                # 2) 윈도우별 대표 레이블 (첫 타임스텝)
                raw_m = batch['labels']['label_motion'][:,0].cpu().numpy()    # shape: (B,)
                trues = raw_m.astype(int) - 1                                 # 0-based
                mask  = trues >= 0                                            # valid windows
                all_preds.append(preds[mask])
                all_trues.append(trues[mask])
        if all_trues:
            mot_p = np.concatenate(all_preds)
            mot_t = np.concatenate(all_trues)
            return {'mot_acc': accuracy_score(mot_t, mot_p)}
        else:
            return {'mot_acc': 0.0}

    def evaluate_emotion(self, loader):
        """Emotion 전용 평가: valence/arousal accuracy 계산"""
        self.emotion_predictor.eval()
        for m in [*self.nets.values(), *self.projs.values()]:
            m.eval()
        val_preds, val_trues = [], []
        aro_preds, aro_trues = [], []
        with torch.no_grad():
            for batch in loader:
                out = self.forward_emotion(batch)
                vp = out['valence_logits'].argmax(-1).cpu().numpy()
                ap = out['arousal_logits'].argmax(-1).cpu().numpy()
                raw_v = batch['valence_reg_emotion'].reshape(-1).cpu().numpy()
                raw_a = batch['arousal_reg_emotion'].reshape(-1).cpu().numpy()
                vt = np.full_like(raw_v, self.cfg.ign_emo, dtype=int)
                vt[raw_v<4] = 0; vt[(raw_v>=4)&(raw_v<7)] = 1; vt[raw_v>=7] = 2
                at = np.full_like(raw_a, self.cfg.ign_emo, dtype=int)
                at[raw_a<4] = 0; at[(raw_a>=4)&(raw_a<7)] = 1; at[raw_a>=7] = 2
                mask = vt != self.cfg.ign_emo
                val_preds.append(vp[mask]); val_trues.append(vt[mask])
                aro_preds.append(ap[mask]); aro_trues.append(at[mask])
        metrics = {}
        if val_trues:
            vp = np.concatenate(val_preds); vt = np.concatenate(val_trues)
            ap = np.concatenate(aro_preds); at = np.concatenate(aro_trues)
            metrics['val_acc'] = accuracy_score(vt, vp)
            metrics['aro_acc'] = accuracy_score(at, ap)
        else:
            metrics['val_acc'] = 0.0
            metrics['aro_acc'] = 0.0
        return metrics

    def evaluate_both(self, split_name: str):
        # select keys
        keys = getattr(self, f"{split_name}_keys")
        # 1) MOTION
        mot_loader = self.make_motion_loader(keys, False)
        all_t, simple_preds, cross_preds = [], [], []

        with torch.no_grad():
            for batch in mot_loader:
                # ground truth
                # 1) 윈도우 단위 레이블 하나로 압축 (여기선 각 윈도우의 첫 프레임 레이블 사용)
                raw_m = batch['labels']['label_motion']            # (B, T)
                # 예시: 첫 타임스텝 레이블로 대표
                mt = (raw_m[:, 0].cpu().numpy() - 1).astype(int)   # (B,)
                mask = mt >= 0
                all_t.append(mt[mask])

                # 2) simple vs cross 예측 (각 윈도우당 하나)
                out_s = self.forward_simple(batch)
                out_c = self.forward_batch(batch)
                ps = out_s['motion_logits'].argmax(-1).cpu().numpy()  # (B,)
                pc = out_c['motion_logits'].argmax(-1).cpu().numpy()  # (B,)
                simple_preds.append(ps[mask])
                cross_preds.append(pc[mask])

        mot_t = np.concatenate(all_t)
        mot_s = np.concatenate(simple_preds)
        mot_c = np.concatenate(cross_preds)

        print(f"[{split_name}] MOTION  SIMPLE={accuracy_score(mot_t,mot_s):.4f}  CROSS={accuracy_score(mot_t,mot_c):.4f}")
        # visualize expects {'mot_p','mot_t'}
        visualize(self.cfg, {'mot_p':mot_s,'mot_t':mot_t}, f"{split_name}_motion_simple")
        visualize(self.cfg, {'mot_p':mot_c,'mot_t':mot_t}, f"{split_name}_motion_cross")

        # 2) EMOTION (valence & arousal)
        emo_loader = self.make_emotion_loader(
            getattr(self, split_name + '_keys'), False
        )
        val_t_buf, aro_t_buf = [], []
        val_s_buf, aro_s_buf = [], []
        val_c_buf, aro_c_buf = [], []

        with torch.no_grad():
            for batch in emo_loader:
                # ground truth bins
                rv = batch['valence_reg_emotion'].reshape(-1).cpu().numpy()
                ra = batch['arousal_reg_emotion'].reshape(-1).cpu().numpy()
                vt = np.full_like(rv, self.cfg.ign_emo, int)
                at = np.full_like(ra, self.cfg.ign_emo, int)
                vt[rv<4] = 0; vt[(rv>=4)&(rv<7)] = 1; vt[rv>=7] = 2
                at[ra<4] = 0; at[(ra>=4)&(ra<7)] = 1; at[ra>=7] = 2
                mask = vt != self.cfg.ign_emo

                val_t_buf.append(vt[mask])
                aro_t_buf.append(at[mask])

                # simple vs cross
                out_s = self.forward_simple(batch)
                out_c = self.forward_emotion(batch)

                vp_s = out_s['valence_logits'].argmax(-1).cpu().numpy()
                ap_s = out_s['arousal_logits'].argmax(-1).cpu().numpy()
                vp_c = out_c['valence_logits'].argmax(-1).cpu().numpy()
                ap_c = out_c['arousal_logits'].argmax(-1).cpu().numpy()

                val_s_buf.append(vp_s[mask])
                aro_s_buf.append(ap_s[mask])
                val_c_buf.append(vp_c[mask])
                aro_c_buf.append(ap_c[mask])

        val_t = np.concatenate(val_t_buf)
        val_s = np.concatenate(val_s_buf)
        val_c = np.concatenate(val_c_buf)
        aro_t = np.concatenate(aro_t_buf)
        aro_s = np.concatenate(aro_s_buf)
        aro_c = np.concatenate(aro_c_buf)

        print(f"[{split_name}] VALENCE  SIMPLE={accuracy_score(val_t,val_s):.4f}  CROSS={accuracy_score(val_t,val_c):.4f}")
        print(f"[{split_name}] AROUSAL  SIMPLE={accuracy_score(aro_t,aro_s):.4f}  CROSS={accuracy_score(aro_t,aro_c):.4f}")

        # visualize expects {'val_p','val_t'} and {'aro_p','aro_t'}
        visualize(self.cfg, {'val_p':val_s,'val_t':val_t}, f"{split_name}_valence_simple")
        visualize(self.cfg, {'val_p':val_c,'val_t':val_t}, f"{split_name}_valence_cross")
        visualize(self.cfg, {'aro_p':aro_s,'aro_t':aro_t}, f"{split_name}_arousal_simple")
        visualize(self.cfg, {'aro_p':aro_c,'aro_t':aro_t}, f"{split_name}_arousal_cross")


    def _get_state(self):
        """현재 모델·옵티마이저 상태를 dict 형태로 반환합니다."""
        state = {
            'nets':               {k: m.state_dict() for k, m in self.nets.items()},
            'projs':              {k: m.state_dict() for k, m in self.projs.items()},
            # predictors
            'motion_predictor':   self.motion_predictor.state_dict(),
            'emotion_predictor':  self.emotion_predictor.state_dict(),
            'bg_head':            self.bg_head.state_dict(),
            'eg_head':            self.eg_head.state_dict(),
            # 옵티마이저, 스케일러
            'optim':              self.optim.state_dict(),
            'scaler':             self.scaler.state_dict(),
            'epoch':              getattr(self, 'current_epoch', None),
        }
        return state


    def train(self, build_model: bool = True):
        mot_tr = self.make_motion_loader(self.train_keys, True)
        mot_va = self.make_motion_loader(self.val_keys,   False)
        # emo_tr = self.make_emotion_loader(self.train_keys, True)
        # emo_va = self.make_emotion_loader(self.val_keys,   False)

        best_metric = float('inf')
        epochs_no_improve = 0

        for ep in range(1, self.cfg.epochs+1):
            # 1) Motion 학습/검증
            mot_train_loss = self.run_motion_epoch(mot_tr, True)
            mot_val_loss   = self.run_motion_epoch(mot_va, False)
            mot_val_acc    = self.evaluate_motion(mot_va)['mot_acc']

            # 2) Emotion 학습/검증
            # emo_train_loss = self.run_emotion_epoch(emo_tr, True)
            # emo_val_loss   = self.run_emotion_epoch(emo_va, False)
            # emo_metrics    = self.evaluate_emotion(emo_va)  # {'val_acc', 'aro_acc'}

            # 3) 로그 출력
            print(f"Epoch{ep:02d} | "
                f"MOT TrainL={mot_train_loss:.3f} ValL={mot_val_loss:.3f} ValAcc={mot_val_acc:.3f} | "
                # f"EMO TrainL={emo_train_loss:.3f} ValL={emo_val_loss:.3f} "
                # f"ValAcc={emo_metrics['val_acc']:.3f} AroAcc={emo_metrics['aro_acc']:.3f}")
            )
            print(torch.sigmoid(self.emotion_predictor.w_param).item())
            # 4) α, β 값도 찍어두면 좋습니다
            with torch.no_grad():
                g = self.modality_logits + self.modality_bias
                alphas = torch.softmax(g, dim=0)
                print(f"→ α={alphas.cpu().tolist()}, β={self.beta.item():.3f}")

            # 5) 복합 검증 지표
            combined_val = mot_val_loss
            if combined_val < best_metric:
                best_metric = combined_val
                epochs_no_improve = 0
                torch.save(self._get_state(), 'weights/best_multitask.pt')
                print(" → New best checkpoint!")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.cfg.patience:
                    print(f"Early stopping at epoch {ep} "
                        f"(no improve for {self.cfg.patience} epochs)")
                    break

        # --- 최종 평가: 가장 좋은 체크포인트 로드 후 full evaluation ---
        self._load_state('weights/best_multitask.pt')
        print(">>> VALIDATION RESULTS")
        self.evaluate_both('val')
        print(">>> TEST RESULTS")
        self.evaluate_both('test')