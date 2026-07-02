# loader.py

import numpy as np
from torch.utils.data import DataLoader

# 의존성 파일 임포트
from data.code.pkl_dataloader import PKLMultiModalDataset
from data.code.collate_fn_mod import collate_fn_unified
from trainers.base_trainer import dataProcessor

def make_motion_loader(cfg, pid_keys, shuffle: bool, dp: dataProcessor, include_indices=False):
    """모션 인식(Motion) 태스크 전용 데이터 로더"""
    
    ds_kwargs = dp.dataset_kwargs.copy()
    ds_kwargs.update({
        'veh_cols':      dp.veh_cols,
         'mode':          cfg.Data.mot_mode,
         'window_sec':    cfg.Data.window_sec_mot,
         'window_stride': cfg.Data.window_stride_mot,})
    
    # [수정] Dataset 객체를 먼저 생성합니다.
    dataset = PKLMultiModalDataset(participant_ids=pid_keys, **ds_kwargs)

    # [수정] 플래그에 따라 Dataset 내부 속성을 설정합니다.
    if include_indices:
        dataset.include_indices = True
    
    return DataLoader(
        dataset, # 수정된 dataset 객체를 사용
        batch_size=cfg.Data.batch_size, shuffle=shuffle, num_workers=cfg.Data.num_workers,
        collate_fn=lambda b: collate_fn_unified(b,
                                                veh_dim=len(dp.veh_cols),
                                                seq_len=cfg.Data.seq_len,
                                                win_samp=int(cfg.Data.window_sec_mot * cfg.Data.fs)
                                            ))

# [수정] include_indices=False 인자를 추가합니다.
def make_emotion_loader(cfg, pid_keys, shuffle: bool, dp: dataProcessor, include_indices=False):
    """감정 인식(Emotion) 태스크 전용 데이터 로더"""

    ds_kwargs = dp.dataset_kwargs.copy()
    ds_kwargs.update({
        'veh_cols':      dp.veh_cols,
        'mode':          cfg.Data.emo_mode,
        'window_sec':    cfg.Data.window_sec_emo,
        'window_stride': cfg.Data.window_stride_emo,
    })

    # [수정] Dataset 객체를 먼저 생성합니다.
    dataset = PKLMultiModalDataset(participant_ids=pid_keys, **ds_kwargs)

    # [수정] 플래그에 따라 Dataset 내부 속성을 설정합니다.
    if include_indices:
        dataset.include_indices = True

    return DataLoader(
        dataset, # 수정된 dataset 객체를 사용
        batch_size=cfg.Data.batch_size, shuffle=shuffle, num_workers=cfg.Data.num_workers, 
        collate_fn=lambda b: collate_fn_unified(b, veh_dim=len(dp.veh_cols), seq_len=cfg.Data.seq_len, win_samp=int(cfg.Data.window_sec_emo * cfg.Data.fs))
        )


def make_multitask_loader(cfg, pid_keys, shuffle: bool, dp: dataProcessor, include_indices=False): # fusion용 데이터 배치 설정
    """
    모션과 감정 인식을 동시에 학습하기 위한 통합 데이터 로더.
    Cross-modal fusion 단계에서 사용합니다.
    """

    ds_kwargs = dp.dataset_kwargs.copy()
    ds_kwargs.update({
        'veh_cols':      dp.veh_cols,
        'mode':          cfg.Data.emo_mode,
        'window_sec':    cfg.Data.window_sec_emo,
        'window_stride': cfg.Data.window_stride_emo,
    })

    #  Dataset 
    dataset = PKLMultiModalDataset(participant_ids=pid_keys, **ds_kwargs)

    # include_indices 플래그에 따라 Dataset 내부 속성 설정
    if include_indices:
        dataset.include_indices = True

    # 3. 속성이 설정된 dataset으로 DataLoader 생성
    return DataLoader(
        dataset, # 수정된 dataset 객체를 사용
        batch_size=cfg.Data.batch_size, 
        shuffle=shuffle, 
        num_workers=cfg.Data.num_workers,
        collate_fn=lambda b: collate_fn_unified(b, veh_dim=len(dp.veh_cols), seq_len=cfg.Data.seq_len, win_samp=int(cfg.Data.window_sec_emo * cfg.Data.fs))
    )
