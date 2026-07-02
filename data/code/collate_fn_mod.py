import torch
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

def collate_fn_unified(batch, **kwargs):

    # 이 부분에서 batch의 모든 키(indices 포함)를 리스트로 묶어 collated 딕셔너리에 저장합니다.
    keys = batch[0].keys()
    collated = {key: [d[key] for d in batch] for key in keys}

    # 이 반복문은 'indices'를 제외한 다른 키들을 텐서로 변환합니다.
    for key, values in collated.items():
        if key in ('imu', 'ppg', 'veh'):
            # pad_sequence를 사용하려면 values가 텐서 리스트여야 합니다.
            # Dataset에서 이미 텐서로 반환했다면 이 코드는 유효합니다.
            padded = pad_sequence(values, batch_first=True, padding_value=0.0)
            collated[key] = padded
            
        elif key == 'ppg_rr':
            values_no_none = [v for v in values if v is not None and v.nelement() > 0]
            if not values_no_none:
                # 배치 내 모든 ppg_rr이 비어있을 경우, 기본 shape을 가진 텐서 생성
                # 예시: (배치 크기, 1)
                padded = torch.full((len(batch), 1), -100.0)
            else:
                padded = pad_sequence(values_no_none, batch_first=True, padding_value=-100.0)
            collated[key] = padded

        elif key in ('ppg_rmssd', 'ppg_sdnn', 'scenario_time', 'scenario_evt', 'phase_evt', 'scenario_type', 'static'):
            if key in ('scenario_evt', 'scenario_type', 'phase_evt'):
                # NaN 값은 0으로 대체하고 정수형으로 변환
                # 데이터의 실제 최대값에 따라 클리핑 로직을 추가할 수 있습니다.
                valid_values = [v if pd.notna(v) else 0 for v in values]
                # 여기서는 .long() 대신 .to(torch.long)를 사용
                collated[key] = torch.tensor(valid_values, dtype=torch.long)
            else:
                collated[key] = torch.stack(values)
                    
        elif key == 'labels':
            label_batch = {}
            # labels 딕셔너리가 비어있지 않은 첫 번째 샘플을 기준으로 키를 가져옵니다.
            valid_labels = [d for d in values if d]
            if not valid_labels:
                collated[key] = {} # 모든 샘플에 label이 없으면 빈 딕셔너리 반환
                continue
                
            sub_keys = valid_labels[0].keys()
            for sub_key in sub_keys:
                sub_values = [d.get(sub_key) for d in values]
                
                if sub_key == 'label_motion':
                    # numpy 배열을 텐서로 변환
                    sub_values_tensor = [torch.from_numpy(v) for v in sub_values if v is not None]
                    if not sub_values_tensor: continue
                    padded = pad_sequence(sub_values_tensor, batch_first=True, padding_value=0)
                    label_batch[sub_key] = padded.long()

                if 'label_tot' in batch[0]:
                    label_batch['label_tot'] = torch.stack([s['label_tot'] for s in batch])
                
                if sub_key == 'label_act':
                    padded_act = pad_sequence(
                        [(torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for v in sub_values if v is not None],

                        batch_first=True,
                        padding_value=-100.0
                    )
                    label_batch['label_act'] = padded_act

                if sub_key == 'label_min_act':
                    padded_min_act = pad_sequence(
                        [(torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for v in sub_values if v is not None],

                        batch_first=True,
                        padding_value=-100.0
                    )
                    label_batch['label_min_act'] = padded_min_act

                # 'valence_reg', 'arousal_reg' 등 다른 텐서 타입의 레이블 처리
                elif sub_values and sub_values[0] is not None and isinstance(sub_values[0], torch.Tensor):
                    label_batch[sub_key] = torch.stack([v for v in sub_values if v is not None])
            collated[key] = label_batch
    
    # 최종 출력 딕셔너리 생성
    batch_out = {
        'imu_emotion': collated.get('imu'),
        'imu_e_lens': torch.tensor([len(d['imu']) for d in batch], dtype=torch.long),
        'ppg_emotion': collated.get('ppg'),
        'veh_emotion': collated.get('veh'),
        'ppg_rr_emotion': collated.get('ppg_rr'),
        'ppg_rmssd_emotion': collated.get('ppg_rmssd'),
        'ppg_sdnn_emotion': collated.get('ppg_sdnn'),
        'scenario_time_e': collated.get('scenario_time'),
        'scenario_evt_e': collated.get('scenario_evt'),
        'phase_evt_e': collated.get('phase_evt'),
        'scenario_type_e': collated.get('scenario_type'),
        'valence_reg_emotion': collated.get('labels', {}).get('valence_reg'),
        'arousal_reg_emotion': collated.get('labels', {}).get('arousal_reg'),
        'survey_e': collated.get('static'),
        'label_motion': collated.get('labels', {}).get('label_motion'),
        'label_tot': collated.get('labels', {}).get('label_tot'),
        'label_act': collated.get('labels', {}).get('label_act'),
        'label_min_act': collated.get('labels', {}).get('label_min_act'),
        
        'imu_motion': collated.get('imu'),
        'veh_mask_emotion': torch.ones((collated.get('veh').shape[0], 1000), dtype=torch.bool) if 'veh' in collated and collated.get('veh') is not None else torch.empty(0),
    }
    
    batch_out['sc_motion_evt'] = collated.get('scenario_evt')
    batch_out['sc_motion_type'] = collated.get('scenario_type')
    batch_out['sc_motion_phase'] = collated.get('phase_evt')
    batch_out['sc_motion_time'] = collated.get('scenario_time')

    if 'veh' in collated and collated.get('veh') is not None and collated.get('veh').shape[2] > 0:
        veh_data = collated.get('veh') # (Batch, Length, Channels) 형태일 것으로 추정
        if veh_data.dim() == 3:
            # 데이터가 (B, L, C) 형태라면 (B, C, L)로 변환
            batch_out['veh_motion'] = veh_data.permute(0, 2, 1)
        else:
            # 이미 (B, C, L) 형태라면 그대로 사용
            batch_out['veh_motion'] = veh_data
        batch_out['veh_mask_motion'] = torch.ones(len(batch), collated.get('veh').shape[1])
    
    # collated 딕셔너리에 'indices' 키가 있는지 확인하고, 최종 batch_out에 추가합니다.
    if 'indices' in collated:
        batch_out['indices'] = torch.tensor(collated['indices'], dtype=torch.long)
    # --------------------

    return batch_out

# def collate_fn_unified(batch, **kwargs):
#     keys = batch[0].keys()
#     collated = {key: [d[key] for d in batch] for key in keys}

#     for key, values in collated.items():
#         if key in ('imu', 'ppg', 'veh'):
#             padded = pad_sequence(values, batch_first=True, padding_value=0.0)
#             collated[key] = padded
            
#         elif key == 'ppg_rr':
#             values_no_none = [v for v in values if v is not None and v.nelement() > 0]
#             if not values_no_none:
#                 padded = torch.full((len(batch), 1), -100.0)
#             else:
#                 padded = pad_sequence(values_no_none, batch_first=True, padding_value=-100.0)
#             collated[key] = padded

#         elif key in ('ppg_rmssd', 'ppg_sdnn', 'scenario_time', 'scenario_evt', 'phase_evt', 'scenario_type', 'static'):
#             if key in ('scenario_evt', 'scenario_type', 'phase_evt'):
#                 valid_values = [v if pd.notna(v) else 0 for v in values]
#                 collated[key] = torch.tensor(valid_values, dtype=torch.long)
#             else:
#                 # [수정] .stack()은 텐서 리스트에만 적용 가능하므로, values가 텐서인지 확인
#                 if all(isinstance(v, torch.Tensor) for v in values):
#                     collated[key] = torch.stack(values)
#                 # 텐서가 아닌 다른 타입은 그대로 두거나 다른 처리가 필요할 수 있음
#                 # 예를 들어, static이 텐서가 아니라면 변환 과정이 필요
#                 elif key == 'static' and not isinstance(values[0], torch.Tensor):
#                      collated[key] = torch.tensor(np.array(values), dtype=torch.float32)

#         # --- [핵심 수정] labels 처리 로직 ---
#         # Dataset이 반환하는 top-level 키를 직접 사용하도록 변경
#         elif key in ['label_motion', 'label_act']:
#              # None이 아닌 numpy/tensor 리스트를 pad_sequence로 처리
#              valid_tensors = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in values if v is not None]
#              if valid_tensors:
#                  padding_value = -100.0 if key == 'label_act' else 0
#                  collated[key] = pad_sequence(valid_tensors, batch_first=True, padding_value=float(padding_value))
        
#         elif key in ['valence_reg', 'arousal_reg', 'label_tot']:
#             # 스칼라 값 또는 단일 값을 가진 텐서들을 stack으로 묶음
#             if values and values[0] is not None:
#                 collated[key] = torch.stack([v for v in values if v is not None])

#     # --- [핵심 수정] 최종 batch_out 생성 로직 ---
#     # collated 딕셔너리에서 직접 키를 가져와 최종 배치를 구성
#     batch_out = {
#         'imu_emotion': collated.get('imu'),
#         'ppg_emotion': collated.get('ppg'),
#         'veh_emotion': collated.get('veh'),
#         'ppg_rr_emotion': collated.get('ppg_rr'),
#         'ppg_rmssd_emotion': collated.get('ppg_rmssd'),
#         'ppg_sdnn_emotion': collated.get('ppg_sdnn'),
#         'scenario_time_e': collated.get('scenario_time'),
#         'scenario_evt_e': collated.get('scenario_evt'),
#         'phase_evt_e': collated.get('phase_evt'),
#         'scenario_type_e': collated.get('scenario_type'),
#         'survey_e': collated.get('static'),
        
#         # Motion 태스크용 키 이름 (Emotion과 동일한 데이터 사용)
#         'imu_motion': collated.get('imu'),
#         'sc_motion_evt': collated.get('scenario_evt'),
#         'sc_motion_type': collated.get('scenario_type'),
#         'sc_motion_phase': collated.get('phase_evt'),
#         'sc_motion_time': collated.get('scenario_time'),

#         # 모든 라벨 데이터
#         'valence_reg_emotion': collated.get('valence_reg'),
#         'arousal_reg_emotion': collated.get('arousal_reg'),
#         'label_motion': collated.get('label_motion'),
#         'label_tot': collated.get('label_tot'),
#         'label_act': collated.get('label_act'),
#     }
    
#     # veh 데이터 처리 (permute 및 mask 생성)
#     if 'veh' in collated and collated.get('veh') is not None:
#         veh_data = collated.get('veh')
#         batch_out['veh_motion'] = veh_data.permute(0, 2, 1) if veh_data.dim() == 3 else veh_data
#         batch_out['veh_mask_motion'] = torch.ones(len(batch), veh_data.shape[1], dtype=torch.bool)

#     # 누락된 키는 None으로 채워넣기 (안전장치)
#     final_keys = ['imu_emotion', 'ppg_emotion', 'veh_emotion', 'ppg_rr_emotion', 'ppg_rmssd_emotion', 
#                   'ppg_sdnn_emotion', 'scenario_time_e', 'scenario_evt_e', 'phase_evt_e', 
#                   'scenario_type_e', 'survey_e', 'imu_motion', 'veh_motion', 'sc_motion_evt', 
#                   'sc_motion_type', 'sc_motion_phase', 'sc_motion_time', 'valence_reg_emotion', 
#                   'arousal_reg_emotion', 'label_motion', 'label_tot', 'label_act', 'veh_mask_motion']
    
#     for k in final_keys:
#         if k not in batch_out:
#             batch_out[k] = None

#     return batch_out