import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import List
from typing import List, Dict, Any

class PKLMultiModalDatasetBaseline(Dataset):
    def __init__(self,
                 participant_ids: List[str],
                 data_map: Dict[str, Dict[str, Any]],
                 veh_cols: List[str],
                 fs: int,
                 mode: str,
                 window_sec: int,
                 window_stride: int):
        self.pids = participant_ids
        self.data_map = data_map
        self.veh_cols = veh_cols
        self.fs = fs
        self.mode = mode
        self.window_size = int(window_sec * fs)
        self.stride = int(window_stride * fs)

        # [수정] 컬럼 이름을 유연하게 찾도록 수정
        self.tot_col_name = None
        self.act_col_name = None

        self.samples = self._create_samples()
        print(f"[Baseline Dataloader] Mode: '{self.mode}', Found {len(self.samples)} samples.")

    def _create_samples(self):
        samples = []
        for pid in self.pids:
            # [수정] 'main' 키 대신 'label' 데이터프레임을 사용
            if 'label' not in self.data_map[pid] or self.data_map[pid]['label'] is None:
                continue
            df_label = self.data_map[pid]['label']
            
            # [수정] 첫 번째 참가자 데이터로 실제 컬럼 이름이 무엇인지 확인
            if self.tot_col_name is None:
                if 'TOT' in df_label.columns: self.tot_col_name = 'TOT'
                elif 'label_tot' in df_label.columns: self.tot_col_name = 'label_tot'
            if self.act_col_name is None:
                if 'VEH1_ACT_LONG_ACCEL' in df_label.columns: self.act_col_name = 'VEH1_ACT_LONG_ACCEL'
                elif 'label_veh1_ACT' in df_label.columns: self.act_col_name = 'label_veh1_ACT'

            if self.mode == 'tot':
                if not self.tot_col_name or self.tot_col_name not in df_label.columns: continue
                
                tot_series = df_label[self.tot_col_name]
                valid_mask = (tot_series != -100) & (tot_series < 15) & (tot_series.notna())
                changed_mask = tot_series.diff() != 0
                valid_indices = df_label.index[valid_mask & changed_mask]

                for idx in valid_indices:
                    start_idx = idx - self.window_size
                    if start_idx >= 0:
                        samples.append({'pid': pid, 'start': start_idx, 'end': idx, 'label_idx': idx})

            elif self.mode == 'act':
                if not self.act_col_name or self.act_col_name not in df_label.columns: continue

                num_frames = len(df_label)
                for i in range(0, num_frames - self.window_size, self.stride):
                    start_idx = i
                    end_idx = i + self.window_size
                    
                    act_window = df_label.iloc[start_idx:end_idx][self.act_col_name]
                    if not ((act_window == -100) | (act_window >= 15) | (act_window.isna())).all():
                         samples.append({'pid': pid, 'start': start_idx, 'end': end_idx})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        pid = sample_info['pid']
        start = sample_info['start']
        end = sample_info['end']

        # Slice data from individual dataframes
        df_veh_slice = self.data_map[pid]['veh'].iloc[start:end]
        df_sc_slice = self.data_map[pid]['sc'].iloc[start:end]
        df_label_slice = self.data_map[pid]['label'].iloc[start:end]

        # --- Process Vehicle Data (with collision fix) ---
        df_veh_processed = df_veh_slice.copy()
        for col in self.veh_cols:
            if 'collision' in col.lower():
                def process_collision_value(x):
                    if isinstance(x, str) and x.lower() == 'collision': return 1
                    try:
                        if pd.notna(x) and float(x) >= 1: return 1
                    except (ValueError, TypeError): pass
                    return 0
                df_veh_processed[col] = df_veh_processed[col].apply(process_collision_value)
        veh_values = df_veh_processed[self.veh_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
        veh = torch.tensor(veh_values.astype(np.float32))

        # --- [FIX] Process SC Data to be SCALAR-LIKE (use last time step) ---
        df_sc_slice_copy = df_sc_slice.copy()
        
        # Robustly handle 'sc_phase' and get the last value
        mapped_series = df_sc_slice_copy['sc_phase'].replace({"before": 0, "on": 1, "after": 2})
        numeric_series = pd.to_numeric(mapped_series, errors='coerce')
        final_series = numeric_series.fillna(0)
        sc_phase_values = final_series.values.astype(np.int64)
        last_sc_phase = sc_phase_values[-1] if len(sc_phase_values) > 0 else 0

        # Get the last value for other SC features
        last_sc_evt = df_sc_slice_copy['sc_scenario'].values[-1] if len(df_sc_slice_copy) > 0 else 0
        last_sc_type = df_sc_slice_copy['sc_scenario_type'].values[-1] if len(df_sc_slice_copy) > 0 else 0
        
        # Robustly find timestamps and get the last value
        if 'timestamps' in df_label_slice.columns:
            sc_time_values = df_label_slice['timestamps'].values
        elif 'timestamps' in df_sc_slice.columns:
            sc_time_values = df_sc_slice['timestamps'].values
        else:
            sc_time_values = np.arange(len(df_veh_slice))
        last_timestamp = sc_time_values[-1] if len(sc_time_values) > 0 else 0.0
        
        # Create SCALAR tensors for the encoder
        sc_evt = torch.tensor(last_sc_evt, dtype=torch.long)
        sc_type = torch.tensor(last_sc_type, dtype=torch.long)
        sc_phase = torch.tensor(last_sc_phase, dtype=torch.long)
        sc_time = torch.tensor(last_timestamp, dtype=torch.float32)

        item = {'veh': veh, 'sc_evt': sc_evt, 'sc_type': sc_type, 'sc_phase': sc_phase, 'sc_time': sc_time}

        # Extract labels based on the task mode
        if self.mode == 'tot':
            label_idx = sample_info['label_idx']
            # 먼저 연속적인 TOT 값을 가져옵니다.
            continuous_tot_value = self.data_map[pid]['label'].loc[label_idx, self.tot_col_name]

            # [수정] 연속 값을 이산적인 클래스로 변환 (Binning)
            if 0 < continuous_tot_value <= 0.93:
                label = 0  # 긴급 (Urgent)
            elif 0.93 < continuous_tot_value <= 1.93:
                label = 1  # 주의 (Caution)
            elif continuous_tot_value > 1.93:
                label = 2  # 안전 (Safe)
            else:
                # 유효하지 않은 값이나 다른 경우는 무시
                label = -100 # CrossEntropyLoss의 ignore_index

            item['label'] = torch.tensor(label, dtype=torch.long) # long 타입으로 변경

        elif self.mode == 'act':
            label = torch.tensor(df_label_slice[self.act_col_name].values, dtype=torch.float32)
            label = torch.nan_to_num(label, nan=-100.0)
            label[label >= 15] = -100.0
            item['label'] = label

        return item
