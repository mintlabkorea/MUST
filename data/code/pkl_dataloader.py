import os
import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy import signal
from typing import Dict, Any, List
print("\n\n--- [확인] 최신 버전의 pkl_dataloader.py 파일을 읽고 있습니다. ---\n\n")

# ─── PPG 유틸리티 함수 ───────────────────────────────────────────────────────────
def polynominal_detrending(arr, degree=3):
    x = np.linspace(0, 20, len(arr))
    coeffs = np.polyfit(x, np.array(arr), degree)
    poly = np.poly1d(coeffs)
    return np.array(arr) - poly(x)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return signal.butter(order, [low, high], btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return signal.filtfilt(b, a, data)

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def detect_peaks(data, fs, min_distance_sec):
    min_dist = int(min_distance_sec * fs)
    peaks, _ = signal.find_peaks(data, distance=min_dist)
    return peaks

def calculate_hrv(rr_intervals):
    if len(rr_intervals) < 2:
        return np.nan, np.nan
    rr_diff = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(rr_diff**2))
    sdnn  = np.std(rr_intervals)
    return rmssd, sdnn

# ─── Dataset 정의 ───────────────────────────────────────────────────────────────
class PKLMultiModalDataset(Dataset):
    @staticmethod
    def safe_int(df: pd.DataFrame, idx: int, col: str, default: int = 0) -> int:
        try:
            v = df.at[idx, col]
            return int(v) if pd.notna(v) else default
        except (KeyError, IndexError):
            return default

    def __init__(
        self,
        *,
        data_map: Dict[str, Dict[str, Any]],
        participant_ids: List[str],
        survey_df: pd.DataFrame,
        imu_cols:   List[str],
        ppg_cols:   List[str],
        sc_cols:    List[str],
        veh_cols:   List[str],
        label_cols: List[str],
        mode: str = "timeline",
        window_sec: float = 3.0,
        window_stride: float = 1.5,
        fs: int = 100
    ):
        self.mode = mode
        self.data       = data_map
        self.ids        = participant_ids
        self.survey_df = survey_df
        self.imu_cols   = imu_cols
        self.ppg_cols   = ppg_cols
        self.sc_cols    = sc_cols
        self.veh_cols   = veh_cols
        self.label_cols = label_cols
        self.mode       = mode
        self.window_sec    = window_sec
        self.window_stride = window_stride
        self.fs            = fs
        self.include_indices = False
        self.phase_map = {"before": 0, "on": 1, "after": 2}
        self.IGNORE_LABEL = -100.0
        self.win_samp    = int(self.window_sec * self.fs)
        self.stride_samp = int(self.window_stride * self.fs)

        all_scn_ids, all_type_ids = set(), set()
        for pid_data in data_map.values():
            if 'sc' in pid_data:
                sc_df = pid_data['sc']
                if 'sc_scenario' in sc_df.columns:
                    all_scn_ids.update(pd.to_numeric(sc_df["sc_scenario"], errors="coerce").dropna().unique())
                if 'sc_scenario_type' in sc_df.columns:
                    all_type_ids.update(pd.to_numeric(sc_df["sc_scenario_type"], errors="coerce").dropna().unique())

        self.all_scn_ids = sorted([int(v) for v in all_scn_ids])
        self.all_type_ids = sorted([int(v) for v in all_type_ids])

        self.scenario_map = {orig: idx for idx, orig in enumerate(all_scn_ids)}
        self.sc_type_map  = {orig: idx for idx, orig in enumerate(all_type_ids)}

        self.samples = []
        for pid in self.ids:
            df_sc = self.data[pid].get('sc')
            df_label = self.data[pid].get('label')
            if df_label is None: continue
            n = len(df_label)

            # --- Label-End Mode Logic (모든 레이블을 윈도우 끝으로 사용) ---
            if self.mode == "timeline":
                # 1) valence/arousal 유효 프레임 모두 가져오기
                valid_label_idx = df_label.index[
                    (df_label["label_valence"] != self.IGNORE_LABEL) &
                    (df_label["label_arousal"] != self.IGNORE_LABEL)
                ].tolist()
                if not valid_label_idx:
                    continue

                # 2) 필터링 없이 모든 레이블 인덱스를 윈도우 끝(end_idx)으로 사용
                for center in valid_label_idx:
                    self.samples.append({
                        "pid": pid,
                        "mode": "timeline",
                        "end_idx": int(center)
                    })
            # --- Phase Mode Logic ---
            elif self.mode == "phase":
                if df_sc is None: continue
                phase_idx_arr = df_sc['sc_phase'].dropna().index.to_numpy()
                valid_label_idx_arr = df_label.index[
                    (df_label["label_valence"] != self.IGNORE_LABEL) &
                    (df_label["label_arousal"] != self.IGNORE_LABEL)
                ].to_numpy()

                if phase_idx_arr.size == 0 or valid_label_idx_arr.size == 0: continue
                min_distances = np.min(np.abs(valid_label_idx_arr[:, None] - phase_idx_arr), axis=1)
                TOLERANCE = int(1.0 * self.fs)
                raw_centers = valid_label_idx_arr[min_distances <= TOLERANCE].tolist()
                if not raw_centers: continue
                
                min_dist = int(self.win_samp / 2) 
                filtered_centers = [raw_centers[0]]
                for center in raw_centers[1:]:
                    if center - filtered_centers[-1] >= min_dist:
                        filtered_centers.append(center)
                
                for center in filtered_centers:
                    self.samples.append({"pid": pid, "mode": "phase", "end_idx": int(center)})
            
            
            # --- Motion Mode Logic ---
            elif self.mode == "motion":
                for start in range(0, max(1, n - self.win_samp + 1), self.stride_samp):
                    end = start + self.win_samp
                    if end > n:
                        end, start = n, max(0, n - self.win_samp)
                    self.samples.append({"pid": pid, "mode": "motion", "start_idx": int(start), "end_idx": int(end)})
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

        # ppg_stats 계산 시에도 통합된 DataFrame에서 직접 PPG 컬럼을 가져옵니다.
        self.ppg_stats: Dict[str, Dict[str, float]] = {}
        for pid in self.ids:
            if 'ppg' in self.data[pid] and 'ppg_ppg_raw' in self.data[pid]['ppg'].columns:
                ppg_data = self.data[pid]['ppg']['ppg_ppg_raw'].values.astype(np.float32)
                mean_raw = np.nanmean(ppg_data) if ppg_data.size > 0 else 0.0
                std_raw  = np.nanstd(ppg_data)  if ppg_data.size > 0 else 1.0
                self.ppg_stats[pid] = {"mean": float(mean_raw), "std": float(std_raw) if std_raw > 0 else 1.0}
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.samples[idx]
        pid = rec["pid"]

        if rec["mode"] in ("phase", "timeline"):
            center = rec["end_idx"]
            start_idx = max(0, center - self.win_samp)
            end_idx = center
        elif rec["mode"] == "motion":
            start_idx, end_idx = rec["start_idx"], rec["end_idx"]
        else:
            raise ValueError(f"Unexpected sample mode: {rec['mode']}")

        def get_padded_window(modality: str, cols: List[str]) -> pd.DataFrame:
            df = self.data[pid][modality]
            win_slice = df[cols].iloc[start_idx:end_idx]
            if len(win_slice) < self.win_samp:
                pad_len = self.win_samp - len(win_slice)
                pad_df = pd.DataFrame(np.zeros((pad_len, len(cols))), columns=cols)
                return pd.concat([pad_df, win_slice], ignore_index=True)
            return win_slice

        win_imu = get_padded_window('imu', self.imu_cols)
        win_ppg = get_padded_window('ppg', self.ppg_cols)
        win_veh = get_padded_window('veh', self.veh_cols)
        win_label = get_padded_window('label', self.label_cols)


        # --- HRV 계산 ---
        raw = win_ppg["ppg_ppg_raw"].values.astype(np.float32)
        detrended = polynominal_detrending(raw, degree=3)
        band_passed = butter_bandpass_filter(detrended, 0.5, 3, self.fs, order=4)
        smoothed = moving_average(band_passed, int(self.fs * 0.5))
        peaks = detect_peaks(smoothed, self.fs, min_distance_sec=0.6)
        rr = np.diff(peaks / self.fs) if len(peaks) > 1 else np.zeros((0,), dtype=np.float32)
        rmssd, sdnn = calculate_hrv(rr)
        rr_len  = int(self.window_sec * 2)
        pad_amt = max(0, rr_len - len(rr))
        rr_fixed = np.concatenate([rr[:rr_len], np.full(pad_amt, -100.0, dtype=np.float32)]).astype(np.float32)

        # --- 데이터 텐서 변환 ---
        stats = self.ppg_stats[pid]
        raw_norm = (raw - stats["mean"]) / stats["std"]
        ppg_seq = torch.from_numpy(raw_norm.astype(np.float32)).unsqueeze(-1)
        imu_seq = torch.from_numpy(win_imu.values.astype(np.float32))
        veh_df = win_veh.copy() # veh_cols가 이미 적용되었으므로 win_veh를 바로 사용
        
        for col in veh_df.columns:
            if col.endswith("_collision"):
                 veh_df[col] = veh_df[col].apply(lambda x: 1 if (isinstance(x, str) and x.lower() == "collision") or (pd.notna(x) and float(x) >= 1) else 0)
        veh_seq = torch.from_numpy(veh_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32))
        
        # --- 시나리오/이벤트 정보 ---
        center_idx = rec["end_idx"]
        sc_df = self.data[pid]['sc']
        label_df = self.data[pid]['label']

        raw_evt = self.safe_int(sc_df, center_idx, "sc_scenario", default=self.all_scn_ids[0] if self.all_scn_ids else 0)
        raw_typ = self.safe_int(sc_df, center_idx, "sc_scenario_type", default=self.all_type_ids[0] if self.all_type_ids else 0)
        try:
            # 1. 'after', 'on' 같은 문자열 값을 직접 가져옵니다.
            phase_str = sc_df.at[center_idx, "sc_phase"]
            # 2. self.phase_map을 사용해 문자열을 숫자로 변환합니다. (e.g., 'after' -> 2)
            phase_evt_val = self.phase_map.get(phase_str, 0) # 맵에 없는 값이면 0('before')으로 처리
        except KeyError:
            # 해당 인덱스(center_idx)에 sc_phase 값이 없는 경우를 대비한 안전장치
            phase_evt_val = 0
        
        center_ts = float(label_df.at[center_idx, "timestamps"]) if "timestamps" in label_df.columns else -100.0

        # --- 라벨 구성 ---
        center_valence = float(label_df.at[center_idx, "label_valence"])
        center_arousal = float(label_df.at[center_idx, "label_arousal"])

        # 윈도우의 마지막 시점(center_idx)의 TOT 값을 기준으로 클래스를 결정합니다.
        try:
            continuous_tot_value = label_df.at[center_idx, 'label_tot']
            
            # 이전에 분석한 데이터 분포의 백분위수 기준을 적용합니다.
            if 0 < continuous_tot_value <= 0.93:
                tot_label = 0  # 빠름 (Fast)
            elif 0.93 < continuous_tot_value <= 1.93:
                tot_label = 1  # 보통 (Medium)
            elif continuous_tot_value > 1.93:
                tot_label = 2  # 느림 (Slow)
            else:
                tot_label = -100 # 그 외의 경우는 학습에서 무시
        except KeyError:
            tot_label = -100 # 라벨 값이 없는 경우
        
        final_tot_label = torch.tensor(tot_label, dtype=torch.long)
        
        # # --- ACT 라벨 생성 (시퀀스 + 마스킹 + minACT 계산) ---
        # # 1. 사용할 모든 ACT 컬럼을 지정합니다.
        # act_cols = ["label_veh1_ACT", "label_veh2_ACT", "label_veh3_ACT"]
        
        # # 2. win_label에 해당 컬럼들이 있는지 확인
        # if all(col in win_label.columns for col in act_cols):
        #     arr_act = win_label[act_cols].values.astype(np.float32) # shape: (300, 3)
        # else:
        #     # 컬럼이 하나라도 없으면 -100으로 채워진 배열 생성
        #     arr_act = np.full((self.win_samp, len(act_cols)), -100.0, dtype=np.float32)

        # # 3. 유효하지 않은 값 (-100, 15 이상)을 마스킹.
        # arr_act[arr_act >= 15] = -100.0
        # arr_act = np.nan_to_num(arr_act, nan=-100.0)

        # # 4. [신규] minACT 라벨 생성
        # #    -100은 계산에서 제외하기 위해 임시로 큰 값(inf)으로 변경
        # arr_act_masked = np.where(arr_act == -100.0, np.inf, arr_act)
        # #    최솟값을 계산하고, inf인 경우(모든 값이 -100이었던 경우) 다시 -100으로 복원
        # min_act = np.min(arr_act_masked, axis=1, keepdims=True) # shape: (300, 1)
        # min_act[min_act == np.inf] = -100.0
        
        # # 5. 최종 텐서 생성
        # final_act_label = torch.from_numpy(arr_act)           # shape: (300, 3)
        # final_min_act_label = torch.from_numpy(min_act)       # shape: (300, 1)
        
        # --- ACT 라벨 생성 (시퀀스 + 마스킹 + minACT 계산) ---

        # 1. 사용할 모든 ACT 컬럼을 지정하고, 윈도우 전체에서 데이터를 먼저 추출합니다.
        act_cols = ["label_veh1_ACT", "label_veh2_ACT", "label_veh3_ACT"]

        # 윈도우(win_label)에 해당 컬럼들이 있는지 확인합니다.
        if all(col in win_label.columns for col in act_cols):
            # (수정) arr_act를 여기서 먼저 정의하고 처리합니다.
            arr_act = win_label[act_cols].values.astype(np.float32) # shape: (win_samp, 3)
        else:
            # 컬럼이 하나라도 없으면 -100으로 채워진 배열을 생성합니다.
            arr_act = np.full((self.win_samp, len(act_cols)), -100.0, dtype=np.float32)

        # 2. Per-frame 데이터 클리닝: 유효하지 않은 값 (15 이상) 및 NaN을 -100으로 통일합니다.
        #    이는 final_act_label을 위한 전처리입니다.
        arr_act[arr_act >= 15] = -100.0
        arr_act = np.nan_to_num(arr_act, nan=-100.0)

        # 3. (수정) final_act_label을 먼저 생성합니다.
        #    이 텐서는 각 프레임별 3개 차량의 ACT 값을 그대로 담고 있습니다.
        final_act_label = torch.from_numpy(arr_act.copy())

        # --- 시나리오 단위 minACT 라벨 계산 ---
        # 4. 시나리오 구간을 식별하기 위해 'sc' 데이터프레임에서 sc_phase 컬럼을 가져옵니다.
        win_sc_slice = self.data[pid]['sc'].iloc[start_idx:end_idx]
        
        if 'sc_phase' in win_sc_slice.columns:
            phases_series = win_sc_slice['sc_phase'].replace(self.phase_map)
            phases = pd.to_numeric(phases_series, errors='coerce').fillna(-1).values
        else:
            phases = np.full(len(win_sc_slice), -1)

        is_in_scenario = (phases >= 0) & (phases <= 2)

        if not np.any(is_in_scenario):
            # 윈도우 내에 시나리오 구간이 없으면 minACT는 유효하지 않습니다.
            scenario_min_act = -100.0
        else:
            # 5. 시나리오 전체 구간(phase 0, 1, 2) 내에서 단일 minACT 값을 계산합니다.
            scenario_indices = np.where(is_in_scenario)[0]
            # minACT 계산 시에는 label 데이터를 사용하는 것이 맞습니다.
            scenario_df = win_label.iloc[scenario_indices]

            # 시나리오 구간 내의 veh1, veh2 ACT 값들을 가져옵니다.
            veh1_act_values = scenario_df.get('label_veh1_ACT', pd.Series(np.nan)).values
            veh2_act_values = scenario_df.get('label_veh2_ACT', pd.Series(np.nan)).values
            
            # 유효한 ACT 값 필터링 (0 초과, 15 미만)
            valid_veh1_act = veh1_act_values[(veh1_act_values > 0) & (veh1_act_values < 15)]
            valid_veh2_act = veh2_act_values[(veh2_act_values > 0) & (veh2_act_values < 15)]

            # 우선순위에 따라 minACT 계산
            if len(valid_veh1_act) > 0:
                # veh1_ACT의 유효값이 있다면, 그중 최솟값을 사용합니다.
                scenario_min_act = np.min(valid_veh1_act)
            elif len(valid_veh2_act) > 0:
                # veh1_ACT 유효값이 없고, veh2_ACT 유효값이 있다면, 그중 최솟값을 사용합니다.
                scenario_min_act = np.min(valid_veh2_act)
            else:
                # 둘 다 유효값이 없다면, 해당 시나리오의 minACT는 -100입니다.
                scenario_min_act = -100.0

        # 6. 'before' phase 프레임에 scenario_min_act 값을 할당합니다.
        #    final_min_act_label은 'before' phase에서 예측할 타겟이 됩니다.
        final_min_act_label = torch.full((self.win_samp, 1), -100.0, dtype=torch.float32)
        before_phase_indices = np.where(phases == 0)[0]

        if len(before_phase_indices) > 0:
            final_min_act_label[before_phase_indices] = float(scenario_min_act)
        
        # Survey
        # static_features = None
        # SURVEY_FEATURE_DIM = 26 # pre_survey.csv의 실제 피처 수 (PID 컬럼 제외)
        
        # Survey
        static_features = None
        fallback_survey = self.data[pid].get('survey')
        if fallback_survey is not None:
            static_features = torch.from_numpy(np.asarray(fallback_survey, dtype=np.float32))

        try:
            pid_num = int("".join(filter(str.isdigit, str(pid))))

            # survey_df가 없거나 PID가 없으면 0으로 대체합니다.
            if static_features is None and self.survey_df is not None and pid_num in self.survey_df.index:
                survey_data = self.survey_df.loc[pid_num].values
                static_features = torch.from_numpy(survey_data.astype(np.float32))
        except (ValueError, KeyError, TypeError):
            pass

        if static_features is None:
            SURVEY_FEATURE_DIM = 26  # Legacy fallback for old external loaders.
            static_features = torch.zeros(SURVEY_FEATURE_DIM, dtype=torch.float32)


        #pid_num = int("".join(filter(str.isdigit, str(pid))))

        # survey_df에서 해당 pid_num의 데이터를 찾아 텐서로 변환
        # survey_data = self.survey_df.loc[pid_num].values
        # static_features = torch.from_numpy(survey_data.astype(np.float32))
        

        sample = {
            "imu": imu_seq, "ppg": ppg_seq, "veh": veh_seq, "static": static_features,
            "ppg_rr": torch.tensor(rr_fixed, dtype=torch.float32),
            "ppg_rmssd": torch.tensor(rmssd, dtype=torch.float32),
            "ppg_sdnn": torch.tensor(sdnn, dtype=torch.float32),
            "scenario_time": torch.tensor(center_ts, dtype=torch.float32),
            "scenario_evt": torch.tensor(self.scenario_map.get(raw_evt, 0), dtype=torch.long),
            "scenario_type": torch.tensor(self.sc_type_map.get(raw_typ, 0), dtype=torch.long),
            "phase_evt": torch.tensor(phase_evt_val, dtype=torch.long),
            "labels": {
                "label_motion": win_label["label_motion"].values.astype(np.int64),
                "valence_reg": torch.tensor([center_valence], dtype=torch.float32),
                "arousal_reg": torch.tensor([center_arousal], dtype=torch.float32),
                "label_tot": final_tot_label,
                "label_act": final_act_label,
                "label_min_act": final_min_act_label,
            },
            "pid": pid, "sample_mode": self.mode
        }

        if self.include_indices:
            sample['indices'] = idx
            
        return sample
    

# ─── [수정] main 블록: 시나리오별 minACT 통계 계산 ────────────────────────────────
if __name__ == "__main__":
    import pickle
    import pandas as pd
    
    # ------------------- 1. 데이터 로딩 및 통합 -------------------
    # 데이터 경로를 실제 환경에 맞게 수정해주세요.
    PKL_PATH = "/home/mintlab01/main_6/data/data/train/train_ver2.pkl"
    all_dfs = []

    print(f"Loading '{PKL_PATH}'...")
    try:
        with open(PKL_PATH, "rb") as f:
            data_dict = pickle.load(f)
        print("File loaded. Integrating DataFrames...")

        for pid, participant_df in data_dict.items():
            if isinstance(participant_df, pd.DataFrame):
                participant_df['pid'] = pid
                all_dfs.append(participant_df)
        
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"Integrated data for {len(data_dict)} participants. (Total rows: {len(df)})")

    except FileNotFoundError:
        print(f"Error: Could not find '{PKL_PATH}'. Please check the file path.")
        exit()

    # --- 2. 데이터 전처리 ---
    print("Preprocessing data (creating unique scenario instance ID)...")
    phase_map = {"before": 0, "on": 1, "after": 2}
    df['sc_phase_numeric'] = df['sc_phase'].map(phase_map)
    df['sc_phase_numeric'].fillna(pd.to_numeric(df['sc_phase'], errors='coerce'), inplace=True)
    df['sc_instance_id'] = (df.groupby('pid')['sc_phase_numeric'].diff() < 0).cumsum()

    # --- 3. 통계 계산 ---
    print("\n--- Calculating statistics for minACT by Scenario ID ---")

    def calculate_scenario_min_act(scenario_df):
        """한 시나리오 인스턴스에서 minACT 값을 계산하는 함수"""
        veh1_act_values = scenario_df.get('label_veh1_ACT', pd.Series(dtype=float)).values
        veh2_act_values = scenario_df.get('label_veh2_ACT', pd.Series(dtype=float)).values
        
        valid_veh1_act = veh1_act_values[(veh1_act_values > 0) & (veh1_act_values < 15)]
        valid_veh2_act = veh2_act_values[(veh2_act_values > 0) & (veh2_act_values < 15)]

        if len(valid_veh1_act) > 0:
            return np.min(valid_veh1_act)
        elif len(valid_veh2_act) > 0:
            return np.min(valid_veh2_act)
        else:
            return np.nan

    # 각 시나리오 인스턴스(참가자별, 시나리오별) 그룹화
    scenario_groups = df[df['sc_phase_numeric'].isin([0, 1, 2])].groupby(['pid', 'sc_instance_id'])
    
    # 각 인스턴스에 대해 minACT 계산
    min_act_per_instance = scenario_groups.apply(calculate_scenario_min_act).rename('min_act')
    
    # 각 인스턴스의 시나리오 이벤트 ID (sc_scenario) 추출
    scenario_id_per_instance = scenario_groups['sc_scenario'].first().rename('sc_scenario')

    # minACT 값과 시나리오 ID를 결합하고, minACT가 NaN인 경우 제거
    scenario_stats_df = pd.concat([min_act_per_instance, scenario_id_per_instance], axis=1).dropna()

    # 시나리오 이벤트 ID(sc_scenario)로 그룹화하여 최종 통계 계산
    final_stats = scenario_stats_df.groupby('sc_scenario')['min_act'].agg(['mean', 'min', 'max', 'std'])

    print("\n--- Scenario-level minACT Statistics ---")
    print(final_stats)

    # --- 4. CSV 파일로 저장 ---
    output_filename = 'scenario_minACT_statistics.csv'
    final_stats.to_csv(output_filename)
    print(f"\n✅ Statistics successfully saved to '{output_filename}'")
