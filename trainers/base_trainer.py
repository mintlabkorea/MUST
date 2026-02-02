import os
import pickle
from typing import Dict, List, Any
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

import numpy as np
import pandas as pd
import torch


class dataProcessor:
    """
    데이터 로딩, 전처리, 분할 및 분석을 총괄하는 클래스.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_map: Dict[str, Dict[str, Any]] = {}
        self.train_keys: List[str] = []
        self.val_keys: List[str] = []
        self.test_keys: List[str] = []
        self.dataset_kwargs: Dict[str, Any] = {}
        self.imu_cols, self.ppg_cols, self.sc_cols, self.veh_cols, self.label_cols = [], [], [], [], []
        self.survey_df = None

    def prepare(self) -> None:
        """데이터 준비를 위한 메인 메서드."""        
        self._load_and_restructure_pkl()
        self._load_and_integrate_survey()
        self._split_keys()
        self._prepare_dataset_metadata()
        
        print("데이터 준비가 완료되었습니다.")

    def _load_and_restructure_pkl(self) -> None:
        """Raw PKL 파일을 로드하고, flat DataFrame을 모달리티별 딕셔너리로 재구성합니다."""
        with open(self.cfg.Project.pkl_all, 'rb') as f:
            raw_data = pickle.load(f)

        self.data_map = raw_data.get('all', raw_data)

        for pid, df in self.data_map.items():
            if isinstance(df, pd.DataFrame):
                self.data_map[pid] = {
                    'imu':   df.filter(regex='^imu_'),
                    'ppg':   df.filter(regex='^ppg_'),
                    'sc':    df.filter(regex='^sc_'),
                    'veh':   df.filter(regex='^veh_'),
                    'label': df.filter(regex='^label_')
                }
        print(f"PKL 로딩 및 재구성 완료: {len(self.data_map)}개 키")

    def _load_and_integrate_survey(self) -> None:
        """Survey CSV를 로드하고, data_map에 통합합니다."""
        try:
            survey_df = pd.read_csv(self.cfg.Project.survey_csv)

            # read_csv 직후, 첫 번째 컬럼(PID)을 인덱스로 설정합니다.
            pid_col_name = survey_df.columns[0]
            survey_df.set_index(pid_col_name, inplace=True)
            self.survey_df = survey_df # 인덱스가 설정된 df를 저장
            
            # survey_map 생성 로직은 인덱스가 설정되었으므로 약간 수정이 필요할 수 있으나,
            # 현재 코드(iterrows)는 인덱스 설정 여부와 관계없이 동일하게 동작합니다.
            survey_items = self.survey_df.columns.tolist()
            survey_map = {
                str(int(pid)): row[survey_items].astype(np.float32).to_numpy()
                for pid, row in self.survey_df.iterrows()
            }

            for pid in self.data_map:
                subj_id = pid.split('_')[0]
                self.data_map[pid]['survey'] = survey_map.get(
                    subj_id, np.zeros(len(survey_items), dtype=np.float32)
                )
            print("📊 Survey 데이터 통합 완료.")
        except Exception as e:
            print(f"Survey 데이터 로딩 실패: {e}. Survey 데이터를 0으로 채웁니다.")
            dummy_survey_len = self.cfg.Encoders.survey['input_dim']
            for pid in self.data_map:
                self.data_map[pid]['survey'] = np.zeros(dummy_survey_len, dtype=np.float32)

    def _split_keys(self) -> None:
        """
        [최종 수정] StratifiedGroupKFold를 사용해 데이터를 분할합니다.
        - Step 1: 각 데이터 키의 대표 라벨을 '최빈값(mode)'으로 추출합니다.
        - Step 2: 라벨이 -100인 키는 계층화에서 제외합니다.
        - Step 3: 유효한 라벨을 가진 키에 대해서만 계층적 그룹 분할을 수행합니다.
        - Step 4: -100 라벨 키는 학습(Train) 데이터에 포함시킵니다.
        """
        print("StratifiedGroupKFold 분할 시작 (Valence+Arousal 최빈값 기준, -100 라벨 제외)...")
        
        all_keys = sorted(self.data_map.keys())
        
        # 1. [수정] 유효한 라벨을 가진 키와 패딩(-100) 라벨을 가진 키를 분리
        valid_keys, valid_labels, valid_groups = [], [], []
        padding_keys = []

        for key in all_keys:
            subj_id = key.split('_')[0]
            
            # [핵심 수정] 대표 라벨을 최빈값(mode)으로 추출
            labels_v = self.data_map[key]['label']['label_valence']
            labels_a = self.data_map[key]['label']['label_arousal']
            
            # -100을 제외한 유효 라벨만 필터링
            valid_labels_v = labels_v[(labels_v >= 1) & (labels_v < 10)]
            valid_labels_a = labels_a[(labels_a >= 1) & (labels_a < 10)]
            # 유효 라벨이 하나라도 존재하면 최빈값을 계산
            if not valid_labels_v.empty and not valid_labels_a.empty:
                raw_v = valid_labels_v.mode().iloc[0]
                raw_a = valid_labels_a.mode().iloc[0]

                label_v = 0 if raw_v < 4 else (1 if raw_v < 7 else 2)
                label_a = 0 if raw_a < 4 else (1 if raw_a < 7 else 2)
                combined_label = label_v * 3 + label_a
                
                valid_keys.append(key)
                valid_labels.append(combined_label)
                valid_groups.append(subj_id)
            else:
                # 세그먼트 전체가 -100 라벨인 경우
                padding_keys.append(key)
        
        print(f"전체 키: {len(all_keys)}개 | 계층화 대상(유효 라벨): {len(valid_keys)}개 | 패딩 라벨: {len(padding_keys)}개")

        # 2. Test Set 분리 (Config 기준, 전체 키에서 수행)
        test_subj_set = set(self.cfg.Data.test_subjects)
        self.test_keys = [key for key in all_keys if key.split('_')[0] in test_subj_set]
        
        # 3. 유효한 키들 중에서 Test Set에 속하지 않는 키들로 Train/Val 분할 수행
        non_test_valid_indices = [i for i, g in enumerate(valid_groups) if g not in test_subj_set]
        train_val_keys = [valid_keys[i] for i in non_test_valid_indices]
        train_val_labels = [valid_labels[i] for i in non_test_valid_indices]
        train_val_groups = [valid_groups[i] for i in non_test_valid_indices]

        # [안전장치] 분할할 그룹이 있는지 확인
        if not train_val_groups:
            raise ValueError("테스트셋을 제외한 후 학습/검증에 사용할 유효한 피실험자 그룹이 없습니다. config의 test_subjects를 확인하세요.")

        num_total_subj = len(set(train_val_groups))
        num_val_subj = len(self.cfg.Data.val_subjects)
        n_splits = round(num_total_subj / num_val_subj) if num_val_subj > 0 and num_total_subj > num_val_subj else 5

        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.cfg.Project.seed)
        
        try:
            train_indices, val_indices = next(sgkf.split(train_val_keys, train_val_labels, train_val_groups))
            self.train_keys = [train_val_keys[i] for i in train_indices]
            self.val_keys = [train_val_keys[i] for i in val_indices]
        except ValueError:
            print("StratifiedGroupKFold 분할 중 오류 발생. 그룹 수가 부족하여 일반 분할로 대체합니다.")
            # fallback to simple random split based on subjects
            unique_train_val_groups = sorted(list(set(train_val_groups)))
            train_pids, val_pids = train_test_split(unique_train_val_groups, test_size=1/n_splits, random_state=self.cfg.Project.seed)
            
            self.train_keys = [k for k, g in zip(train_val_keys, train_val_groups) if g in train_pids]
            self.val_keys = [k for k, g in zip(train_val_keys, train_val_groups) if g in val_pids]

        # 4. 패딩 키 처리: Test Set에 속하지 않는 패딩 키들은 모두 Train Set에 추가
        padding_keys_for_train = [key for key in padding_keys if key.split('_')[0] not in test_subj_set]
        self.train_keys.extend(padding_keys_for_train)
        
        # 5. 최종 정리 및 확인
        self.train_keys = sorted(list(set(self.train_keys)))
        val_pids_from_split = sorted(list(set([k.split('_')[0] for k in self.val_keys])))
        self.cfg.Data.val_subjects = val_pids_from_split
        
        print(f"Config의 val_subjects가 분할 결과로 업데이트됨: {val_pids_from_split}")
        if not all([self.train_keys, self.val_keys, self.test_keys]):
            raise RuntimeError("Train/Val/Test 세트 중 하나 이상이 비어있습니다!")
        print(f"분할 완료: Train {len(self.train_keys)}, Val {len(self.val_keys)}, Test {len(self.test_keys)}개 키")

    def _prepare_dataset_metadata(self) -> None:
        """데이터셋 생성에 필요한 메타데이터(컬럼 목록 등)를 준비하고 검증합니다."""
        first_key = self.train_keys[0]
        sample_data = self.data_map[first_key]

        self.imu_cols = sample_data['imu'].columns.tolist()
        self.ppg_cols = sample_data['ppg'].columns.tolist()
        self.sc_cols = sample_data['sc'].columns.tolist()
        self.veh_cols = sample_data['veh'].columns.tolist()
        self.label_cols = sample_data['label'].columns.tolist()
        survey_len = len(sample_data['survey'])

        assert len(self.imu_cols) == self.cfg.Encoders.imu['input_dim'], "IMU input_dim 불일치"
        assert len(self.veh_cols) == self.cfg.Encoders.veh['input_dim'], "Vehicle input_dim 불일치"
        assert survey_len == self.cfg.Encoders.survey['input_dim'], "Survey input_dim 불일치"

        self.dataset_kwargs = {
            "data_map": self.data_map,
            "survey_df": self.survey_df,
            "imu_cols": self.imu_cols,
            "ppg_cols": self.ppg_cols,
            "sc_cols": self.sc_cols,
            "veh_cols": self.veh_cols,
            "label_cols": self.label_cols,
            "fs": self.cfg.Data.fs,
        }

class TrainerBase(dataProcessor):
    """
    [개선] DataProcessor를 상속받아 데이터 준비 기능을 내재화.
    공통적인 학습, 검증, 체크포인트 유틸리티를 제공하는 기본 트레이너.
    """
    def __init__(self, cfg, model, optimizer, loss_fn, device):
        super().__init__(cfg) # DataProcessor 초기화
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.current_epoch = 0

    def _load_state(self, path: str) -> None:
        """저장된 체크포인트에서 모델과 옵티마이저 상태를 로드합니다."""
        print(f"체크포인트를 로드합니다: {path}")
        state = torch.load(path, map_location=self.device)
        
        # [개선] state 딕셔너리의 키가 실제 모델에 존재하는지 확인하며 안전하게 로드
        for name, module in self.nets.items():
            if name in state.get('nets', {}):
                module.load_state_dict(state['nets'][name])
        
        for name, module in self.projs.items():
            if name in state.get('projs', {}):
                module.load_state_dict(state['projs'][name])

        # [개선] 각 head의 존재 여부를 확인하고 로드
        if hasattr(self, 'valence_head') and 'valence_head' in state:
            self.valence_head.load_state_dict(state['valence_head'])
        if hasattr(self, 'arousal_head') and 'arousal_head' in state:
            self.arousal_head.load_state_dict(state['arousal_head'])
        if hasattr(self, 'motion_head') and 'motion_head' in state:
            self.motion_head.load_state_dict(state['motion_head'])

        if 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
        if 'scaler' in state and hasattr(self, 'scaler'):
            self.scaler.load_state_dict(state['scaler'])
        
        self.current_epoch = state.get('epoch', 0)
        print(f"체크포인트 로드 완료. Epoch {self.current_epoch}에서 재시작합니다.")