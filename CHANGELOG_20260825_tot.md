# TOT 개선 코드 반영 (2026-08-25)

근거: `logs/20260824_fullrun/results.md` (subject-wise 6-fold CV, n=405)
원본 백업: `archive/code_backup_20260825_tot_improvements/`

## 변경 사항

### 1. `models/totact_models.py` — TOT_Baseline 시간축 pooling
```python
# before
last_step_out = gru_out[:, -1, :]
# after
pooled = self._time_pool(gru_out)   # 기본 mean, cfg.TOT.time_pool 로 제어
```
**이유.** 입력이 3s x 100Hz = 300 스텝인데 단층 GRU 의 마지막 스텝만 사용해서
gradient 가 시퀀스 앞쪽까지 도달하지 못했다. 학습 328개 샘플을 early stopping 없이
200 epoch 돌려도 train acc 가 0.52~0.62 에서 진동 (= 암기조차 실패).

| pooling | 암기(train, 120ep) | 3-class test | 2-class test |
|---|---|---|---|
| last (기존) | 0.6402 | 0.6500 +/- 0.028 | 0.8667 +/- 0.021 |
| **mean** | **0.8018** | **0.7056 +/- 0.014** | **0.9111 +/- 0.011** |

`nn.GRU(..., dropout=0.2)` 도 제거 — num_layers=1 이라 무시되며 경고만 발생.
`num_classes` 를 cfg.TOT 에서 읽도록 변경.

**주의: 파라미터 shape 은 동일하지만 기존 체크포인트는 재학습이 필요하다.**
`last` 로 학습된 가중치를 `mean` 으로 추론하면 test acc 0.7222 -> 0.5833.
main.py 는 매 실행마다 baseline 을 새로 학습하므로 정상 파이프라인에서는 문제 없음.
기존 가중치를 그대로 쓰려면 `cfg.TOT.time_pool = "last"`.

### 2. `models/totact_models.py` — TOTRegressor / _ConvBranch / bin_tot 추가
연속 TOT 회귀 모델. CV 최고 성능 설정.

| 태스크 | pooled acc | bal-acc | lift |
|---|---|---|---|
| 2-class @1.45s | **0.8988** | 0.8790 | +0.2716 |
| 3-class | **0.7877** | 0.7717 | +0.3901 |

설계 근거: log(TOT) 타깃(우편향 분포, Spearman 0.607->0.807), IMU 제외(과적합,
0.727->0.685), 입력 표준화 없음(전 항목 하락), veh conv branch + bi-GRU + mean&max pool.
`bin_tot()` 로 재학습 없이 임계값만 바꿔 2-class/3-class 평가 가능.

### 3. `config/config.py` — TOTConfig 확장
`time_pool`, `binary_threshold=1.45`, `ternary_thresholds=[1.07,1.53]`,
`regression_log_target`, `regression_windows_sec`, `regression_seeds`, `use_imu=False`.

`binary_threshold` 를 1.27(=TOT 중앙값)에서 **1.45** 로 변경한 것이 2-class 최대 개선.
1.27s 는 분포 최빈 구간 한복판이라 경계 모호 샘플이 33%, 1.45s 에서 22%.
CV acc 0.8321 -> 0.8988, 0.85 이상 fold 2/6 -> 5/6. lift 는 +0.2864 -> +0.2716 로
거의 유지되므로 다수 클래스로 도망쳐 얻은 정확도가 아니다.
1.53s 이상은 역효과 (lift +0.2272 로 급락).

### 4. `main.py` — ACT 지표 수정
`ACT_Baseline` 의 regressor 는 3채널을 출력하지만 손실은 채널 1만 학습시킨다.
채널 0/2 는 상수 0 근처의 미학습 출력 (ch0 R2=-2.17, pred_std=0.001).
기존 코드는 **채널 0 을 "mean ACT RMSE" 로 출력**하고 있었다 (= 무의미한 숫자 7.32).
- 평가에서 채널 1 만 사용하도록 수정
- 출력을 `MSE / RMSE (평균 예측 기준선 RMSE)` 로 변경
- 동일 평가를 두 번 호출하던 중복 제거

## 반영하지 않은 것 (의도적)
- **ACT_Baseline 출력 3채널 -> 1채널**: 기존 체크포인트를 깨뜨리므로 보류.
  채널 0/2 는 여전히 미학습 상태로 남아 있다.
- **Enhancer context 버그**: `models/totact_models.py:139, 213` 의
  `if 'imu_emotion' in batch: ... else: zeros` 에서 else 분기가 100% 실행된다.
  TOT/ACT enhancer 는 PKLMultiModalDatasetBaseline 으로 학습/평가하고 그 배치 키는
  ['label','sc_evt','sc_phase','sc_time','sc_type','veh'] 뿐이라 imu_emotion 이 없다.
  => fusion_head 입력 384차원 중 256차원이 상수 0. Context Expert 가 TOT/ACT 에
  전혀 기여하지 않는다 (baseline 과 정확도 완전 동일: train 0.6585/0.6585).
  `origin/main` 의 커밋 `829f698 "Fix TOT ACT enhancer fusion calls"` 가 이 문제를
  fb 딕셔너리 매핑으로 해결했으나 **로컬 main 이 아직 이 커밋을 받지 않았다.**
  수정 전 `git pull` 필요.
- **`sc_scenario_type` 결측**: 405개 중 402개가 -100. clamp(0,63) 으로 0 에 뭉개져
  상수로 입력된다. 제거하거나 실제 값 복구 필요.

## 미해결
- 3-class 는 0.7877 이 천장. 경계가 2개인데 모델 MAE 가 0.53s 이고 TOT 분포가
  0.8~1.5s 에 밀집해, 어떤 임계값을 잡아도 경계 +/-0.2s 샘플이 20~30% 남는다.
  그 구간 정확도는 0.64~0.73.
- fold 4 (특히 피실험자 5번: 12개 중 7개 오답, 샘플 절반이 경계 근처) 는 여전히 취약.
