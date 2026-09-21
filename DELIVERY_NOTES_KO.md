# MUST 전달 패키지 — 2026-09-20

현재 작업 폴더의 수정 사항을 포함한 소스 스냅샷입니다. 이번 패키징에서 모델 동작을 추가로 수정하지 않았습니다.
기존 폴더에 덮어쓰기보다 새 폴더에 압축을 풀어 사용해 주세요.

## 빠른 실행 확인

Python 환경에서 패키지 폴더로 이동한 뒤 실행합니다.

```bash
pip install -r requirements.txt
python main.py --profile sample --smoke
```

이 명령은 합성 데이터 로딩과 배치 차원을 검사합니다. 모델 전체 학습·추론이나 성능 검증은 수행하지 않습니다.
실제 데이터가 없는 이 패키지에서 `python main.py`만 실행해도 sample 검사로 전환됩니다.

## 포함 / 제외

- 포함: main.py, config, models, trainers(기존 fusion 버전 포함), data의 로더 코드, scripts, requirements.txt, 합성 샘플, 변경 내역.
- 제외: 실제 데이터, 학습 가중치, archive, logs, results, Git 이력, 캐시, 기존 ZIP.
- `data/sample/sample_train.pkl`, `sample_survey.csv`는 실제 피험자 데이터가 아닌 합성 샘플입니다.
- 기존 실험용 모델·트레이너도 비교를 위해 그대로 포함했습니다. 모든 개별 버전의 독립 실행을 보장하는 패키지는 아닙니다.

## 반영된 변경

- 저장소 기준 경로 설정과 sample/full 프로필, 합성 샘플 실행 지원.
- 현재 작업 폴더의 main/config, emotion/motion trainer, fusion v28/v28_assym 수정본 포함.
- TOT baseline 시간축 pooling, TOT 회귀 모델·관련 설정, ACT 평가 채널 수정 등은 `CHANGELOG_20260825_tot.md` 참조.
- 위 변경 내역의 logs/archive 경로는 작성 당시 로컬 기록이며 이 ZIP에는 포함하지 않았습니다.
- TOT/ACT enhancer의 context 전달 문제 등 기존 변경 내역에 기록된 미해결 사항도 남아 있습니다. 이 패키지는 전체 문제 해결본이 아닙니다.

## presurvey 확인 사항 — 미해결

1. 현재 survey encoder 입력 설정은 31차원입니다. 포함된 31차원 CSV는 합성 샘플이며, 실제 설문의 31차원 항목 구성·전처리 정의는 확인되지 않았습니다. 로컬에서 확인된 실제 pre_survey.csv는 ID 제외 12차원이었으며 이 ZIP에는 포함하지 않았습니다.
2. full 프로필도 별도 경로 설정이 없으면 합성 survey CSV를 사용합니다. CSV에 없는 피험자는 0 벡터로 대체되고, CSV 로딩 실패 시에는 설정 차원의 0 벡터로 대체됩니다. 실제 설문이 없어 실행 가능하다는 뜻이며 실제 설문을 정상 활용한다는 뜻은 아닙니다. 12차원 CSV를 그대로 연결하면 차원 검증에 실패합니다.
3. 기본 설정(감정: ppg/sc/survey, 모션: imu/veh)의 fusion v28, v28_assym, v28_bi, v29, v30, sample은 survey를 실제 예측 경로에서 사용하지 않습니다. main.py의 현재 진입점은 v28_assym입니다. 2026-09-18 합성 배치와 초기화된 모델의 eval forward에서 survey 호출 0회, 설문값 변경·삭제 시 출력 차이 0을 확인했습니다. 학습 가중치 기반 성능 평가는 아닙니다.
4. EmotionTrainer의 감정 사전학습·단독 추론 경로는 survey를 context로 사용합니다. 다른 과거 fusion 구현까지 모두 미사용인 것은 아닙니다.
5. 위 presurvey 문제를 이번 패키징에서 수정하지 않았습니다.

## 실제 데이터로 전체 실행

합성 샘플 검사를 넘어서 전체 파이프라인을 실행하려면 실제 PKL과 호환되는 pretrained emotion/motion 가중치를 별도로 준비해야 합니다. main.py는 해당 가중치를 로드하며, 이 패키지의 기본 실행에서 두 사전학습을 새로 수행하지 않습니다. 실제 survey를 연결하려면 31차원 입력 정의도 확인해야 합니다.

```bash
MUST_PKL_ALL=/path/to/train_ver2.pkl \
MUST_SURVEY_CSV=/path/to/survey_31_features.csv \
MUST_EMOTION_CKPT=/path/to/best_emotion.pt \
MUST_MOTION_CKPT=/path/to/best_motion.pt \
python main.py --profile full
```

가중치는 모델 구성과 호환되어야 합니다. TOT pooling 설정 변경과 기존 가중치의 호환성은 변경 내역을 확인해 주세요.

## 소스 식별

기준 로컬 Git HEAD: `c0d06a0fd997ab79107ba1fd46560a2ec0b2c7df`. 커밋 이후 작업 폴더 수정도 포함했습니다. 원격 최신 버전과 동기화한 릴리스라는 의미는 아닙니다.
