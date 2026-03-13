# 🃏 CardGuard AI — 실시간 카드게임 어뷰징 탐지 시스템

카지노 카드게임(Texas Hold'em 기준)에서 비정상적인 베팅 패턴을 실시간으로 잡아내는 AI 파이프라인입니다.

단순히 "모델 갖다 쓰는" 수준이 아니라 게임 시뮬레이터 → 로그 생성 → 피처 엔지니어링 → 탐지 모델 → 서빙까지 전 과정을 직접 설계했습니다.

---

## 왜 만들었나

온라인 카드게임에서 어뷰징 유저(콜루전, 봇 플레이, 칩 덤핑 등)를 잡는 건 생각보다 어렵습니다. 단순히 승률이 높다고 이상한 게 아니고, 베팅 타이밍이나 패턴의 *흐름*이 사람이랑 다릅니다.

그걸 시계열로 보자는 게 이 프로젝트의 출발점이었고, LSTM + Isolation Forest를 조합해서 둘 다 의심하는 케이스를 최종 플래그로 잡는 방식으로 구현했습니다.

---

## 시스템 구조

```
게임 시뮬레이터
    └── 플레이어 행동 로그 생성 (정상 / 어뷰저 / 봇 3종)
            └── 피처 엔지니어링
                    ├── Isolation Forest  →  이상치 스코어
                    └── LSTM Autoencoder  →  재구성 오차
                            └── 앙상블 스코어링
                                    └── 플래그 판정 + 리포트
```

---

## 탐지 대상 어뷰징 유형

| 유형 | 설명 | 탐지 방식 |
|------|------|-----------|
| 콜루전 | 여러 계정이 팀플레이 | 베팅 동조 패턴 |
| 칩 덤핑 | 의도적으로 칩 넘겨주기 | 폴드 타이밍 + 베팅 금액 |
| AI 봇 | 사람이 아닌 프로그램 | 반응 시간 분포, 패턴 반복성 |
| 과도한 올인 | 확률 무시한 무리한 베팅 | 핸드 강도 대비 베팅 비율 |

---

## 실험 결과

| 모델 | Precision | Recall | F1 |
|------|-----------|--------|----|
| Isolation Forest only | 0.71 | 0.68 | 0.69 |
| LSTM Autoencoder only | 0.76 | 0.74 | 0.75 |
| **앙상블 (최종)** | **0.84** | **0.81** | **0.82** |

> 데이터: 시뮬레이터로 생성한 10만 게임 세션 (정상 80%, 어뷰저 20%)

---

## 프로젝트 구조

```
gamespring-ai/
├── src/
│   ├── simulator/
│   │   ├── poker_game.py       # 텍사스 홀덤 게임 엔진
│   │   ├── player.py           # 플레이어 행동 모델 (정상/어뷰저/봇)
│   │   └── log_generator.py    # 게임 로그 생성기
│   ├── detection/
│   │   ├── feature_engineering.py   # 피처 추출
│   │   ├── isolation_forest.py      # IF 기반 탐지
│   │   ├── lstm_model.py            # LSTM Autoencoder
│   │   └── detector.py             # 앙상블 탐지기
│   └── utils/
│       └── data_utils.py
├── scripts/
│   ├── generate_data.py        # 데이터 생성 실행
│   └── train_model.py          # 모델 학습 실행
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
└── tests/
```

---

## 빠른 시작

```bash
git clone https://github.com/yourname/gamespring-ai.git
cd gamespring-ai
pip install -r requirements.txt

# 시뮬레이션 데이터 생성
python scripts/generate_data.py --sessions 10000 --output data/raw/

# 모델 학습
python scripts/train_model.py --data data/raw/ --output data/processed/

# 탐지 실행
python scripts/run_detector.py --input data/raw/sample_session.json
```

---

## 주요 설계 결정

**왜 LSTM + Isolation Forest 조합인가**

LSTM 혼자 쓰면 학습 데이터에 없는 새로운 어뷰징 패턴을 못 잡습니다. IF는 비지도라서 그 부분을 커버해주는데, 반면 IF는 시계열 맥락을 모릅니다. 각자 잘하는 걸 조합하는 게 맞다고 판단했습니다.

**왜 시뮬레이터를 직접 만들었나**

실제 게임 로그가 없으니까요. 근데 그냥 랜덤 숫자 생성하면 의미가 없어서 실제 포커 룰대로 핸드 강도 계산, 팟 오즈 계산까지 구현했습니다. 덕분에 어뷰저 행동을 훨씬 현실적으로 모사할 수 있었습니다.

---

## 앞으로 할 것들

- [ ] 실시간 스트리밍 처리 (Kafka 연동)
- [ ] 콜루전 탐지 강화 (그래프 뉴럴네트워크)
- [ ] 대시보드 UI (Grafana or 자체 React 페이지)
- [ ] AWS Lambda 배포 테스트

---

## 개발 환경

- Python 3.10
- PyTorch 2.1
- scikit-learn 1.3
- pandas, numpy

---

*질문이나 피드백은 이슈로 남겨주세요.*
