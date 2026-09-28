# Dual-Encoder Sensor-Fusion LSTM-AE 결과 요약

## 한 줄 결론

이번 고정 조건의 단일 실행에서는 진동과 전류를 분리해 latent concat한 dual-encoder 모델이 기존 single-encoder LSTM-AE보다 **F1과 Recall에서 낮은 결과**를 보였습니다. FP는 줄었지만 FN이 늘었으므로, 이 결과만으로 sensor-fusion 구조의 이점을 확인할 수는 없습니다.

## 비교 조건

두 모델은 다음을 동일하게 유지했습니다.

- 입력 채널: `AI0_Vibration`, `AI1_Vibration`, `AI2_Current`
- 절대값 변환, 정상 train 15,000행 기반 MinMax scaling, sequence 20, label offset 100
- Train 14,880개 / Validation 1,180개 / Test 4,180개 window
- 정상 train 데이터만으로 autoencoder 학습, 마지막 timestep의 3채널 평균 reconstruction MSE score
- validation의 첫 `precision == recall` 지점으로 threshold 선정, strict `score > threshold` test 판정
- Trainable parameter 수: 두 모델 모두 63,171개

변경한 항목은 encoder 구조뿐입니다.

- Single-encoder: 3채널을 하나의 encoder로 처리
- Dual-encoder: 진동 2채널 `2 → 64 → 16`, 전류 1채널 `1 → 32 → 16`을 각각 인코딩한 뒤 `16 + 16` latent concat

## 구조 변경: 무엇이 달라졌나

### 기존 single-encoder LSTM-AE

세 센서를 처음부터 하나의 입력으로 처리합니다.

```text
AI0_Vibration ─┐
AI1_Vibration ─┼→ 단일 encoder LSTM (3 → 64 → 32)
AI2_Current   ─┘                 │
                                └→ 32차원 latent
                                     │ (sequence 길이만큼 반복)
                                     ▼
                           공통 decoder LSTM (32 → 32 → 64)
                                     │
                                     ▼
                             Linear (64 → 3)
                                     │
                                     ▼
                  AI0_Vibration, AI1_Vibration, AI2_Current reconstruction
```

### 이번 dual-encoder sensor-fusion LSTM-AE

진동과 전류만 encoder 단계에서 분리합니다. 두 latent를 단순 concatenate한 뒤에는 **하나의 공통 decoder와 하나의 Linear 출력층**이 세 원 신호를 함께 reconstruction합니다.

```text
AI0_Vibration ─┐
                ├→ 진동 encoder LSTM (2 → 64 → 16) ─┐
AI1_Vibration ─┘                                      │
                                                       ├→ concat → 32차원 fused latent
AI2_Current ─────→ 전류 encoder LSTM (1 → 32 → 16) ──┘             │
                                                                      │ (sequence 길이만큼 반복)
                                                                      ▼
                                                            공통 decoder LSTM (32 → 32 → 64)
                                                                      │
                                                                      ▼
                                                              Linear (64 → 3)
                                                                      │
                                                                      ▼
                                           AI0_Vibration, AI1_Vibration, AI2_Current reconstruction
```

따라서 이번 변경은 **각 센서별 decoder를 만든 것**이나 **Linear layer에서 fusion한 것**이 아닙니다. 변경점은 `단일 3채널 encoder`를 `진동 encoder + 전류 encoder + latent concat`으로 바꾼 부분뿐이며, decoder와 최종 Linear reconstruction 방식은 기존과 같은 계열로 유지했습니다.

## Test 결과

| 지표 | Single-encoder baseline | Dual-encoder fusion | 차이 (Dual − Baseline) |
| --- | ---: | ---: | ---: |
| Accuracy | 96.84% | 96.77% | -0.07%p |
| Precision | 58.96% | 58.96% | +0.01%p |
| Recall | 87.78% | 82.22% | -5.56%p |
| F1 | 70.54% | 68.68% | -1.86%p |
| FP | 110 | 103 | -7 |
| FN | 22 | 32 | +10 |
| Threshold | 약 0.003053 | 0.003528 | — |
| Trainable parameters | 63,171 | 63,171 | 0 |

| 모델 | Test confusion matrix `[[TN, FP], [FN, TP]]` |
| --- | --- |
| Single-encoder baseline | `[[3890, 110], [22, 158]]` |
| Dual-encoder fusion | `[[3897, 103], [32, 148]]` |

## 해석

- Dual-encoder는 정상 데이터를 이상으로 오판한 FP를 7개 줄였습니다.
- 반면 실제 이상을 정상으로 놓친 FN이 10개 늘었습니다.
- 이상 탐지에서 FN 감소가 중요한 목적이라면, 이번 결과는 dual-encoder가 baseline보다 불리했음을 보여 줍니다.
- Accuracy는 정상 test window가 많은 데이터 구성의 영향을 크게 받으므로, 이 실험에서는 F1, Recall, FP/FN을 함께 보는 것이 더 적절합니다.
- 파라미터 수를 맞춘 상태에서도 F1이 1.86%p 낮았으므로, **이번 조건에서는 encoder 분리와 단순 concat이 개선을 만들었다는 근거가 없습니다.**

## 이 결론의 범위와 한계

- 이 결과는 하나의 데이터셋, 고정 split, seed 42, 고정된 구조 크기, 단일 학습 실행에 한정됩니다.
- 통계적 유의성 검정이나 여러 seed 반복 실험은 수행하지 않았습니다. 따라서 성능 차이가 일반적으로 재현된다고 주장할 수 없습니다.
- Threshold, split, scaling, feature engineering은 의도적으로 변경하지 않았습니다. 이들 항목의 개선 가능성은 별도 실험 주제입니다.
- Baseline 수치는 `pytorch_reproduction.ipynb`에 저장된 실행 output(cell 21, 23)에서 확인했습니다. Dual-encoder 수치는 `results/dual_encoder_run01/metrics.json`에서 생성됐습니다. Baseline의 별도 `metrics.json` 파일은 현재 없으므로 notebook의 자동 비교 cell은 아직 실행되지 않았습니다.

## 생성 산출물

- Dual 지표: `results/dual_encoder_run01/metrics.json`
- 학습 이력: `results/dual_encoder_run01/history.csv`
- 학습 요약: `results/dual_encoder_run01/training.json`
- Loss / threshold / score distribution / confusion matrix 그림: `results/dual_encoder_run01/*.png`
