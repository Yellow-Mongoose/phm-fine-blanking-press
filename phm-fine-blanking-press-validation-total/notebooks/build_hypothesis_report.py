"""Build the GitHub-facing, static KAMP hypothesis-validation report.

The report deliberately contains no model fitting or score calculation.  It is a
curated reading layer over the immutable artifacts produced by the three
experiments.  Re-run this script only when the selected result folders change.
"""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "KAMP_가설검증_통합보고서.ipynb"

# These paths are intentionally relative to the notebook so GitHub can render
# the figures after the four project directories are placed in one repository.
H1 = "../results/h1_signed_input"
H2 = "../results/h2_p2p_fusion"
H3 = "../results/h3_psd_periodicity"


def md(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip(),
    }


def image(path: str, alt: str, caption: str):
    return md(f"![{alt}]({path})\n\n*그림. {caption}*")


cells = [
    md(
        """
        # KAMP 프레스 이상탐지 — 가설 검증 통합 보고서

        ## 한 문장 결론

        **부호를 보존한 LSTM-AE 입력은 절댓값 입력보다 더 나은 창 단위 F1을 보였고(H1),
        진동 P2P는 이 고정 프로토콜에서 유용한 보조 판별 신호였으며(H2), 전류 PSD·주기성은
        단독 신호로는 강한 재현율을 보였지만 현 정규화 규칙에서는 AE와의 fusion을 정당하게 계산할 수 없었다(H3).**

        이 문서는 세 실험이 남긴 완료된 산출물만 읽는 **정적·재현 가능 보고서**다. 학습이나
        임계값 재선택을 실행하지 않는다. 숫자·그래프의 원본은 각 실험 폴더의 `REPORT.md`,
        `config.json`, `preflight.json`, CSV 및 `figures/`에 보존한다.
        """
    ),
    md(
        """
        ## 읽는 순서: 관찰에서 다음 가설로

        ```text
        원 신호
          └─ H1. abs()를 제거해 부호를 보존하면 재구성 기반 판별이 좋아지는가?
                └─ H2. H1의 signed AE가 놓치는 형태를 진동 P2P가 보완하는가?
                      └─ H3. 전류의 주파수·주기성 정보도 독립적인 보완 신호인가?
        ```

        각 장은 같은 순서로 읽는다.

        1. **질문** — 무엇을 바꾸거나 추가했는가
        2. **관찰** — 완료된 결과에서 직접 확인되는 수치와 그림
        3. **해석** — 그 관찰이 뜻하는 범위
        4. **판정과 한계** — 주장 가능한 수준과 주장하면 안 되는 것
        5. **연결** — 왜 다음 가설이 필요한가
        """
    ),
    md(
        """
        ## 0. 비교 전에 알아둘 평가 경계

        | 항목 | H1: abs vs signed | H2: P2P fusion | H3: PSD·주기성 |
        |---|---:|---:|---:|
        | 공통 기반 | W20, label offset 100, 정상 Train 15,000행 | H1 signed checkpoint를 고정 | H1 signed checkpoint·scaler를 고정 |
        | 검증 구성 | 정상 880 / 이상 300 윈도 | 정상 880 / 이상 300 윈도 | 정상 880 / 이상 300의 연속성 적격 윈도 |
        | 임계값 정책 | 원 프로토콜의 `precision == recall` 규칙 | 정상 Validation FPR ≤ 1% | 정상 Validation FPR ≤ 1% |
        | Test 모집단 | 4,180 창 | 4,180 창 | 2,094 창 (150 ms 이하 간격 적격 창) |

        **따라서 H1의 F1과 H2/H3의 F1을 한 줄의 리더보드로 순위화하지 않는다.** H1은
        입력 부호의 ablation이고, H2/H3는 서로 다른 점수·임계값 규칙 및 (H3의 경우) 다른
        적격 모집단에서 수행한 후속 검증이다. 직접 비교는 각 장에서 정의한 같은 모집단·같은
        정책 안에서만 한다.
        """
    ),
    md(
        """
        ## 1. H1 — 부호를 버리면 정보도 버리는가?

        ### 질문

        기존 입력의 `abs()`만 제거하고 signed 파형을 그대로 LSTM-AE에 넣었을 때, 동일한
        창·라벨·초기화·학습 제어 아래에서 판별 결과가 달라지는지 확인한다. 각 변형은 자신의
        정상 Train MinMaxScaler를 따로 적합했으므로 **재구성 MSE의 절대 크기끼리**가 아니라
        validation 전용 임계값 후의 예측·성능을 비교한다.
        """
    ),
    image(f"{H1}/figures/01_signed_waveforms_and_distributions.png", "Signed raw waveforms", "부호가 있는 원 진동 파형과 분포. H1에서 보존하려는 정보의 형태를 보여준다."),
    image(f"{H1}/figures/04_training_curves.png", "H1 training curves", "두 입력 변형의 학습 진행. 동일한 학습 통제 아래의 완료 기록이다."),
    md(
        """
        ### 관찰: signed 입력은 F1을 높이고 오탐을 줄였다

        | 입력 | Precision | Recall | F1 | FPR | FP | FN |
        |---|---:|---:|---:|---:|---:|---:|
        | absolute | 0.5778 | 0.8667 | 0.6933 | 0.0285 | 114 | 24 |
        | signed | 0.6977 | 0.8333 | **0.7595** | **0.0163** | **65** | 30 |
        | signed − absolute | +0.1199 | −0.0333 | **+0.0662** | −0.0123 | −49 | +6 |

        Signed는 4,180개 공통 Test 창에서 49개의 오탐을 제거했지만, 미탐은 6개 늘었다.
        즉 이 결과의 핵심은 “모든 이상을 더 많이 잡았다”가 아니라, **오탐을 크게 낮추면서
        전체 F1을 개선했다**는 균형 변화다.
        """
    ),
    image(f"{H1}/figures/06_test_scores_and_confusion_matrices.png", "H1 test scores and confusion matrices", "H1의 공통 Test 창에서 임계값·점수 분포·혼동행렬을 함께 확인한다."),
    image(f"{H1}/figures/08_prediction_changes.png", "H1 prediction changes", "입력 변형 때문에 실제 판정이 바뀐 창의 수. 성능 차이를 예측 수준에서 확인한다."),
    md(
        """
        ### H1 판정과 연결

        **판정: 고정된 단일 seed 프로토콜에서 H1은 부분적으로 지지된다.** signed 입력은 F1과
        precision을 개선했으나 recall은 소폭 낮아졌다. 이는 신호의 방향 정보가 재구성 기반
        이상 점수에 유용할 수 있음을 시사하지만, 통계적 유의성이나 다른 설비로의 일반화를
        증명하지는 않는다.

        H1은 signed AE를 후속 실험의 고정 기준 모델로 삼을 근거를 만든다. 그러나 AE만으로
        남는 FN이 있으므로, 다음 단계에서는 파형의 **진폭 변화량(P2P)** 이 그 빈틈을 보완할
        수 있는지 묻는다.
        """
    ),
    md(
        """
        ## 2. H2 — 재구성 오차와 진동 폭은 서로 보완적인가?

        ### 질문

        H1에서 저장한 signed LSTM-AE checkpoint는 바꾸지 않는다. raw signed `AI0_Vibration`,
        `AI1_Vibration`의 W20 P2P를 정상 Train Q95·IQR 기준으로 점수화하고, AE 점수와
        `normalized_AE + α × normalized_P2P`로 결합한다. α는 `[0, 0.5, 1.0, 2.0]`에서
        Validation F1만으로 고른다.

        여기서 중요한 설계 원칙은 Test로 α·임계값·기준값을 고르지 않았다는 점이다.
        """
    ),
    image(f"{H2}/figures/p2p_distributions.png", "P2P distributions", "정상 Train 기준 상한과 상태별 P2P 분포. P2P가 어떤 변화를 점수화하는지 보여준다."),
    image(f"{H2}/figures/fusion_weight_validation.png", "H2 alpha selection", "α는 Test가 아니라 Validation에서 선택됐다."),
    md(
        """
        ### 관찰: P2P를 포함한 방법이 이 정책에서 더 높은 F1을 기록했다

        | 방법 | α | Precision | Recall | F1 | FPR | FP | FN |
        |---|---:|---:|---:|---:|---:|---:|---:|
        | A: AE | — | 0.8750 | 0.5444 | 0.6712 | 0.0035 | 14 | 82 |
        | B: P2P | — | 1.0000 | 0.6944 | **0.8197** | 0.0000 | 0 | 55 |
        | C: AE + P2P | 2.0 | 1.0000 | 0.6944 | **0.8197** | 0.0000 | 0 | 55 |

        A 대비 C의 변화는 precision +0.1250, recall +0.1500, F1 +0.1484, FP −14, FN −27이다.
        모든 값은 같은 4,180 Test 창 및 FPR 목표 임계값 정책 안에서의 비교다.
        """
    ),
    image(f"{H2}/figures/test_metric_comparison.png", "H2 test metric comparison", "동일 H2 프로토콜 안에서 AE, P2P, fusion의 성능을 비교한다."),
    image(f"{H2}/figures/prediction_change_counts.png", "H2 prediction changes", "방법 간 실제 예측 전이의 수. 상관계수만으로 보완성을 주장하지 않기 위한 증거다."),
    md(
        """
        ### 해석의 경계: “fusion 승리”가 아니라 “P2P 지배”

        **판정: H2는 이 고정 checkpoint·프로토콜에서는 지지된다.** 하지만 C가 B와 정확히 같은
        결과였으므로, 이는 부드러운 AE–P2P 상보 결합의 증거가 아니다. 정상 Train P2P 점수는
        대부분 0이어서 IQR도 0이었고, 사전에 정한 `1e-6` floor가 비영(非零) P2P 초과를 크게
        만들었다. 따라서 α>0의 결합은 사실상 P2P 순위에 지배됐다.

        이 결과는 “진폭 폭”이 유용한 이상 단서일 수 있음을 보여준다. 다음 질문은 다른 종류의
        단서, 즉 **전류 파형의 주파수 분포·주기성**이 유효한지를 검증하는 것이다.
        """
    ),
    md(
        """
        ## 3. H3 — 주파수·주기성은 유용하지만, 결합 가능한 점수인가?

        ### 질문

        H1 signed checkpoint와 scaler를 다시 학습하지 않고 고정한 채, 전류 신호에서 PSD 기반
        spectral entropy와 autocorrelation(선택 lag 1) 점수를 계산한다. 시간 간격이 150 ms를
        넘는 창은 제외하여, W20 내부의 19개 간격 모두 연속적인 창만 A/B/C/D의 공통 대상으로
        쓴다. 이 필터 때문에 H3의 Test 모집단은 2,094개이며 앞 장과 직접 순위 비교할 수 없다.
        """
    ),
    image(f"{H3}/figures/representative_normal_psd.png", "Representative normal PSD", "정상 창의 대표 PSD. PSD 기반 비교가 보는 신호 표현이다."),
    image(f"{H3}/figures/representative_anomaly_psd.png", "Representative anomaly PSD", "이상 창의 대표 PSD. 파형의 시간 영역 변화와 분리해 주파수 분포를 본다."),
    md(
        """
        ### 관찰: PSD 단독은 재현율 1.0, AE는 높은 균형 성능

        | 방법 | 사용 가능 | Precision | Recall | F1 | FPR | FP | FN | PR-AUC |
        |---|---|---:|---:|---:|---:|---:|---:|---:|
        | A: AE | 예 | 0.8925 | 0.9326 | **0.9121** | 0.0050 | 10 | 6 | 0.9700 |
        | B: PSD | 예 | 0.5329 | **1.0000** | 0.6953 | 0.0389 | 78 | 0 | **1.0000** |
        | C: AE + PSD | 아니오 | — | — | — | — | — | — | — |
        | D: AE + ACF | 아니오 | — | — | — | — | — | — | — |

        PSD는 모든 이상 창을 포착했지만 정상 창 78개도 이상으로 분류했다. AE는 더 적은 FP와
        FN의 균형으로 높은 F1을 기록했다. 이는 “PSD가 무용하다”가 아니라, **이 임계값 정책에서
        PSD는 민감하지만 특이도가 낮은 단독 경보**였다는 뜻이다.
        """
    ),
    image(f"{H3}/figures/ae_vs_psd_score.png", "AE versus PSD scores", "AE·PSD 점수 관계를 시각화한다. 점수 상관만으로 fusion 가능성을 결론내리지 않는다."),
    image(f"{H3}/figures/distribution_psd_score.png", "PSD score distribution", "PSD 점수의 상태별 분포와 임계값을 확인한다."),
    md(
        """
        ### 중요한 음성 결과: 결합 점수를 만들지 않았다

        C와 D는 실패한 계산 결과가 아니라 **의도적으로 unavailable 처리한 결과**다. one-sided
        PSD/ACF 점수의 normal-train IQR이 0 또는 너무 작아 `(score − median) / IQR` 정규화를
        안전하게 수행할 수 없었다. 임의의 epsilon이나 다른 정규화를 사후에 넣지 않았기 때문에,
        보기에 좋은 fusion 수치를 만들어 내는 대신 재현 규칙을 훼손하지 않았다.

        **판정: PSD/ACF feature의 fusion 가설은 현 규칙에서 미판정이다.** PSD 단독의 관찰은
        보존하되, fusion의 성능 주장은 하지 않는다.
        """
    ),
    md(
        """
        ## 4. 종합 결론 — 무엇을 채택하고, 무엇을 보류하는가?

        | 주장 | 근거 | 현재 결론 |
        |---|---|---|
        | 부호 보존은 유용하다 | H1: 공통 Test에서 F1 +0.0662, FP −49 | **채택 후보** — signed LSTM-AE를 기준선으로 사용 |
        | P2P는 보조 단서다 | H2: FPR 목표 정책에서 F1 +0.1484, FN −27 | **유망** — 다만 fusion은 P2P 지배 현상으로 해석 |
        | PSD·ACF를 AE와 결합할 수 있다 | H3: normal-train IQR=0으로 C/D 불가 | **보류** — 안정적 점수화 규칙이 먼저 필요 |

        ### 이 보고서가 주장하지 않는 것

        - 단일 seed 결과가 통계적으로 유의하거나 다른 설비·기간에 일반화된다는 주장
        - 서로 다른 임계값 정책 또는 서로 다른 H3 적격 모집단의 F1을 순위화하는 주장
        - reconstruction error, P2P, PSD 점수만으로 특정 물리 고장 원인을 확정하는 주장
        - Test 결과를 보고 추가 조정한 뒤 같은 Test를 최종 검증으로 쓰는 주장
        """
    ),
    md(
        """
        ## 5. 다음 실험 설계

        1. **독립 split 또는 시간 블록 holdout**에서 H1/H2를 재검증하고, 여러 seed의 평균·신뢰구간을 보고한다.
        2. **P2P 점수의 상수성**을 해결할 수 있는 사전 정의 정규화(예: quantile transform,
           bounded rank score)를 별도 가설로 등록한다. H2의 floor는 그대로 유지한 비교군도 둔다.
        3. **PSD/ACF fusion**은 zero-IQR 상황을 처리하는 규칙을 Test를 보지 않고 확정한 뒤,
           새 holdout에서 다시 평가한다.
        4. 창 단위 판정뿐 아니라 **구간/이벤트 단위**의 조기경보 성능과 false alarm burden을 보고한다.

        ---

        ## 재현 및 원본 산출물

        | 가설 | 선택한 완료 산출물 | 핵심 원본 |
        |---|---|---|
        | H1 | `phm-fine-blanking-press-val/results/gpu_ablation_20261004` | `REPORT.md`, `ablation_comparison.csv`, `prediction_changes.csv` |
        | H2 | `phm-fine-blanking-press-val2/results/h2_snapshot_20261004` | `REPORT.md`, `selection.json`, `prediction_changes.csv` |
        | H3 | `phm-fine-blanking-press-val3/results/gpu_psd_analysis_20261004_final3` | `REPORT.md`, `result.json`, `config.json` |

        GitHub 공개본에는 이 노트북, 위의 작은 설정·요약 파일, 보고서에 쓰인 PNG와 재실행 코드를
        포함한다. 원본 CSV, 대형 중간 feature/score CSV, 모델 checkpoint·scaler는 `.gitignore`로
        제외하거나 필요할 때만 릴리스 자산/Git LFS로 분리한다.
        """
    ),
]

nb = {
    "cells": cells,
    "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "report": {
        "purpose": "Static, GitHub-facing synthesis of completed H1/H2/H3 artifacts.",
        "selected_runs": {"h1": H1, "h2": H2, "h3": H3},
    },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUT)
