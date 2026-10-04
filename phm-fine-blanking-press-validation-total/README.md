# Fine-Blanking Press — Hypothesis Validation

KAMP 프레스 이상탐지의 세 가설(H1–H3)을 하나의 공개용 패키지로 정리한 저장소다.
핵심 보고서는 [notebooks/KAMP_가설검증_통합보고서.ipynb](notebooks/KAMP_가설검증_통합보고서.ipynb)다.

## 먼저 읽을 파일

- `SETUP_AND_EXPERIMENTS.md` — 실행 환경, 공통 프로토콜, 가설별 변경점과 해석 경계
- `notebooks/KAMP_가설검증_통합보고서.ipynb` — 질문 → 관찰 → 해석 → 한계를 연결한 최종 보고서
- `results/h*/REPORT.md` — 각 실험에서 생성된 원본 결과 요약

## 폴더 안내

```text
notebooks/  GitHub에서 바로 읽는 정적 통합 보고서
results/    보고서에 사용한 최종 설정·수치표·PNG (대형 중간 산출물과 모델은 제외)
src/        세 실험의 재실행 스크립트
```

## 빠른 재현

```powershell
conda env create -f environment.yml
conda activate kamp_repo

# H1: 입력 부호 ablation — 완료된 새 결과 폴더를 지정한다.
python src/run_ablation.py --run --data-dir .\data --output-dir .\work\h1
```

H2/H3는 H1에서 새로 생성된 signed checkpoint와 scaler를 입력으로 사용한다. 전체 명령과
입력·출력 관계는 `SETUP_AND_EXPERIMENTS.md`를 따른다. 제공된 `results/`는 보고서 검증용
경량 스냅샷이며 모델 가중치·원본 데이터는 포함하지 않는다.

## 공개 범위

원본 CSV, checkpoint, scaler, 재구성 배열, 대형 feature/score CSV는 커밋하지 않는다. 데이터
배포 조건을 확인한 뒤 별도 다운로드 안내 또는 GitHub Release/Git LFS로 제공한다.
