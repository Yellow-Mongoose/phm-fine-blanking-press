"""Create a reproducible Markdown report, CSV tables, figures and compact ZIP."""
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics

OUT = Path(__file__).resolve().parents[1] / 'results/normalization/report'
NAMES = {'N01':'정상 기준 z-score', 'N02':'정상 기준 min-max',
         'N03':'구간 평균 제거 + 정상 기준 z-score', 'N04':'구간별 z-score'}


def table(rows, columns):
    def value(x):
        return f'{x:.4f}' if isinstance(x, (float, np.floating)) else str(x)
    return '\n'.join(['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(value(row[c]) for c in columns)+' |' for row in rows])


def summarize(runs):
    OUT.mkdir(parents=True, exist_ok=True)
    rows=[]; checks=[]; summaries=[]; curves={}
    reference=None
    for case in sorted(runs):
        run=Path(runs[case]); cfg=json.loads((run/'config/experiment.json').read_text())
        selected=json.loads((run/'selection/primary.json').read_text())
        history=pd.read_csv(run/'history.csv')
        if reference is None: reference=cfg
        for key in ('split_hash','source_hashes','train','seed'):
            assert cfg[key]==reference[key], (case,key)
        assert np.isfinite(history[['train_loss','val_normal_loss']].to_numpy()).all()
        assert cfg['spec']['family']=='baseline' and cfg['spec']['backbone']=='lstm'
        best=int(history.loc[history.val_normal_loss.idxmin(),'epoch'])
        assert selected['epoch']==best, (case,'selected epoch')
        summaries.append(dict(experiment=case,method=NAMES[case],epochs=len(history),
            selected_epoch=selected['epoch'],stop_reason=selected['stop_reason'],
            parameter_count=selected['parameter_count'],run_dir=str(run)))
        curves[case]=history
        for split,folder in [('Valid','valid/final_evaluation'),('Test','test/primary')]:
            metrics=json.loads((run/folder/'metrics.json').read_text())
            for policy in ('normal_q99','valid_max_f1'):
                m=metrics['label_10s__'+policy]
                assert m['TN']+m['FP']+m['FN']+m['TP']==m['n']
                assert np.isclose(m['FPR'],m['FP']/m['normal'])
                rows.append(dict(experiment=case,method=NAMES[case],split=split,policy=policy,
                    selected_epoch=selected['epoch'],threshold=selected['policies'][policy]['threshold'],**m))
            # Current/future label equality limits forecasting interpretation.
            scores=pd.read_csv(run/folder/'scores.csv')
            # Inference scores are float32. Reading CSV as float64 changes
            # equality at a threshold by rounding; restore inference precision.
            score_values=scores.score.to_numpy(dtype=np.float32)
            labels=scores.label_10s.to_numpy()
            for policy in ('normal_q99','valid_max_f1'):
                rule=selected['policies'][policy]
                pred=(score_values>=rule['threshold'] if rule['operator']=='>='
                      else score_values>rule['threshold'])
                expected=skmetrics.confusion_matrix(labels,pred,labels=[0,1]).ravel().tolist()
                stored=metrics['label_10s__'+policy]
                assert expected==[stored[k] for k in ('TN','FP','FN','TP')]
                assert np.array_equal(pred,scores[policy+'_prediction'].to_numpy())
                assert np.isclose(skmetrics.average_precision_score(labels,score_values),stored['AP'])
                assert np.isclose(skmetrics.roc_auc_score(labels,score_values),stored['ROC_AUC'])
            checks.append(dict(experiment=case,split=split,windows=len(scores),
                labels_equal=bool((scores.label_current==scores.label_10s).all()),
                scores_finite=bool(np.isfinite(scores.score).all())))
            assert np.isfinite(scores.score).all()
            pd.read_csv(run/folder/'group_metrics.csv').to_csv(OUT/f'{case}_{split.lower()}_groups.csv',index=False)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'metrics_comparison.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'training_summary.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'verification.csv',index=False)
    primary=frame.query("policy == 'normal_q99'")
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,metric in zip(axes,['AP','recall','FPR']):
        for offset,split in [(-.18,'Valid'),(.18,'Test')]:
            values=primary.query('split == @split').set_index('experiment').loc[sorted(runs),metric]
            ax.bar(np.arange(4)+offset,values,width=.36,label=split)
        ax.set(xticks=np.arange(4),xticklabels=sorted(runs),ylabel=metric,title=metric)
        ax.set_ylim(0,1 if metric!='FPR' else max(.03,float(primary.FPR.max())*1.2))
        ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'metric_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for case,h in curves.items():
        axes[0].plot(h.epoch,h['label_10s__normal_q99__AP'],label=case)
        axes[1].plot(h.epoch,h['label_10s__normal_q99__F1'],label=case)
    for ax,title in zip(axes,['Valid AP by epoch','Valid F1 by epoch (normal q99)']):
        ax.set(xlabel='Epoch',ylabel=title,title=title,ylim=(0,1));ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'validation_curves.png',dpi=160);plt.close(fig)
    valid=primary.query("split == 'Valid'").set_index('experiment')
    test=primary.query("split == 'Test'").set_index('experiment')
    best_valid=str(valid.AP.idxmax())
    diff=[]
    for case in ('N02','N03','N04'):
        diff.append(dict(experiment=case,Valid_AP_delta=float(valid.loc[case,'AP']-valid.loc['N01','AP']),
            Test_AP_delta=float(test.loc[case,'AP']-test.loc['N01','AP']),
            Test_recall_delta=float(test.loc[case,'recall']-test.loc['N01','recall']),
            Test_FPR_delta=float(test.loc[case,'FPR']-test.loc['N01','FPR'])))
    pd.DataFrame(diff).to_csv(OUT/'deltas_vs_N01.csv',index=False)
    columns=['experiment','AP','ROC_AUC','precision','recall','F1','FPR','FP','FN']
    text=f'''# 정규화·상대신호 실험 결과 보고서

## 결과 개요

N01·N03·N02·N04의 본학습과 최종 Test 평가를 완료했다. 정상 Valid 복원 손실이 가장 작은 epoch를 각 실험의 대표 가중치로 선택했다. 대표 판정은 정상 Valid 점수의 99% 분위수(normal_q99)이며, 모든 가중치와 임계값을 고정한 뒤 Test를 평가했다.

최종 선택 가중치의 **Valid AP**가 가장 높은 실험은 **{best_valid} ({NAMES[best_valid]})**, AP={valid.loc[best_valid,'AP']:.4f}였다. 이는 단일 seed와 현재 수집 구간에서의 비교 결과이다. Test 성능을 기준으로 설정을 추가 조정하지 않았다.

## 데이터와 공통 조건

- 원본 정상 20,000행(2022-07-12), 이상 600행(2022-07-17). 진동 2채널·전류 1채널.
- 기존 중복 제거·공백 분리·구간별 분할·20행 윈도우와 10초 뒤 라벨 프로토콜을 유지했다.
- seed={reference['seed']}, CPU float32, PyTorch 계산 스레드 2개, baseline LSTM 구조 동일.
- 학습 설정: `{json.dumps(reference['train'],ensure_ascii=False)}`.
- 최대 800 epoch, 최소 200 epoch 이후 정상 Valid loss의 150 epoch 무개선 시 조기 종료. 실험마다 종료 epoch가 달라도 같은 규칙을 적용했다.
- 동일 CSV 해시·분할 해시·seed·학습 설정, 유한 loss/score, 대표 epoch와 혼동행렬 합계를 검증했다. 저장된 점수를 원래 float32 정밀도로 읽어 두 판정 정책의 예측·혼동행렬·AP·ROC-AUC도 다시 계산해 저장 지표와 일치함을 확인했다.
- 분할 해시: `{reference['split_hash']}`.

{table(checks,['experiment','split','windows','labels_equal','scores_finite'])}

## 변환 방식

| 실험 | 파형 입력 | 의도 |
|---|---|---|
| N01 | 정상 Train 평균·표준편차로 z-score | 기준선·진폭 차이 유지, 센서 규모 조정 |
| N02 | 정상 Train 최소·최대로 min-max, clipping 없음 | N01과 스케일링 기준 비교 |
| N03 | 각 구간 평균 제거 후 정상 Train 상대신호 기준 z-score | 기준선 영향을 줄이고 진폭 차이 보존 |
| N04 | 각 구간 평균·표준편차로 z-score | 기준선·진폭 영향을 함께 줄이고 파형 비교 |

N03의 scaler는 겹치는 정상 Train 윈도우에서 fit하며 N01/N02는 원본 정상 Train 행에서 fit한다. 따라서 N01↔N03에는 중심화뿐 아니라 표본 반복 가중 방식의 차이도 있다. N04는 추가 전역 scaler가 없다. 모든 구간 통계는 완성된 20행 안에서 계산하며 판정 시점은 구간 끝이다. 20행이 완전한 기계 주기를 담는다는 보장은 없다.

## 학습 완료 내역

{table(summaries,['experiment','epochs','selected_epoch','stop_reason','parameter_count'])}

손실의 단위와 센서별 가중은 정규화에 따라 달라지므로 서로 다른 실험의 loss 절대값으로 순위를 정하지 않는다.

## Valid 결과 — 대표 가중치, normal_q99

{table(primary.query("split == 'Valid'").to_dict('records'),columns)}

## Test 결과 — 고정된 가중치와 임계값, normal_q99

{table(primary.query("split == 'Test'").to_dict('records'),columns)}

## 관측된 효과와 적용 권고

1. **현재 기준 실험은 N01을 유지하는 것이 타당하다.** 대표 가중치의 Valid AP는 N01={valid.loc['N01','AP']:.4f}, N02={valid.loc['N02','AP']:.4f}, N03={valid.loc['N03','AP']:.4f}, N04={valid.loc['N04','AP']:.4f}였다. 단일 seed 결과이므로 최적 방식이 확정됐다는 뜻은 아니다.
2. **min-max도 Test에서 높은 성능을 보였다.** N02의 Test AP는 {test.loc['N02','AP']:.4f}, recall은 {test.loc['N02','recall']:.4f}로 N01의 {test.loc['N01','AP']:.4f}, {test.loc['N01','recall']:.4f}와 비슷했다. N01은 FP={int(test.loc['N01','FP'])}, FN={int(test.loc['N01','FN'])}, N02는 FP={int(test.loc['N02','FP'])}, FN={int(test.loc['N02','FN'])}였다. z-score가 항상 min-max보다 우수하다고 말할 근거는 없다. Test 차이를 근거로 실험 설정을 변경하지 않았다.
3. **평균 제거는 오탐과 미탐의 교환을 만들었다.** N03은 Test FP={int(test.loc['N03','FP'])}로 N01의 {int(test.loc['N01','FP'])}보다 적었지만 FN={int(test.loc['N03','FN'])}로 N01의 {int(test.loc['N01','FN'])}보다 많았다. 이상 recall은 {test.loc['N03','recall']*100:.2f}%로 N01의 {test.loc['N01','recall']*100:.2f}%보다 낮았다. 이 결과는 평균 제거를 무조건 적용하는 것을 지지하지 않는다. 이상 구분에 유용한 평균 정보 또는 수집 조건 정보가 제거됐을 가능성이 있으며 현재 데이터로 두 원인을 분리할 수 없다.
4. **구간별 z-score만 사용하는 것은 현재 기본 입력으로 권하지 않는다.** N04의 Test AP={test.loc['N04','AP']:.4f}, recall={test.loc['N04','recall']*100:.2f}%, FP={int(test.loc['N04','FP'])}, FN={int(test.loc['N04','FN'])}였다. 진폭 정보 손실과 구간별 통계 변화가 성능 저하에 기여했을 가능성과 일치하지만, 원인을 단독으로 입증한 실험은 아니다. 상대 파형을 사용하려면 후속 N05/N06에서 AC RMS·P2P 등 크기 특징을 함께 보존하는 구성을 검토한다.

AP는 average precision이며 사다리꼴 PR-AUC와 계산이 다르다. Recall은 이상 탐지율, FPR은 정상 오탐 비율이다. Valid FPR 약 1%는 분위수 임계값의 구성에 따른 결과이므로 독립적인 성능 우위의 증거로 해석하지 않는다. Test FPR이 같은 수준을 유지하는지 함께 본다. 겹치는 윈도우가 있어 표본을 독립 관측으로 해석하지 않는다.

![지표 비교](metric_comparison.png)

## N01 대비 변화

아래 delta는 비율의 차이이며, 0.01은 1%p이다. AP·recall delta가 양수면 증가, FPR delta가 양수면 오탐 증가이다.

{table(diff,['experiment','Valid_AP_delta','Test_AP_delta','Test_recall_delta','Test_FPR_delta'])}

- N02↔N01: 정상 분포의 변동 척도와 최소·최대 범위 중 어떤 스케일이 이 모델에 유리했는지 비교한다. 두 변환 모두 고정된 선형 변환이므로 날짜 편향 제거를 의미하지 않는다.
- N03↔N01: 기준선 정보를 제거해도 이상 구분 능력이 유지되는지 확인한다. 성능 하락은 유용한 평균 정보가 제거된 결과일 수도 있어 기준선이 편향이었다고 단정할 수 없다.
- N04↔N03: 진폭 제거의 영향과 구간별 통계 사용의 영향을 함께 비교한다. 진폭이 이상 근거라면 구간별 z-score가 이상을 숨길 수 있다.

![검증 학습 곡선](validation_curves.png)

## 해석의 한계와 다음 작업

정상과 이상 날짜가 완전히 겹쳐 있어 센서 차이가 고장인지 날짜별 부하·속도·설치 조건인지 분리할 수 없다. 높은 점수 또는 상대신호의 성능 유지가 날짜 편향 제거를 입증하지 않는다. 현재 라벨과 10초 뒤 라벨의 동일 여부는 위 검증 표에 기록했다. 파일 내 상태 전환이 없어 이 결과를 고장 발생 10초 전 예측 능력으로 해석할 수 없다.

다음 비교 후보는 Valid 결과를 우선 근거로 정하고, 실제 요구하는 허용 오탐률과 미탐 비용을 함께 고려한다. 추가 데이터 없이 Test를 보고 설정을 반복 조정하지 않는다. 후속 작업은 같은 seed 목록으로 반복해 변동성을 확인하고, N05/N06으로 AC RMS·P2P·FFT 보존 효과를 비교하며, 여러 날짜·운전 조건의 정상과 이상을 확보해 세션별 일반화를 평가하는 것이다.

이번 결과는 단일 seed 실험이다. 신뢰구간이나 통계적 우월성을 주장하지 않는다. 점수의 센서별 분해는 복원 오차 기여 설명이며 고장 원인 진단이 아니다.

## 결과 파일

- `metrics_comparison.csv`: Valid/Test, normal_q99/valid_max_f1 지표 전체.
- `training_summary.csv`: 종료 epoch·선택 epoch·실험 경로.
- `deltas_vs_N01.csv`: 기준 실험 대비 차이.
- `verification.csv`: 윈도우 수·라벨 동일성·유한 점수 검사.
- `N*_valid_groups.csv`, `N*_test_groups.csv`: 운전 상태·구간별 오탐과 미탐.
- 각 원본 실행 폴더: history, 가중치, scaler, 전처리 설정, 점수·혼동행렬·복원 예시.

'''
    text += '\n'.join(f'- {case}: `{path}`' for case,path in sorted(runs.items()))+'\n'
    (OUT/'normalization_report.md').write_text(text)
    with zipfile.ZipFile(OUT.parent/'normalization_report.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file(): z.write(p,arcname='report/'+p.name)
    print('Report:',OUT/'normalization_report.md',flush=True)


if __name__=='__main__':
    summarize(json.loads((OUT.parent/'campaign_runs.json').read_text()))
