"""Finish N01, train N03/N02/N04 sequentially, freeze choices, evaluate Test.

No hyperparameter selection is made from Test results. Run from repository root.
"""
import gc
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from run_n01 import ROOT, OUT

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.matplotlib'))
HERE = Path(__file__).resolve().parent
STATE = OUT / 'campaign_status.json'
RESULTS = OUT / 'campaign_runs.json'
LOCK = OUT / '.campaign.lock'
ORDER = ('N01', 'N03', 'N02', 'N04')


def record(stage, **details):
    payload = dict(stage=stage, pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), **details)
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temp.replace(STATE)
    print(stage, details, flush=True)


def prepare(case):
    from IPython.display import display
    notebook = next(HERE.glob(case + '_*.ipynb'))
    ns = {'__name__': '__main__', 'display': display}
    for i, cell in enumerate(json.loads(notebook.read_text())['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if source.startswith('# 화면 확인 전용입니다.'):
            break
        exec(compile(source, f'{notebook.name}:cell{i}', 'exec'), ns)
        if 'def detect_environment' in source:
            ns.update(DATA_MODE='folder', DATA_ROOT=Path('/Users/pds2023/Desktop/KAMP/3. 소성가공 예지보전 AI 데이터셋'),
                OUTPUT_ROOT=OUT, DEVICE_REQUEST='cpu', ALLOW_CPU_TRAINING=True,
                INSTALL_MISSING=False, RUN_SMOKE=False, RUN_TRAINING=True, RUN_TEST=False,
                WANDB_ENABLED=False, SHOW_INLINE_RESULTS=False)
    ns['torch'].set_num_threads(2)
    return ns


def save_runs(runs):
    temp = RESULTS.with_suffix('.tmp')
    temp.write_text(json.dumps(runs, ensure_ascii=False, indent=2))
    temp.replace(RESULTS)


def verify(ns, reference):
    assert ns['DATA']['split_hash'] == reference['split_hash'], 'Data split differs'
    assert ns['TRAIN_CONFIG'] == reference['train'], 'Training settings differ'
    assert ns['SEED'] == reference['seed'], 'Seed differs'
    assert ns['DATA']['source_hashes'] == reference['source_hashes'], 'CSV content differs'


def main():
    # Avoid launching a second campaign. Stale locks require explicit inspection.
    fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    runs = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    try:
        record('waiting_for_N01')
        while True:
            state = json.loads((OUT / 'N01_runner_status.json').read_text())
            if state['state'] == 'complete':
                runs['N01'] = state['run_dir']; save_runs(runs); break
            if state['state'] == 'failed':
                raise RuntimeError('N01 failed: ' + state['error'])
            os.kill(state['pid'], 0)
            time.sleep(5)
        reference = json.loads((Path(runs['N01']) / 'config/experiment.json').read_text())
        for case in ORDER[1:]:
            if case in runs and (Path(runs[case]) / 'selection/primary.json').exists():
                continue
            record('preparing', experiment=case)
            ns = prepare(case); verify(ns, reference)
            def on_epoch(run, epoch):
                runs[case] = str(Path(run).resolve()); save_runs(runs)
                record('training', experiment=case, epoch=epoch, run_dir=runs[case])
            # Resume an interrupted run only under the notebook's compatibility checks.
            resume = runs.get(case)
            run, _ = ns['run_training'](ns['DATA'], ns['SPEC'], ns['TRAIN_CONFIG'],
                resume=resume, on_epoch=on_epoch)
            runs[case] = str(Path(run).resolve()); save_runs(runs)
            del ns; gc.collect()
        # Freeze all four model/threshold choices before opening any Test results.
        for case in ORDER:
            ns = prepare(case); verify(ns, reference)
            run = Path(runs[case])
            selected = json.loads(ns['result_path'](run, 'selection.json').read_text())
            ns['save_json'](ns['result_path'](run, 'test_release.json'),
                dict(selection_hash=ns['digest'](selected), policy='fixed campaign: best normal Valid loss; normal_q99 primary',
                     frozen_utc=datetime.now(timezone.utc).isoformat()))
            del ns; gc.collect()
        for case in ORDER:
            record('test_evaluation', experiment=case)
            run = Path(runs[case])
            if (run / 'test/primary/metrics.json').exists():
                continue
            ns = prepare(case); verify(ns, reference)
            ns['evaluate_test'](ns['DATA'], run, ns['SPEC'], ns['TRAIN_CONFIG'])
            del ns; gc.collect()
        record('creating_report')
        from summarize_campaign import summarize
        summarize(runs)
        record('complete', experiments=runs, report=str(OUT / 'report/normalization_report.md'))
    except BaseException as error:
        record('failed', error=str(error)); traceback.print_exc(); raise
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
