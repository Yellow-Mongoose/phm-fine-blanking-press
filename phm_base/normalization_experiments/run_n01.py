"""Run N01 training from its notebook, without rerunning smoke or Test.

Run from the repository root using .venv/bin/python -u.
The notebook remains unchanged; progress and result path go to a text log.
"""
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'phm_base/results/normalization'
OUT.mkdir(parents=True, exist_ok=True)
STATUS = OUT / 'N01_runner_status.json'


def status(state, **details):
    payload = dict(state=state, pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), **details)
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temporary.replace(STATUS)


def main():
    if STATUS.exists():
        previous = json.loads(STATUS.read_text())
        if previous.get('state') in ('preparing', 'training'):
            try:
                os.kill(previous['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('An N01 runner is already active.')
        if previous.get('state') == 'complete':
            raise RuntimeError('N01 has already completed; inspect the recorded run_dir.')
    status('preparing')
    print('N01: preparing data and model definitions; smoke/Test/W&B disabled.', flush=True)
    notebook = Path(__file__).with_name('N01_normal_reference_zscore.ipynb')
    cells = json.loads(notebook.read_text())['cells']
    namespace = {'__name__': '__main__'}
    from IPython.display import display
    namespace['display'] = display
    try:
        for index, cell in enumerate(cells):
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            if source.startswith('# 화면 확인 전용입니다.'):
                break  # All shared definitions and data preparation are now ready.
            print(f'Preparing notebook cell {index}', flush=True)
            exec(compile(source, f'{notebook.name}:cell{index}', 'exec'), namespace)
            if 'def detect_environment' in source:
                namespace.update(DATA_MODE='folder',
                    DATA_ROOT=Path('/Users/pds2023/Desktop/KAMP/3. 소성가공 예지보전 AI 데이터셋'),
                    OUTPUT_ROOT=OUT, DEVICE_REQUEST='cpu', ALLOW_CPU_TRAINING=True,
                    INSTALL_MISSING=False, RUN_SMOKE=False, RUN_TRAINING=True,
                    RUN_TEST=False, WANDB_ENABLED=False, SHOW_INLINE_RESULTS=False)
        namespace['torch'].set_num_threads(2)
        data = namespace['DATA']
        print('Split hash:', data['split_hash'], flush=True)
        print('Training configuration:', namespace['TRAIN_CONFIG'], flush=True)
        status('training', split_hash=data['split_hash'])
        run, selection = namespace['run_training'](data, namespace['SPEC'], namespace['TRAIN_CONFIG'])
        status('complete', run_dir=str(Path(run).resolve()), selection=selection)
        print('TRAINING COMPLETE. RUN_DIR:', Path(run).resolve(), flush=True)
        print('Test has not been executed.', flush=True)
    except BaseException as error:
        status('failed', error=str(error))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
