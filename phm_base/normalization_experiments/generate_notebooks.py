"""Generate standalone normalization experiments; never execute notebook cells."""
import ast
import copy
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(__file__).resolve().parent
CASES = [
    ('N01', 'normal_reference_zscore', 'main/01_baseline_lstm_raw.ipynb', 'standard', 'reference'),
    ('N02', 'normal_reference_minmax', 'main/01_baseline_lstm_raw.ipynb', 'minmax', 'reference'),
    ('N03', 'centered_reference_zscore', 'main/01_baseline_lstm_raw.ipynb', 'standard', 'centered'),
    ('N04', 'window_zscore', 'main/01_baseline_lstm_raw.ipynb', 'standard', 'window'),
    ('N05', 'sensor_reference_features', 'main/04_sensor_lstm_all_features.ipynb', 'standard', 'reference'),
    ('N06', 'sensor_centered_features', 'main/04_sensor_lstm_all_features.ipynb', 'standard', 'centered'),
]

OLD = """    fitrows=sources['normal'].query("split == 'train'")[FEATURES].to_numpy(dtype=float)
    if spec['absolute']: fitrows=abs(fitrows)
    raw_scaler=(MinMaxScaler(clip=False) if spec['raw_scaler']=='minmax' else StandardScaler()).fit(fitrows)
    transformed={}
    scalers={'raw':raw_scaler}
    for split,x in raw.items():
        r=abs(x) if spec['absolute'] else x
        transformed[split]={'raw':raw_scaler.transform(r.reshape(-1,3)).reshape(r.shape)}
        transformed[split].update(feature_values(x,spec['fft_log']))
"""

NEW = """    mode=spec['relative_mode']
    eps=spec['relative_epsilon']
    # Each complete window contains only observations available at its endpoint.
    # Never center over an entire file, day, or across a segment/split boundary.
    def relative(x):
        if mode == 'reference':
            return x
        centered=x-x.mean(axis=1,keepdims=True)
        if mode == 'centered':
            return centered
        if mode == 'window':
            return centered/np.maximum(x.std(axis=1,keepdims=True,ddof=0),eps)
        raise ValueError('Unknown relative_mode: '+mode)

    transformed={}
    scalers={}
    raw_scaler=None
    if mode != 'window':
        # Preserve original row weighting for the reference controls. Centered
        # experiments fit on normal TRAIN windows (overlapping windows included).
        fitrows=(sources['normal'].query("split == 'train'")[FEATURES].to_numpy(dtype=float)
                 if mode == 'reference' else relative(raw['train']).reshape(-1,3))
        raw_scaler=(MinMaxScaler(clip=False) if spec['raw_scaler']=='minmax'
                    else StandardScaler()).fit(fitrows)
        scalers['raw']=raw_scaler
    for split,x in raw.items():
        r=relative(x)
        if raw_scaler is not None:
            r=raw_scaler.transform(r.reshape(-1,3)).reshape(r.shape)
        transformed[split]={'raw':r}
        # Keep amplitude in auxiliary features, computed BEFORE relative scaling.
        # Original FFT already removes the window mean; P2P is shift invariant.
        # AC RMS removes offsets but preserves vibration/current amplitude.
        values=feature_values(x,spec['fft_log'])
        if spec['rms_mode']=='ac':
            values['rms']=np.sqrt(np.mean((x-x.mean(axis=1,keepdims=True))**2,axis=1))
        transformed[split].update(values)
"""


def generate():
    for case_id,name,template,scaler,mode in CASES:
        nb=copy.deepcopy(json.loads((ROOT/'notebooks'/template).read_text()))
        changed=0
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                source=''.join(cell['source'])
                if 'SPEC = json.loads' in source:
                    match=re.search(r"SPEC = json.loads\(r'''(.*?)'''\)",source)
                    spec=json.loads(match.group(1))
                    spec.update(id=case_id,name=name,raw_scaler=scaler,
                                relative_mode=mode,relative_epsilon=1e-8,rms_mode='ac')
                    source=source[:match.start()]+"SPEC = json.loads(r'''"+json.dumps(spec)+"''')"+source[match.end():]
                    source=re.sub(r"ANCHOR = .*", "ANCHOR = {'experiment_id':SPEC['id'],'study_id':None,'trial_id':None,'run_id':None}",source)
                if OLD in source:
                    source=source.replace(OLD,NEW);changed+=1
                    source=source.replace("{'scalers':stats,'state_threshold':state_threshold", "{'relative_mode':mode,'relative_epsilon':eps,'rms_mode':spec['rms_mode'],'relative_fit_weighting':'normal_train_rows' if mode=='reference' else 'normal_train_windows' if mode=='centered' else 'per_window','scalers':stats,'state_threshold':state_threshold")
                source=source.replace("CAMP = 'unified_v1'", "CAMP = 'normalization_v1'")
                for key in ('WANDB_ENABLED',):
                    source=re.sub(rf'^{key} = True',f'{key} = False',source,flags=re.M)
                for key in ('RUN_SMOKE','RUN_TRAINING','RUN_TEST','EXPORT_RESULTS'):
                    source=re.sub(rf'^{key}\s*=\s*False',f'{key} = True',source,flags=re.M)
                source=re.sub(r'^STORAGE_COMPATIBLE_CODE_HASHES = .*', 'STORAGE_COMPATIBLE_CODE_HASHES = []',source,flags=re.M)
                cell['source']=source.splitlines(keepends=True)
                cell['outputs']=[];cell['execution_count']=None
        if changed!=1:
            raise RuntimeError(f'{template}: expected exactly one preprocessing replacement, got {changed}')
        nb['cells'][0]['source']=[f'# {case_id} — {name}\n\n',
            '정규화 비교 실험입니다. 원본 CSV를 수정하지 않습니다. 점검·본학습·Test·ZIP 저장은 기본 ON이며 W&B는 OFF입니다.\n\n',
            f'상대 변환: `{mode}`, raw scaler: `{scaler}`. N05/N06 보조 특징은 AC RMS·P2P·FFT입니다.\n\n',
            '먼저 이 폴더의 README.md를 읽고 설정하세요. 구간별 변환은 완성된 20행 윈도우를 사용하므로 판정 시점은 윈도우 끝입니다.\n']
        # The remaining inherited documentation describes the shared protocol.
        # Replace the condition summary to avoid showing the template's identity.
        for cell in nb['cells']:
            if cell['cell_type']=='markdown' and '이 노트북의 실험 조건' in ''.join(cell['source']):
                cell['source']=[f'## 실험 조건 — {case_id}\n\n',
                    f'`relative_mode={mode}`, `raw_scaler={scaler}`. 센서·윈도우·분할·학습 설정은 원본 템플릿과 같습니다.\n',
                    'N01~N04는 동일 baseline LSTM, N05~N06는 동일 sensor LSTM 구조입니다. 학습 epoch별 모델 선택과 최종 평가 규칙도 유지합니다.\n']
        codes=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        canonical='\n'.join(re.sub(r"^CODE_HASH = .*",'',s,flags=re.M) for s in codes)
        code_hash=hashlib.sha256(canonical.encode()).hexdigest()
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                source=re.sub(r"^CODE_HASH = .*",f"CODE_HASH = '{code_hash}'",''.join(cell['source']),flags=re.M)
                ast.parse(source)
                cell['source']=source.splitlines(keepends=True)
        path=DEST/f'{case_id}_{name}.ipynb'
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
        print(path.name)


if __name__=='__main__':
    generate()
