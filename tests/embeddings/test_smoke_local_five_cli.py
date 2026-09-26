from __future__ import annotations
import importlib.util, json, subprocess, sys
from pathlib import Path
import pytest

def test_local_five_preflight_writes_full_contract_without_loading_models(tmp_path):
    image_root=tmp_path/'images'; image_root.mkdir(); image=image_root/'nested'/'x.jpg'; image.parent.mkdir(); image.write_bytes(b'x'); stat=image.stat()
    manifest=tmp_path/'manifest.jsonl'; manifest.write_text(json.dumps({'image_id':'x','source_path':'nested/x.jpg','source_size':stat.st_size,'source_mtime_ns':stat.st_mtime_ns,'source_sha256':'2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881'})+'\n')
    out=tmp_path/'out'; root=Path(__file__).parents[2]
    result=subprocess.run([sys.executable,str(root/'scripts/embeddings/smoke_local_five.py'),'--manifest',str(manifest),'--image-root',str(image_root),'--output',str(out),'--preflight-only'],cwd=root,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads((out/'report.json').read_text()); contract=json.loads((out/'feature-contract.json').read_text())
    assert report['state']=='preflight_complete' and report['manifest_rows']==1
    assert set(contract['features'])=={'siglip_embedding','clip_embedding','laion_clip','a1_siglip','nima','musiq_aesthetic','musiq_technical','sharpness','contrast','colorfulness'}
    assert contract['features']['siglip_embedding']['checkpoint_sha256'] == report['siglip']['sha256']
    assert report['image_root']==str(image_root.resolve()) and report['source_fingerprints_sha256']


def test_local_five_active_contract_has_pinned_real_provenance():
    root=Path(__file__).parents[2]
    spec=importlib.util.spec_from_file_location('smoke_local_five_contract_test',root/'scripts/embeddings/smoke_local_five.py')
    smoke=importlib.util.module_from_spec(spec); spec.loader.exec_module(smoke)
    contract=smoke.contract()['features']
    for name in smoke.LOCAL_FIVE:
        value=contract[name]
        assert value['dtype']=='float32' and value['range'] in ([-1,1],[0,1])
        assert value['direction']=='higher_is_better'
        assert value['model_revision'] and value['normalization']
        assert value['preprocessing']
        assert len(value['checkpoint_sha256'])==64
        assert 'blocked' not in value['model_revision']
        assert 'blocked' not in value['normalization']
    assert contract['a1_siglip']['preprocessing']['input_signal']=='siglip_embedding'
    assert contract['a1_siglip']['preprocessing']['input_normalization']=='L2 unit norm across 768 dimensions'
    assert contract['a1_siglip']['preprocessing']['input_dtype']=='float32'
    assert 'expected_rating' in contract['a1_siglip']['normalization']
    siglip_pre=contract['siglip_embedding']['preprocessing']
    assert siglip_pre['resize']=='aspect-preserving thumbnail' and siglip_pre['padding']=='black centered square canvas'
    assert siglip_pre['model_normalization']=={
        'transform':'float32(pixel) / 127.5 - 1.0',
        'equivalent_mean':[0.5,0.5,0.5], 'equivalent_std':[0.5,0.5,0.5],
        'input_range':[0,255], 'output_range':[-1.0,1.0]}
    for name in ('sharpness','contrast','colorfulness'):
        pre=contract[name]['preprocessing']
        assert pre['input']=='raw decoded uint8 RGB pixels'
        assert 'model_normalization' not in pre and 'mean' not in pre and 'std' not in pre
    assert '15-bit fixed-point' in contract['sharpness']['preprocessing']['grayscale']
    assert 'rgb2gray15' in contract['sharpness']['model_revision']
    assert contract['sharpness']['normalization'].endswith('clipped [0,1]')


def test_local_five_preflight_reuses_same_identity_and_rejects_changed_source(tmp_path):
    image_root=tmp_path/'images'; image_root.mkdir(); image=image_root/'a.jpg'; image.write_bytes(b'one'); stat=image.stat()
    source_hash=__import__('hashlib').sha256(b'one').hexdigest()
    manifest=tmp_path/'manifest.jsonl'; manifest.write_text(json.dumps({'image_id':'a','source_path':'a.jpg','source_size':stat.st_size,'source_mtime_ns':stat.st_mtime_ns,'source_sha256':source_hash})+'\n')
    out=tmp_path/'out'; root=Path(__file__).parents[2]
    cmd=[sys.executable,str(root/'scripts/embeddings/smoke_local_five.py'),'--manifest',str(manifest),'--image-root',str(image_root),'--output',str(out),'--preflight-only']
    assert subprocess.run(cmd,cwd=root,capture_output=True,text=True).returncode==0
    assert subprocess.run(cmd,cwd=root,capture_output=True,text=True).returncode==0
    image.write_bytes(b'two'); __import__('os').utime(image,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    changed=subprocess.run(cmd,cwd=root,capture_output=True,text=True)
    assert changed.returncode!=0 and 'source fingerprint changed' in changed.stderr


def test_successful_smoke_writes_resume_metrics_and_cli_artifacts_with_injected_runner(tmp_path):
    root = Path(__file__).parents[2]
    spec = importlib.util.spec_from_file_location('smoke_local_five_test', root/'scripts/embeddings/smoke_local_five.py')
    assert spec and spec.loader
    smoke = importlib.util.module_from_spec(spec); spec.loader.exec_module(smoke)
    calls = []

    class FakeRunner:
        def __init__(self, config, *, batch_extractor):
            self.config = config
            assert batch_extractor is extractor
        def run(self, items, feature_contract, *, run_id):
            calls.append(run_id)
            return {'run_id': run_id, 'state': 'completed', 'completed_images': 1,
                    'failed_images': 0, 'decode_seconds': .01, 'resident_memory_gb': .5,
                    'swap_scope': 'host_wide', 'swap_baseline_gb': 1.0,
                    'swap_used_gb': 1.25, 'swap_delta_gb': .25,
                    'provider_seconds': {'siglip_embedding': .02},
                    'provider_images': {'siglip_embedding': 1}, 'provider_failures': {}}

    extractor = object()
    def reader(cache_dir, command):
        assert cache_dir == tmp_path/'out'/'cache'
        return {'command': command, 'errors': []}

    result = smoke.successful_smoke([object()], smoke.contract(), tmp_path/'out',
                                    extractor_factory=lambda: extractor,
                                    runner_factory=FakeRunner, cache_reader=reader)
    assert calls == ['local-five-smoke-initial', 'local-five-smoke-resume']
    assert result['initial_result']['completed_images'] == 1
    metrics = json.loads((tmp_path/'out'/'metrics.json').read_text())
    assert metrics['initial_throughput_images_per_second'] > 0
    assert metrics['provider_seconds'] == {'siglip_embedding': .02}
    assert metrics['swap_scope'] == 'host_wide' and metrics['swap_delta_gb'] == .25
    for name in ('initial-result.json', 'resume-result.json', 'status.json', 'failures.json', 'verify.json'):
        assert (tmp_path/'out'/name).exists()


def test_prepared_manifest_runs_local_five_and_resumes(tmp_path):
    import hashlib
    import numpy as np
    from PIL import Image
    root=Path(__file__).parents[2]
    spec=importlib.util.spec_from_file_location('smoke_local_five_e2e',root/'scripts/embeddings/smoke_local_five.py')
    smoke=importlib.util.module_from_spec(spec); spec.loader.exec_module(smoke)
    image_root=tmp_path/'images'; (image_root/'nested').mkdir(parents=True)
    image=image_root/'nested'/'small.png'; Image.new('RGB',(18,10),(90,50,20)).save(image)
    stat=image.stat(); digest=hashlib.sha256(image.read_bytes()).hexdigest()
    manifest=tmp_path/'prepared.jsonl'; manifest.write_text(json.dumps({'image_id':'fixture:1','source_path':'nested/small.png','source_size':stat.st_size,'source_mtime_ns':stat.st_mtime_ns,'source_sha256':digest})+'\n')
    items=smoke.load_manifest(manifest,image_root=image_root)
    class FakeLocalFive:
        names=smoke.LOCAL_FIVE
        def extract(self,images):
            out=np.zeros((len(images),768),dtype=np.float32); out[:,0]=1.
            return {'siglip_embedding':out,'a1_siglip':np.full((len(images),1),.5,dtype=np.float32),
                    'sharpness':np.zeros((len(images),1),dtype=np.float32),
                    'contrast':np.zeros((len(images),1),dtype=np.float32),
                    'colorfulness':np.zeros((len(images),1),dtype=np.float32)}
    output=tmp_path/'smoke'
    result=smoke.successful_smoke(items,smoke.contract(),output,extractor_factory=FakeLocalFive)
    assert result['initial_result']['state']=='completed' and result['initial_result']['completed_images']==1
    assert result['resume_result']['state']=='completed' and result['resume_result']['completed_images']==1
    assert json.loads((output/'verify.json').read_text())['ledger']=='ok'
    import os
    from image_aesthetics.embeddings.ten_signal_runner import RunnerConfig, TenSignalRunner
    mutated=bytearray(image.read_bytes()); mutated[0]^=1; image.write_bytes(mutated); os.utime(image,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    with pytest.raises(ValueError,match='source fingerprint changed'):
        TenSignalRunner(RunnerConfig(output/'cache'),batch_extractor=FakeLocalFive()).run(
            items,smoke.contract(),run_id='must-reject-mutated-source')
