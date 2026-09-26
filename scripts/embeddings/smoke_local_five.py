#!/usr/bin/env python3
"""Non-downloading local-five preflight and smoke runner."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).parents[2]
sys.path.insert(0,str(ROOT/'src'))
from image_aesthetics.embeddings.feature_contract import FEATURES, DIMENSIONS
from image_aesthetics.embeddings.local_five_providers import (A1_PATH,A1_SHA256,LOCAL_FIVE,SIGLIP_MODEL,SIGLIP_REVISION,SIGLIP_SHA256,LocalFiveExtractor,local_five_contract_metadata)
from image_aesthetics.embeddings.preprocess import load_manifest
from image_aesthetics.embeddings.metal_lock import metal_owner
from image_aesthetics.embeddings.ten_signal_runner import (RunnerConfig, TenSignalRunner,
                                                           source_fingerprints_sha256)

def dump(path:Path,value:dict): path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def cache_command(cache_dir: Path, command: str) -> dict:
    """Run the public cache inspection CLI and return its JSON response."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/embeddings/cache_ten_signal.py"), command,
         "--cache-dir", str(cache_dir)],
        cwd=ROOT, env=environment, check=True, capture_output=True, text=True,
    )
    return json.loads(completed.stdout)


def successful_smoke(items, feature_contract: dict, output: Path, *,
                     extractor_factory=LocalFiveExtractor,
                     runner_factory=TenSignalRunner,
                     cache_reader=cache_command) -> dict:
    """Run and resume a real cache, then preserve the public inspection output.

    The injection points are deliberately small so this successful path can be
    exercised without a local Metal device or a model checkpoint.
    """
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "cache"
    def run_once(run_id: str) -> tuple[dict, float]:
        started = time.perf_counter()
        result = runner_factory(RunnerConfig(cache_dir), batch_extractor=extractor).run(
            items, feature_contract, run_id=run_id)
        return result, time.perf_counter() - started

    # Keep the lease across construction, both runs and destruction. This
    # prevents another dataset process from retaining a second Metal model.
    with metal_owner():
        extractor = extractor_factory()
        initial, initial_elapsed = run_once("local-five-smoke-initial")
        resumed, resume_elapsed = run_once("local-five-smoke-resume")
        del extractor
        import gc
        gc.collect()
    inspections = {command: cache_reader(cache_dir, command)
                   for command in ("status", "failures", "verify")}
    for command, value in inspections.items():
        dump(output / f"{command}.json", value)
    dump(output / "initial-result.json", initial)
    dump(output / "resume-result.json", resumed)
    completed = int(initial.get("completed_images", 0))
    metrics = {
        "schema_version": 1,
        "initial": initial,
        "resume": resumed,
        "initial_elapsed_seconds": initial_elapsed,
        "resume_elapsed_seconds": resume_elapsed,
        "initial_throughput_images_per_second": completed / initial_elapsed if initial_elapsed else 0.0,
        "resident_memory_gb": initial.get("resident_memory_gb"),
        "swap_scope": initial.get("swap_scope", "host_wide"),
        "swap_baseline_gb": initial.get("swap_baseline_gb"),
        "swap_used_gb": initial.get("swap_used_gb"),
        "swap_delta_gb": initial.get("swap_delta_gb"),
        "provider_seconds": initial.get("provider_seconds", {}),
        "provider_images": initial.get("provider_images", {}),
        "provider_failures": initial.get("provider_failures", {}),
        "decode_seconds": initial.get("decode_seconds"),
        "inspection_artifacts": {name: f"{name}.json" for name in inspections},
    }
    dump(output / "metrics.json", metrics)
    return {"initial_result": initial, "resume_result": resumed,
            "initial_elapsed_seconds": initial_elapsed, "resume_elapsed_seconds": resume_elapsed,
            "artifacts": ["initial-result.json", "resume-result.json", "metrics.json",
                          "status.json", "failures.json", "verify.json", "cache/progress.json"]}

def contract():
    meta=local_five_contract_metadata(); features={}
    for n in FEATURES:
        if n in LOCAL_FIVE:
            features[n] = {"dimension": DIMENSIONS.get(n, 1),
                           **{key: meta[n][key] for key in (
                               "dtype", "range", "direction", "model_revision",
                               "checkpoint_sha256", "preprocessing", "normalization")}}
        else:
            # These are explicit inactive declarations, not active features or
            # pretend checkpoint identities. They remain in the fixed schema.
            features[n] = {
                "dimension": DIMENSIONS.get(n, 1), "dtype": "float32",
                "range": [-1, 1] if n in DIMENSIONS else [0, 1],
                "direction": "higher_is_better", "model_revision": f"inactive-local-five:{n}",
                "checkpoint_sha256": hashlib.sha256(f"inactive-local-five:{n}".encode()).hexdigest(),
                "preprocessing": {"status": "not_generated_by_local_five"},
                "normalization": "not_generated_by_local_five",
            }
    return {"features":features}
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--image-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args(argv)
 items=load_manifest(a.manifest,image_root=a.image_root); c=contract()
 items=TenSignalRunner._fingerprint_sources(items)
 manifest_hash=hashlib.sha256(a.manifest.read_bytes()).hexdigest()
 fingerprint_hash=source_fingerprints_sha256(items)
 identity=TenSignalRunner._cache_identity(items,c,None)[0]
 report_path=a.output/'report.json';contract_path=a.output/'feature-contract.json'
 if a.output.exists():
  if not a.output.is_dir(): raise SystemExit(f'refusing to overwrite non-directory output {a.output}')
  if any(a.output.iterdir()):
   try:
    previous=json.loads(report_path.read_text())
    previous_contract=json.loads(contract_path.read_text())
   except (OSError,ValueError) as exc:
    raise SystemExit(f'refusing to reuse output without readable identity metadata: {exc}')
   expected={"manifest_sha256":manifest_hash,"source_fingerprints_sha256":fingerprint_hash,
             "image_root":str(a.image_root.resolve(strict=True)),"cache_identity":identity}
   if previous_contract != c or any(previous.get(key)!=value for key,value in expected.items()):
    raise SystemExit(f'refusing to reuse output with mismatched identity: {a.output}')
 else:
  a.output.mkdir(parents=True)
 report={"schema_version":1,"mode":"local-five-real-no-download","manifest":str(a.manifest),"manifest_sha256":manifest_hash,"image_root":str(a.image_root.resolve(strict=True)),"source_fingerprints_sha256":fingerprint_hash,"cache_identity":identity,"manifest_rows":len(items),"active_features":list(LOCAL_FIVE),"blocked_features":[x for x in FEATURES if x not in LOCAL_FIVE],"a1":{"path":str(A1_PATH),"sha256":A1_SHA256},"siglip":{"model":SIGLIP_MODEL,"revision":SIGLIP_REVISION,"sha256":SIGLIP_SHA256},"contract_metadata":local_five_contract_metadata(),"state":"preflight"}
 dump(report_path,report);dump(contract_path,c)
 if a.preflight_only: report['state']='preflight_complete';dump(report_path,report);return 0
 try:
  report.update(state='completed', **successful_smoke(items, c, a.output))
 except Exception as e:
  report.update(state='blocked',error_type=type(e).__name__,error_message=str(e));dump(report_path,report);return 2
 dump(report_path,report);return 0
if __name__=='__main__': raise SystemExit(main())
