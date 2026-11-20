"""File-first experiment driver. Open start_here.py or use VS Code F5 presets."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime

ROOT = Path(__file__).resolve().parent
METHODS = ("fixed_5", "fixed_8", "fixed_10", "fixed_16", "aac", "hmm",
           "threshold", "emission_only", "random_matched")


def resolve(path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def code_identity():
    # Changing start_here.STAGE is a UI action, not an algorithm change.
    names = ("algorithms.py", "adapter.py", "compat.py", "experiment.py", "summarize.py")
    return {name: digest_file(ROOT/name) for name in names}


def collection_protocol(cfg):
    return {k: cfg[k] for k in ("checkpoint", "suite", "task_ids", "horizon", "observation_size", "collection_chunk")} | {
        "initial_states": cfg["initial_states"]["collect"], "seed": cfg["seeds"]["collect"]}


def pending_specs(tasks, initial_ids, repeats, completed_cases):
    specs = {}
    for task in tasks:
        all_specs = [(initial_id, repeat_id, within_repeat)
                     for repeat_id in range(repeats)
                     for within_repeat, initial_id in enumerate(initial_ids)]
        specs[task] = [x for x in all_specs
                       if f"task{task:02d}_init{x[0]:02d}_repeat{x[1]:02d}" not in completed_cases]
    return specs


def load_config(path):
    cfg = read_json(path)
    # Backward-compatible defaults allow immutable legacy configs to remain
    # readable; protocol freeze still rejects these placeholder selections.
    cfg.setdefault("threshold", {"change_threshold": 0.25, "selected_from_validation": False})
    cfg.setdefault("random_matched", {"lengths": [5], "probabilities": [1.0],
                                      "source_plan_sha256": None})
    splits = cfg["initial_states"]
    for name, ids in splits.items():
        if not ids or len(set(ids)) != len(ids) or min(ids) < 0:
            raise ValueError(f"Invalid initial states: {name}")
    keys = list(splits)
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            if set(splits[a]) & set(splits[b]):
                raise ValueError(f"Initial-state leakage between {a} and {b}")
    if cfg["suite"] not in {"libero_spatial", "libero_object", "libero_10"} or not cfg["task_ids"] or cfg["horizon"] != 16:
        raise ValueError("Smoke protocol requires a supported LIBERO suite, tasks, and observed runtime H=16")
    if not 1 <= cfg["fixed_chunk"] <= cfg["horizon"] or not 1 <= cfg["collection_chunk"] <= cfg["horizon"]:
        raise ValueError("Invalid fixed chunk")
    if not 1 <= cfg["min_chunk"] <= cfg["horizon"]:
        raise ValueError("Invalid HMM-family minimum chunk")
    if not 1 <= cfg["aac"]["min_chunk"] <= cfg["horizon"]:
        raise ValueError("Invalid AAC minimum chunk")
    if cfg["aac"]["samples"] < 4 or not 0 < cfg["hmm"]["stay_threshold"] < 1:
        raise ValueError("Invalid AAC sample count or HMM threshold")
    if any(cfg["hmm"][k] < 1 for k in ("states", "iterations", "restarts")) or cfg["hmm"]["variance_floor"] <= 0:
        raise ValueError("Invalid HMM fitting settings")
    if cfg["threshold"]["change_threshold"] <= 0:
        raise ValueError("Threshold baseline requires a positive change threshold")
    random_cfg = cfg["random_matched"]
    if (len(random_cfg["lengths"]) != len(random_cfg["probabilities"]) or
            any(not 1 <= x <= cfg["horizon"] for x in random_cfg["lengths"]) or
            abs(sum(random_cfg["probabilities"])-1.0) > 1e-9):
        raise ValueError("Invalid matched-random distribution")
    return cfg


def checkpoint_identity(cfg):
    directory = resolve(cfg["checkpoint"])
    files = [p for p in sorted(directory.iterdir())
             if p.is_file() and p.suffix in (".json", ".safetensors")]
    signature = {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files}
    cache_path = ROOT/"artifacts/checkpoint_hash_cache.json"
    if cache_path.exists():
        cache = read_json(cache_path)
        if cache.get("directory") == str(directory.resolve()) and cache.get("signature") == signature:
            return cache["sha256"]
    print("체크포인트 무결성 기록 중 (캐시가 없거나 파일이 변경됨)...", flush=True)
    identity = {p.name: digest_file(p) for p in files}
    write_json(cache_path, {"directory": str(directory.resolve()), "signature": signature, "sha256": identity})
    return identity


def environment_identity():
    from importlib.metadata import version
    from compat import check_versions
    import torch
    import inspect
    from lerobot.policies.groot.groot_n1 import GR00TN15
    from lerobot.policies.groot.action_head import flow_matching_action_head
    versions = check_versions()
    versions.update({k: version(k) for k in ("numpy", "scipy", "safetensors", "tokenizers", "mujoco", "robosuite", "gymnasium")})
    return {"versions": versions, "python": platform.python_version(), "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "groot_n1_sha256": digest_file(inspect.getfile(GR00TN15)),
            "action_head_sha256": digest_file(flow_matching_action_head.__file__)}


def fingerprint_observation(observation):
    """Hash nested numerical observations; reject an empty fingerprint."""
    import numpy as np
    from collections.abc import Mapping
    h = hashlib.sha256()
    leaves = 0

    def visit(value, path):
        nonlocal leaves
        if isinstance(value, Mapping):
            for key in sorted(value, key=str):
                visit(value[key], path + [str(key)])
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, path + [str(index)])
        else:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            a = np.asarray(value)
            if a.dtype.kind in "biuf":
                header = json.dumps([path, a.dtype.str, list(a.shape)], ensure_ascii=False)
                h.update(header.encode("utf-8") + b"\0")
                h.update(np.ascontiguousarray(a).tobytes())
                leaves += 1

    visit(observation, [])
    if not leaves:
        raise RuntimeError("No numerical observation leaves; refusing an empty initial-state fingerprint")
    return h.hexdigest()


def preflight(cfg):
    import torch
    from flash_attn import flash_attn_func
    from compat import installed, tokenizer_audit
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Check the VS Code Python interpreter.")
    env = environment_identity()
    q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        y = flash_attn_func(q, q, q)
        assert y.shape == q.shape and torch.isfinite(y).all()
    saved_cfg = read_json(resolve(cfg["checkpoint"])/"config.json")
    assert saved_cfg["type"] == "groot" and saved_cfg["chunk_size"] > 0
    assert saved_cfg["output_features"]["action"]["shape"] == [7]
    assert saved_cfg["input_features"]["observation.state"]["shape"] == [8]
    artifacts = ROOT/"artifacts"
    artifacts.mkdir(exist_ok=True)
    with installed(resolve(cfg["checkpoint"])):
        audit = tokenizer_audit(artifacts/"tokenizer_audit.json")
    # Inspect saved initial states without spawning/rendering simulation environments.
    from libero.libero import benchmark
    from lerobot.envs.libero import get_task_init_states
    suite = benchmark.get_benchmark_dict()[cfg["suite"]]()
    required = max(i for ids in cfg["initial_states"].values() for i in ids)
    available = {str(t): len(get_task_init_states(suite, t)) for t in cfg["task_ids"]}
    if any(n <= required for n in available.values()):
        raise RuntimeError(f"Insufficient saved initial states: {available}")
    write_json(artifacts/"preflight.json", {"environment": env, "checkpoint": checkpoint_identity(cfg),
               "initial_state_counts": available, "tokenizer_backend_sha256": audit["backend_sha256"],
               "hardware_checks_passed": True,
               "tokenizer_policy": "legacy_regex_pinned_to_official_aac_transformers_4.51.3",
               "protocol_horizon": cfg["horizon"], "wrapper_chunk_size": saved_cfg["chunk_size"],
               "runtime_horizon_verified_by_this_preflight": False})
    print("사전 검사 완료. tokenizer는 공식 AAC의 transformers 4.51.3 동작과 맞추기 위해 legacy regex로 고정합니다.")


def assert_preflight(cfg):
    p = ROOT/"artifacts/preflight.json"
    if not p.exists():
        raise RuntimeError("먼저 preflight 실행 설정을 실행하세요.")
    manifest = read_json(p)
    if manifest.get("protocol_horizon") != cfg["horizon"]:
        raise RuntimeError("Run 01 Preflight again after the horizon repair.")
    if environment_identity() != manifest["environment"]:
        raise RuntimeError("환경이 사전 검사 때와 다릅니다. preflight를 다시 실행하세요.")
    if checkpoint_identity(cfg) != manifest["checkpoint"]:
        raise RuntimeError("체크포인트가 사전 검사 때와 다릅니다. 이전 결과와 섞지 마세요.")


def run_evaluation(cfg, stage, method, result_dir, task_override=None, resume=False):
    import numpy as np
    import torch
    from types import MethodType
    from compat import installed
    from algorithms import DiagonalHMM, action_features
    from adapter import EpisodeController

    assert_preflight(cfg)
    result_dir = Path(result_dir)
    manifest_value = {"stage": stage, "method": method, "config": cfg,
               "code": code_identity(), "environment": environment_identity(),
               "checkpoint": read_json(ROOT/"artifacts/preflight.json")["checkpoint"],
               "hmm_sha256": digest_file(resolve(cfg["hmm_model"])) if method in ("hmm", "emission_only") else None,
               "latency_note": "Sequential action-head samples with shared VLM; not official AAC parallel latency."}
    existing_records = []
    if result_dir.exists():
        if not resume:
            raise FileExistsError(result_dir)
        if not (result_dir/"manifest.json").exists() or read_json(result_dir/"manifest.json") != manifest_value:
            raise RuntimeError("Resume refused: manifest differs from current frozen execution identity")
        if (result_dir/"episodes.jsonl").exists():
            existing_records = [json.loads(line) for line in (result_dir/"episodes.jsonl").read_text().splitlines() if line.strip()]
        cases = [x["case"] for x in existing_records]
        if len(cases) != len(set(cases)) or any(x.get("status") != "completed" for x in existing_records):
            raise RuntimeError("Resume refused: duplicate or incomplete episode records")
        (result_dir/"COMPLETE.json").unlink(missing_ok=True)
    else:
        result_dir.mkdir(parents=True)
        write_json(result_dir/"manifest.json", manifest_value)
    hmm = DiagonalHMM.load(resolve(cfg["hmm_model"])) if method in ("hmm", "emission_only") else None
    effective = copy.deepcopy(cfg)
    if stage == "collect":
        if method != "fixed":
            raise ValueError("Collect uses a fixed policy, never the fitted HMM")
        effective["fixed_chunk"] = cfg["collection_chunk"]
    tasks = [0] if stage == "smoke" else cfg["task_ids"]
    if task_override is not None:
        if task_override not in tasks:
            raise ValueError("Task override is outside the configured task set")
        tasks = [task_override]
    base_initial_ids = cfg["initial_states"][stage]
    repeats = cfg["test_repeats"] if stage == "test" else 1
    completed_cases = {x["case"] for x in existing_records}
    specs = pending_specs(tasks, base_initial_ids, repeats, completed_cases)
    pending_counts = {len(value) for value in specs.values()}
    if len(pending_counts) != 1:
        raise RuntimeError("Resume with uneven task progress requires --task-id task sharding")
    pending_per_task = pending_counts.pop()
    expected_total = len(tasks) * len(base_initial_ids) * repeats
    if pending_per_task == 0:
        write_json(result_dir/"COMPLETE.json", {"episodes": len(existing_records), "stage": stage, "method": method})
        print(f"[resume] {method}: all {len(existing_records)} episodes already complete")
        return
    counts = defaultdict(int)
    plans_file = (result_dir/"plans.jsonl").open("a" if resume else "x", encoding="utf-8")
    episodes_file = (result_dir/"episodes.jsonl").open("a" if resume else "x", encoding="utf-8")
    arrays_dir = result_dir/"episodes"
    arrays_dir.mkdir(exist_ok=resume)
    created_envs = []
    old_argv = sys.argv[:]
    from lerobot.scripts import lerobot_eval as ev
    old_rollout, old_make_env = ev.rollout, ev.make_env

    def make_env(*args, **kwargs):
        envs = old_make_env(*args, **kwargs)
        created_envs.append(envs)
        return envs

    def rollout(*args, **kwargs):
        if args:
            raise RuntimeError("LeRobot rollout API changed; this adapter expects keyword arguments")
        env, policy = kwargs["env"], kwargs["policy"]
        if env.num_envs != 1 or not hasattr(env, "envs"):
            raise RuntimeError("Use one synchronous environment; parallel evaluation is disabled")
        base = env.envs[0].unwrapped
        task = int(base.task_id)
        index = counts[task]
        if index >= len(specs[task]):
            raise RuntimeError("Unexpected extra evaluation episode")
        initial_id, repeat_id, within_repeat = specs[task][index]
        if initial_id >= len(base._init_states):
            raise RuntimeError("Initial state out of bounds; refusing modulo wraparound")
        # LIBERO may reset internally on success. Force the next starting index
        # explicitly, rather than trusting a counter that depends on past success.
        base.init_state_id = initial_id
        seed = cfg["seeds"][stage] + repeat_id*10000 + task*100 + within_repeat
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        kwargs["seeds"] = [seed]
        case = f"task{task:02d}_init{initial_id:02d}_repeat{repeat_id:02d}"
        record = {"case": case, "stage": stage, "method": method, "task_id": task,
                  "initial_state_id": initial_id, "seed": seed, "repeat_id": repeat_id,
                  "initial_state_sha256": hashlib.sha256(np.asarray(base._init_states[initial_id]).tobytes()).hexdigest()}

        def record_plan(info):
            plans_file.write(json.dumps({**record, **info}, allow_nan=False)+"\n")
            plans_file.flush()

        controller = EpisodeController(policy, method, effective, seed, kwargs["postprocessor"],
                         kwargs["env_postprocessor"], record_plan, hmm_model=hmm,
                         check_parity=sum(counts.values()) == 0)
        original_select, original_reset = policy.select_action, env.reset

        def reset(*a, **kw):
            obs, info = original_reset(*a, **kw)
            if "initial_observation_sha256" not in record:
                record["initial_observation_sha256"] = fingerprint_observation(obs)
            return obs, info

        env.reset = reset
        policy.select_action = MethodType(lambda self, observation: controller.select_action(observation), policy)
        gpu_device = next(policy.parameters()).device
        if gpu_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(gpu_device)
        begin = time.perf_counter()
        try:
            result = old_rollout(**kwargs)
            elapsed = time.perf_counter()-begin
            actions = result["action"][0].detach().cpu().numpy()
            predicted_physical = np.asarray(controller.executed_physical)
            # Verify the statistics/HMM see the same commands as the simulator API.
            np.testing.assert_allclose(actions, predicted_physical, rtol=1e-5, atol=1e-5)
            success = bool(result["success"][0].any().item())
            features = action_features(actions)
            np.savez_compressed(arrays_dir/f"{case}.npz", actions=actions, features=features,
                                initial_state_id=initial_id, task_id=task, seed=seed)
            record.update(status="completed", success=success, steps=len(actions),
                          replans=controller.replans, candidates=controller.generated,
                          planning_s=controller.planning_s, filter_s=controller.online_filter_s,
                          rollout_wall_s=elapsed, parity_check_s=controller.parity_s,
                          mean_selected_chunk=float(np.mean(controller.planned_lengths)),
                          mean_executed_per_plan=len(actions)/controller.replans,
                          sampler_parity_checked=controller.parity_checked,
                          peak_gpu_gib=torch.cuda.max_memory_allocated(gpu_device)/(1024**3) if gpu_device.type == "cuda" else 0.0)
            episodes_file.write(json.dumps(record, allow_nan=False)+"\n")
            episodes_file.flush()
            counts[task] += 1
            print(f"[{method}] {case}: success={success}, steps={len(actions)}, replans={controller.replans}", flush=True)
            return result
        except BaseException:
            write_json(result_dir/f"ERROR_{case}.json", {**record, "status": "runtime_error",
                       "traceback": traceback.format_exc(), "note": "NOT counted as a task failure"})
            raise
        finally:
            policy.select_action, env.reset = original_select, original_reset

    ev.rollout, ev.make_env = rollout, make_env
    sys.argv = ["lerobot-eval", f"--policy.path={resolve(cfg['checkpoint'])}", "--policy.device=cuda",
                f"--policy.n_action_steps={cfg['fixed_chunk']}", "--env.type=libero", f"--env.task={cfg['suite']}",
                f"--env.task_ids={json.dumps(tasks)}", "--env.init_states=true", "--env.max_parallel_tasks=1",
                f"--env.observation_height={cfg['observation_size']}", f"--env.observation_width={cfg['observation_size']}",
                "--eval.batch_size=1", "--eval.use_async_envs=false", f"--eval.n_episodes={pending_per_task}",
                f"--seed={cfg['seeds'][stage]}", f"--output_dir={result_dir}"]
    try:
        with installed(resolve(cfg["checkpoint"])):
            ev.main()
        if dict(counts) != {task: pending_per_task for task in tasks}:
            raise RuntimeError(f"Incomplete rollout set: {dict(counts)}")
        total = len(existing_records) + sum(counts.values())
        if total != expected_total:
            raise RuntimeError(f"Resume total mismatch: {total} != {expected_total}")
        write_json(result_dir/"COMPLETE.json", {"episodes": total, "stage": stage, "method": method})
    finally:
        ev.rollout, ev.make_env, sys.argv = old_rollout, old_make_env, old_argv
        plans_file.close()
        episodes_file.close()
        # Native eval closes on success, but not on every exception path.
        if not (result_dir/"COMPLETE.json").exists():
            for envs in created_envs:
                try:
                    ev.close_envs(envs)
                except Exception:
                    pass


def latest_complete(cfg, stage, method):
    matches = []
    for path in resolve(cfg["output_root"]).glob(f"{stage}_{method}_*"):
        if (path/"COMPLETE.json").is_file():
            matches.append(path)
    if not matches:
        raise RuntimeError(f"No completed {stage}/{method} run")
    return sorted(matches)[-1]


def fit_stage(cfg):
    import numpy as np
    from algorithms import fit_hmm
    run = latest_complete(cfg, "collect", "fixed")
    training_manifest = read_json(run/"manifest.json")
    # Only the explicit collection split is admissible; never glob test artifacts.
    if collection_protocol(training_manifest["config"]) != collection_protocol(cfg) or training_manifest["code"] != code_identity():
        raise RuntimeError("Collection config/code differs. Recollect or review provenance; do not silently mix runs.")
    destination = resolve(cfg["hmm_model"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError("HMM artifact already exists. Choose a new hmm_model filename in config to preserve it.")
    files = sorted((run/"episodes").glob("*.npz"))
    if len(files) != len(cfg["task_ids"])*len(cfg["initial_states"]["collect"]):
        raise RuntimeError("Incomplete collection files")
    sequences, cases = [], []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            initial = int(data["initial_state_id"])
            if initial not in cfg["initial_states"]["collect"]:
                raise RuntimeError("Training/test split contamination")
            if len(data["features"]) >= 2:
                sequences.append(data["features"].copy())
                cases.append(path.stem)
    hc = cfg["hmm"]
    def progress(restart, iteration, likelihood):
        print(f"HMM 초기값 {restart}/{hc['restarts']}, 반복 {iteration}/{hc['iterations']}, log-likelihood={likelihood:.2f}", flush=True)
    model, history = fit_hmm(sequences, hc["states"], hc["iterations"], hc["restarts"],
                             hc["seed"], hc["variance_floor"], progress=progress)
    model.save(destination, collection=str(run), cases=cases, config=cfg,
               feature_definition="command_features_v1", training_uses_success_labels=False)
    write_json(destination.with_suffix(".training.json"), {"history": history, "cases": cases,
               "collection": str(run), "mean_features_by_state": (model.means*model.scale+model.center).tolist(),
               "transition": model.transition.tolist()})
    print(f"HMM CPU 학습 완료: {destination}. 상태 번호는 의미 라벨이 아닙니다.")


def frozen_identity(cfg):
    audit = ROOT/"artifacts/tokenizer_audit.json"
    return {"config": cfg, "code": code_identity(), "environment": environment_identity(),
            "hmm_sha256": digest_file(resolve(cfg["hmm_model"])),
            "checkpoint": checkpoint_identity(cfg), "tokenizer_audit_sha256": digest_file(audit)}


def freeze(cfg):
    assert_preflight(cfg)
    if not cfg["threshold"].get("selected_from_validation"):
        raise RuntimeError("Threshold baseline has not been selected from validation data")
    if not cfg["random_matched"].get("source_plan_sha256"):
        raise RuntimeError("Matched-random distribution has not been derived from frozen validation HMM plans")
    reviews = cfg["reviews"]
    required = ("tokenizer_reviewed", "preprocessing_reviewed", "aac_port_reviewed", "validation_reviewed")
    if not all(reviews[k] is True for k in required) or not reviews["notes"].strip():
        raise RuntimeError("최종평가 잠금: 토크나이저·전처리·AAC 이식·검증 결과를 실제 검토한 뒤 reviews와 notes를 기록하세요.")
    provenance = {}
    for method in METHODS:
        run = latest_complete(cfg, "validation", method)
        manifest = read_json(run/"manifest.json")
        # review fields change only when documenting completed review.
        a, b = copy.deepcopy(manifest["config"]), copy.deepcopy(cfg)
        a.pop("reviews"); b.pop("reviews")
        if a != b or manifest["code"] != code_identity() or manifest["environment"] != environment_identity():
            raise RuntimeError(f"{method} validation is stale for this protocol")
        if method in ("hmm", "emission_only") and manifest["hmm_sha256"] != digest_file(resolve(cfg["hmm_model"])):
            raise RuntimeError("HMM changed after validation")
        provenance[method] = str(run)
    lock = ROOT/"artifacts/protocol_lock.json"
    if lock.exists():
        raise RuntimeError("Protocol already frozen. Do not overwrite after seeing test outcomes.")
    write_json(lock, {"identity": frozen_identity(cfg), "validation": provenance})
    print("프로토콜 고정 완료. 이제 test300 실행 설정으로 총 900회 평가할 수 있습니다.")


def verify_lock(cfg):
    lock = ROOT/"artifacts/protocol_lock.json"
    if not lock.exists() or read_json(lock)["identity"] != frozen_identity(cfg):
        raise RuntimeError("No matching frozen protocol; run validation/review/freeze first")


def launch_child(stage, method, config_path, output, task_id=None, resume=False):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.parent/(output.name+".terminal.log")
    command = [sys.executable, "-u", str(ROOT/"experiment.py"), "--stage", stage, "--method", method,
               "--config", str(config_path), "--result-dir", str(output), "--worker"]
    if task_id is not None:
        command += ["--task-id", str(task_id)]
    if resume:
        command.append("--resume")
    with log.open("a" if resume else "x", encoding="utf-8") as f:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   bufsize=1, cwd=ROOT)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                f.write(line); f.flush()
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise
    if code:
        raise RuntimeError(f"{stage}/{method} stopped (exit={code}). Inspect {log}. No success-rate substitution.")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="preflight", choices=["preflight", "smoke", "collect", "train_hmm", "validation", "freeze", "test300", "test", "report"])
    parser.add_argument("--method", default="fixed_16", choices=("fixed",)+METHODS)
    parser.add_argument("--config", default=str(ROOT/"config.json"))
    parser.add_argument("--result-dir")
    parser.add_argument("--group-dir", help="Resume/create a test300 method-block group at this path")
    parser.add_argument("--task-id", type=int, help="Run a single task shard")
    parser.add_argument("--resume", action="store_true", help="Append only missing episodes after identity checks")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", str(cfg["cpu_threads"]))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cfg["cpu_threads"]))
    # This runner may use only one visible GPU; change the VS Code preset if your
    # allocated device is different. No broad process control or installation.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if args.stage == "preflight":
        preflight(cfg)
    elif args.stage == "train_hmm":
        fit_stage(cfg)
    elif args.stage == "freeze":
        freeze(cfg)
    elif args.stage == "report":
        from summarize import summarize
        summarize(resolve(cfg["output_root"]))
    elif args.stage == "test300":
        verify_lock(cfg)
        group = Path(args.group_dir) if args.group_dir else resolve(cfg["output_root"])/f"test300_{stamp}"
        group.mkdir(parents=True, exist_ok=True)
        group_lock = group/"protocol_lock.json"
        current_lock = read_json(ROOT/"artifacts/protocol_lock.json")
        if group_lock.exists() and read_json(group_lock) != current_lock:
            raise RuntimeError("Resume group uses a different protocol lock")
        if not group_lock.exists():
            write_json(group_lock, current_lock)
        for method in METHODS:
            for task_id in cfg["task_ids"]:
                destination = group/method/f"task{task_id:02d}"
                if (destination/"COMPLETE.json").exists():
                    print(f"[resume] already complete: {method}/task{task_id:02d}", flush=True)
                    continue
                launch_child("test", method, args.config, destination,
                             task_id=task_id, resume=destination.exists())
        from summarize import summarize
        summarize(resolve(cfg["output_root"]), group)
    else:
        if args.stage == "test":
            verify_lock(cfg)
        method = "fixed" if args.stage == "collect" else args.method
        destination = Path(args.result_dir) if args.result_dir else resolve(cfg["output_root"])/f"{args.stage}_{method}_{stamp}"
        if args.worker:
            run_evaluation(cfg, args.stage, method, destination,
                           task_override=args.task_id, resume=args.resume)
        else:
            launch_child(args.stage, method, args.config, destination,
                         task_id=args.task_id, resume=args.resume)


if __name__ == "__main__":
    main()
