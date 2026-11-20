"""LeRobot adapter. Integration requires the user's remote GPU environment.

Uses the native GR00T backbone and action head, with single-observation batches.
AAC samples action heads sequentially while reusing VLM features (3090 memory
budget); this is NOT the paper's parallel latency implementation.
"""
from collections import deque
from contextlib import contextmanager
from time import perf_counter
import numpy as np
from algorithms import (choose_aac, EmissionOnlyController, HMMController,
                        RandomMatchedController, ThresholdController, prefix)


@contextmanager
def seeded_torch(seed, device):
    import torch
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        yield


def synchronize(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sample_candidates(policy, batch, count, plan_seed):
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    policy.eval()
    allowed = {"state", "state_mask", "embodiment_id"}
    inputs = {k: v for k, v in batch.items() if k in allowed or k.startswith("eagle_")}
    groot = policy._groot_model
    device = next(policy.parameters()).device
    synchronize(device)
    begin = perf_counter()
    candidates, head_times = [], []
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                               enabled=policy.config.use_bf16):
        backbone_inputs, action_inputs = groot.prepare_input(inputs)
        backbone = groot.backbone(backbone_inputs)
        synchronize(device)
        encoder_s = perf_counter()-begin
        for n in range(count):
            started = perf_counter()
            with seeded_torch(plan_seed+n, device):
                # Protect shared features from any dictionary assignments in action head.
                out = groot.action_head.get_action(BatchFeature(data=dict(backbone)),
                                                   BatchFeature(data=dict(action_inputs)))
            groot.validate_data(out, backbone, is_training=False)
            candidate = out["action_pred"][:, :, :7]
            expected = (1, int(groot.action_horizon), 7)  # horizon16_repair_v1
            if candidate.shape != expected:
                raise RuntimeError(f"Expected internal model shape {expected}, got {candidate.shape}")
            if not torch.isfinite(candidate).all():
                raise RuntimeError("Non-finite model prediction")
            candidates.append(candidate)
            synchronize(device)
            head_times.append(perf_counter()-started)
    return torch.cat(candidates, dim=0), {"encoder_s": encoder_s, "head_s": head_times}


def executable_candidates(normalized, postprocessor, env_postprocessor):
    # Postprocessor's 3-D input branch selects only the last action. Flatten H
    # into a batch so every timestep is preserved, then restore N,H,7.
    shape = normalized.shape
    flat = normalized.reshape(-1, 7).clone()
    out = env_postprocessor({"action": postprocessor(flat)})["action"]
    if out.shape != flat.shape:
        raise RuntimeError(f"Action conversion changed shape: {flat.shape} -> {out.shape}")
    return out.detach().float().cpu().numpy().reshape(shape)


class EpisodeController:
    def __init__(self, policy, method, cfg, episode_seed, postprocessor, env_postprocessor,
                 record_plan, hmm_model=None, check_parity=False):
        actual_horizon = int(policy._groot_model.action_horizon)
        if cfg["horizon"] != actual_horizon:
            raise RuntimeError(f"Protocol horizon {cfg['horizon']} differs from internal model horizon {actual_horizon}")
        if not 1 <= cfg["fixed_chunk"] <= actual_horizon:
            raise RuntimeError("Fixed execution length exceeds actual generated horizon")
        self.policy, self.method, self.cfg = policy, method, cfg
        self.episode_seed = episode_seed
        self.postprocessor, self.env_postprocessor = postprocessor, env_postprocessor
        self.record_plan = record_plan
        self.queue = deque()  # unbounded; enqueue an explicitly sliced prefix
        self.replans, self.steps, self.generated = 0, 0, 0
        self.planning_s, self.online_filter_s, self.parity_s = 0.0, 0.0, 0.0
        self.check_parity = check_parity
        self.parity_checked = False
        self.executed_physical, self.planned_lengths = [], []
        if method == "hmm":
            self.adaptive = HMMController(hmm_model, cfg["min_chunk"], cfg["hmm"]["stay_threshold"])
        elif method == "emission_only":
            self.adaptive = EmissionOnlyController(hmm_model, cfg["min_chunk"], cfg["hmm"]["stay_threshold"])
        elif method == "threshold":
            self.adaptive = ThresholdController(cfg["min_chunk"], cfg["threshold"]["change_threshold"])
        elif method == "random_matched":
            rc = cfg["random_matched"]
            self.adaptive = RandomMatchedController(rc["lengths"], rc["probabilities"], episode_seed)
        else:
            self.adaptive = None
        self.hmm = self.adaptive if method in ("hmm", "emission_only") else None

    def select_action(self, batch):
        import torch
        if not self.queue:
            device = next(self.policy.parameters()).device
            count = self.cfg["aac"]["samples"] if self.method == "aac" else 1
            seed = (self.episode_seed*100003 + self.replans*101) % (2**31-1)
            # First observation only: compare the custom shared-feature sampler with
            # native predict_action_chunk at exactly the same diffusion-noise seed.
            if self.check_parity and not self.parity_checked:
                began = perf_counter()
                with seeded_torch(seed, device):
                    native = self.policy.predict_action_chunk(batch)
                print("기본 모델 출력:", tuple(native.shape), "설정 청크:", self.policy.config.chunk_size, "내부 예측 길이:", self.policy._groot_model.action_horizon, flush=True)
                probe, _ = sample_candidates(self.policy, batch, 1, seed)
                torch.testing.assert_close(probe[0:1].float(), native.float(), atol=1e-3, rtol=1e-3)
                # Check vectorized unnormalization against the actual one-action path.
                vector = executable_candidates(probe, self.postprocessor, self.env_postprocessor)[0, 0]
                single = self.env_postprocessor({"action": self.postprocessor(probe[:, 0].clone())})["action"]
                np.testing.assert_allclose(vector, single.detach().float().cpu().numpy()[0], atol=1e-6)
                synchronize(device)
                self.parity_s += perf_counter()-began
                self.parity_checked = True
            synchronize(device)
            began = perf_counter()
            candidates, timing = sample_candidates(self.policy, batch, count, seed)
            raw = candidates.detach().float().cpu().numpy()
            physical = executable_candidates(candidates, self.postprocessor, self.env_postprocessor)
            if not np.isfinite(physical).all():
                raise RuntimeError("Non-finite executable action")
            detail = {}
            if self.method == "fixed":
                h = self.cfg["fixed_chunk"]
            elif self.method.startswith("fixed_"):
                h = int(self.method.split("_", 1)[1])
            elif self.method == "aac":
                h, detail = choose_aac(raw, physical, self.cfg["aac"])
            elif self.method in ("hmm", "emission_only", "threshold", "random_matched"):
                h, detail = self.adaptive.select(physical[0])
            else:
                raise ValueError(self.method)
            # Candidate zero always supplies executed actions; AAC changes length,
            # not the trajectory-selection strategy.
            for j in range(len(prefix(raw[0], h))):
                self.queue.append((candidates[0:1, j, :], physical[0, j].copy()))
            synchronize(device)
            elapsed = perf_counter()-began
            self.planning_s += elapsed
            self.generated += count
            self.planned_lengths.append(int(h))
            self.record_plan({"replan": self.replans, "step": self.steps, "h": int(h), "samples": count,
                              "plan_seed": seed, "planning_s": elapsed, **timing, **detail})
            self.replans += 1
        action, physical_action = self.queue.popleft()
        if self.adaptive is not None:
            began = perf_counter()
            self.adaptive.executed(physical_action)
            self.online_filter_s += perf_counter()-began
        self.executed_physical.append(physical_action)
        self.steps += 1
        return action
