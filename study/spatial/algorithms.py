"""Pure NumPy/SciPy research algorithms; no robot or GPU dependency.

AAC equations are independently implemented. See DESIGN.md for departures from
the authors' simulator implementation. HMM is an unvalidated research baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

FEATURE_NAMES = ["translation_norm", "rotation_norm", "translation_change",
                 "rotation_change", "gripper_sign", "gripper_toggle"]


def action_features(actions, previous=None):
    """Features of executable 7-D commands, not measured velocity or contact.

    Previous means last command actually executed, never an unexecuted plan.
    A first episode action has zero change features.
    """
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 7 or not np.isfinite(a).all() or not len(a):
        raise ValueError("Expected finite, nonempty (T, 7) actions")
    prev = a[0] if previous is None else np.asarray(previous, dtype=np.float64)
    preceding = np.vstack([prev, a[:-1]])
    change = a - preceding
    grip = (a[:, 6] > 0).astype(float)
    return np.column_stack([
        np.linalg.norm(a[:, :3], axis=1), np.linalg.norm(a[:, 3:6], axis=1),
        np.linalg.norm(change[:, :3], axis=1), np.linalg.norm(change[:, 3:6], axis=1),
        grip, (grip != (preceding[:, 6] > 0)).astype(float),
    ])


def entropy_curve(normalized, executable, ridge=1e-6):
    """N candidates x H future timesteps x 7 actions; covariance across N ONLY."""
    a, physical = np.asarray(normalized, float), np.asarray(executable, float)
    if a.ndim != 3 or a.shape[-1] != 7 or a.shape != physical.shape:
        raise ValueError("Expected matching (N, H, 7) action arrays")
    if a.shape[0] < 4 or ridge <= 0 or not np.isfinite(a).all() or not np.isfinite(physical).all():
        raise ValueError("AAC requires N>=4, positive ridge, and finite actions")
    parts = np.zeros((a.shape[1], 3))
    for i in range(a.shape[1]):
        for block, sl in enumerate((slice(0, 3), slice(3, 6))):
            covariance = np.cov(a[:, i, sl], rowvar=False, ddof=1) + ridge * np.eye(3)
            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0:
                raise ValueError("Non-positive covariance determinant")
            parts[i, block] = 0.5 * (3 * np.log(2 * np.pi * np.e) + logdet)
        # Sign specifies a gripper class. Swapping close/open labels leaves entropy unchanged.
        p = np.clip(np.mean(physical[:, i, 6] > 0), 1e-9, 1-1e-9)
        parts[i, 2] = -p*np.log(p) - (1-p)*np.log(1-p)
    per_step = parts.sum(axis=1)
    return per_step, np.cumsum(per_step) / np.arange(1, len(per_step)+1)


def elbow_length(prefix_mean, minimum):
    e = np.asarray(prefix_mean, float)
    if len(e) < 2 or not np.isfinite(e).all() or not 1 <= minimum <= len(e):
        raise ValueError("Invalid entropy horizon / minimum")
    # diff[0] is E_2-E_1: execute h=1 BEFORE the largest increase.
    # This is Eq.(5), not an absolute-entropy threshold. Negative/flat differences
    # retain the equation's argmax rule; do not silently force a long horizon.
    return max(int(np.argmax(np.diff(e))) + 1, minimum)


def movement_lower_bound(executable_first, threshold):
    """Author-code-style magnitude condition, generalized from H=16 to H.

    ||sum(delta_position)|| + ||composed rotvec|| + 0.2 * gripper_toggle.
    Units depend on the checkpoint/controller. NOT a universally tuned threshold.
    """
    a = np.asarray(executable_first, float)
    rotation = Rotation.identity()
    position = np.zeros(3)
    for j, command in enumerate(a):
        position += command[:3]
        rotation = Rotation.from_rotvec(command[3:6]) * rotation
        toggle = np.any((a[:j+1, 6] > 0) != (a[0, 6] > 0))
        magnitude = np.linalg.norm(position) + np.linalg.norm(rotation.as_rotvec()) + 0.2*toggle
        if j+1 >= 2 and magnitude > threshold:
            return j+1
    return len(a)


def choose_aac(normalized, executable, cfg):
    per_step, mean = entropy_curve(normalized, executable, cfg["ridge"])
    h = elbow_length(mean, cfg["min_chunk"])
    if cfg["mode"] == "author_code_adapted":
        h = max(h, movement_lower_bound(executable[0], cfg["movement_threshold"]))
    elif cfg["mode"] != "paper_eq5":
        raise ValueError("Unknown AAC mode")
    return h, {"entropy": per_step.tolist(), "prefix_mean": mean.tolist()}


@dataclass
class DiagonalHMM:
    start: np.ndarray
    transition: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    center: np.ndarray
    scale: np.ndarray

    def emission(self, x):
        z = (np.asarray(x, float) - self.center) / self.scale
        return -0.5 * (np.log(2*np.pi*self.variances)[None, :, :] +
                       (z[:, None, :] - self.means[None, :, :])**2 /
                       self.variances[None, :, :]).sum(axis=2)

    def filter_one(self, feature, previous=None):
        prior = self.start if previous is None else previous @ self.transition
        logits = np.log(np.maximum(prior, 1e-300)) + self.emission(np.asarray(feature)[None])[0]
        return np.exp(logits - logsumexp(logits))

    def forward_backward(self, x):
        b = self.emission(x)
        n, k = b.shape
        alpha = np.empty((n, k))
        lt = np.log(np.maximum(self.transition, 1e-300))
        alpha[0] = np.log(np.maximum(self.start, 1e-300)) + b[0]
        for t in range(1, n):
            alpha[t] = b[t] + logsumexp(alpha[t-1, :, None] + lt, axis=0)
        ll = float(logsumexp(alpha[-1]))
        beta = np.zeros((n, k))
        for t in range(n-2, -1, -1):
            beta[t] = logsumexp(lt + b[t+1, None, :] + beta[t+1, None, :], axis=1)
        gamma = np.exp(alpha + beta - ll)
        xis = np.zeros((k, k))
        for t in range(n-1):
            xis += np.exp(alpha[t, :, None] + lt + b[t+1, None, :] + beta[t+1, None, :] - ll)
        return ll, gamma, xis

    def save(self, path, **metadata):
        import json
        np.savez_compressed(path, start=self.start, transition=self.transition,
                            means=self.means, variances=self.variances,
                            center=self.center, scale=self.scale,
                            metadata=np.array(json.dumps(metadata, ensure_ascii=False)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            obj = cls(**{k: d[k].copy() for k in
                         ("start", "transition", "means", "variances", "center", "scale")})
        if not all(np.isfinite(v).all() for v in vars(obj).values()):
            raise ValueError("Invalid HMM parameters")
        return obj


def fit_hmm(sequences, states=3, iterations=40, restarts=3, seed=731, variance_floor=0.02, progress=None):
    """Baum-Welch, independent episode boundaries, diagonal Gaussian emissions.

    Includes binary gripper features as a Gaussian approximation; this is a
    lightweight baseline, not a calibrated contact/safety/uncertainty estimator.
    """
    if not sequences or any(len(x) < 2 for x in sequences):
        raise ValueError("Need nonempty independent sequences of at least 2 actions")
    joined = np.concatenate(sequences).astype(float)
    if joined.shape[1] != len(FEATURE_NAMES) or len(joined) < states*10 or not np.isfinite(joined).all():
        raise ValueError("Insufficient or invalid training features")
    center, scale = joined.mean(0), np.maximum(joined.std(0), 1e-6)
    z = (joined-center)/scale
    rng = np.random.default_rng(seed)
    best_model, best_ll, histories = None, -np.inf, []
    for restart in range(restarts):
        means = z[rng.choice(len(z), states, replace=False)].copy()
        for _ in range(20):
            labels = ((z[:, None]-means[None])**2).sum(2).argmin(1)
            for k in range(states):
                subset = z[labels == k]
                means[k] = subset.mean(0) if len(subset) else z[rng.integers(len(z))]
        transition = 0.1*np.ones((states, states))/states + 0.9*np.eye(states)
        model = DiagonalHMM(np.ones(states)/states, transition, means,
                            np.ones_like(means), center, scale)
        history = []
        for iteration in range(iterations):
            start_count = np.full(states, 1e-3)
            counts = np.full((states, states), 1e-3)
            occupancy = np.zeros(states)
            sums, squares = np.zeros_like(means), np.zeros_like(means)
            ll = 0.0
            for x in sequences:
                seq_ll, gamma, xi = model.forward_backward(x)
                norm_x = (x-center)/scale
                ll += seq_ll
                start_count += gamma[0]
                counts += xi
                occupancy += gamma.sum(0)
                sums += gamma.T @ norm_x
                squares += gamma.T @ (norm_x**2)
            history.append(ll)
            if progress is not None:
                progress(restart+1, iteration+1, ll)
            if iteration and abs(history[-1]-history[-2]) < 1e-5 * max(1, abs(history[-2])):
                break
            model.start = start_count/start_count.sum()
            model.transition = counts/counts.sum(1, keepdims=True)
            populated = occupancy > 1e-6
            model.means[populated] = sums[populated]/occupancy[populated, None]
            model.variances[populated] = np.maximum(
                squares[populated]/occupancy[populated, None] - model.means[populated]**2,
                variance_floor)
        final_ll = sum(model.forward_backward(x)[0] for x in sequences)
        histories.append({"restart": restart, "log_likelihood": history, "final": final_ll})
        if final_ll > best_ll:
            best_ll, best_model = final_ll, model
    return best_model, histories


class HMMController:
    """Filter actual executed commands; look ahead on predicted commands only.

    No future observation, simulator ground truth, success label, or unexecuted
    plan changes the persistent posterior.
    """
    def __init__(self, model, minimum=5, threshold=0.6):
        self.model, self.minimum, self.threshold = model, minimum, threshold
        self.posterior, self.previous_action = None, None

    def select(self, planned_actions):
        features = action_features(planned_actions, self.previous_action)
        q = self.posterior
        predictions = []
        for x in features:
            q = self.model.filter_one(x, q)
            predictions.append(q)
        predictions = np.asarray(predictions)
        anchor = int(predictions[0].argmax())
        # A low-confidence first state or a later regime shift requests an early replan.
        candidates = np.flatnonzero(predictions[:, anchor] < self.threshold)
        h = len(features) if not len(candidates) else max(self.minimum, int(candidates[0]))
        return min(h, len(features)), {"anchor_state": anchor,
                                      "state_posterior": predictions.tolist()}

    def executed(self, action):
        x = action_features(np.asarray(action)[None], self.previous_action)[0]
        self.posterior = self.model.filter_one(x, self.posterior)
        self.previous_action = np.asarray(action).copy()


class EmissionOnlyController(HMMController):
    """Memoryless HMM ablation: use fitted emissions but no transition prior."""
    def select(self, planned_actions):
        features = action_features(planned_actions, self.previous_action)
        predictions = np.asarray([self.model.filter_one(x, None) for x in features])
        anchor = int(predictions[0].argmax())
        candidates = np.flatnonzero(predictions[:, anchor] < self.threshold)
        h = len(features) if not len(candidates) else max(self.minimum, int(candidates[0]))
        return min(h, len(features)), {"anchor_state": anchor,
                                      "state_posterior": predictions.tolist(),
                                      "ablation": "emission_only"}

    def executed(self, action):
        self.previous_action = np.asarray(action).copy()


class ThresholdController:
    """Simple change-point baseline with no learned state or temporal memory."""
    def __init__(self, minimum=5, threshold=0.25):
        self.minimum, self.threshold = minimum, threshold
        self.previous_action = None

    def select(self, planned_actions):
        features = action_features(planned_actions, self.previous_action)
        score = features[:, 2] + features[:, 3] + 0.2 * features[:, 5]
        candidates = np.flatnonzero(score >= self.threshold)
        candidates = candidates[candidates >= self.minimum]
        h = len(features) if not len(candidates) else int(candidates[0])
        return min(max(h, self.minimum), len(features)), {"change_score": score.tolist()}

    def executed(self, action):
        self.previous_action = np.asarray(action).copy()


class RandomMatchedController:
    """Draw chunks from a frozen empirical distribution, independent of actions."""
    def __init__(self, lengths, probabilities, seed):
        self.lengths = np.asarray(lengths, dtype=int)
        self.probabilities = np.asarray(probabilities, dtype=float)
        if (self.lengths < 1).any() or len(self.lengths) != len(self.probabilities):
            raise ValueError("Invalid matched-random distribution")
        if not np.isclose(self.probabilities.sum(), 1.0) or (self.probabilities < 0).any():
            raise ValueError("Matched-random probabilities must be nonnegative and sum to one")
        self.rng = np.random.default_rng(seed)

    def select(self, planned_actions):
        h = int(self.rng.choice(self.lengths, p=self.probabilities))
        return min(h, len(planned_actions)), {"random_draw": h}

    def executed(self, action):
        pass


def prefix(actions, h):
    """Explicit leading prefix: never use a bounded deque to trim a full chunk."""
    if not 1 <= h <= len(actions):
        raise ValueError("Invalid execution horizon")
    return actions[:h]
