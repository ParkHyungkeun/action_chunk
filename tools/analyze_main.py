#!/usr/bin/env python3
"""Fail-closed statistical report for the frozen paired main experiment."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("spatial", "object", "long")
METHODS = ("fixed_5", "fixed_8", "fixed_10", "fixed_16", "aac", "hmm",
           "threshold", "emission_only", "random_matched")
REFERENCE = "hmm"
PRIMARY = "fixed_16"
BOOTSTRAPS = 20_000


def wilson(successes, n, z=1.959963984540054):
    p = successes / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return center-half, center+half


def exact_mcnemar(b, c):
    n = b+c
    if n == 0:
        return 1.0
    k = min(b, c)
    lower = sum(math.comb(n, i) for i in range(k+1)) / (2**n)
    return min(1.0, 2*lower)


def holm(items):
    ordered = sorted(items, key=lambda x: x[1])
    adjusted, running = {}, 0.0
    size = len(ordered)
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (size-rank)*p))
        adjusted[name] = running
    return adjusted


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


pointer = json.loads((ROOT / "ACTIVE_MAIN_RUN.json").read_text())
group = Path(pointer["group"])
complete = json.loads((group / "COMPLETE.json").read_text())
if complete != {"completed": 12150, "expected": 12150, "failures": []}:
    raise RuntimeError("Main experiment is not exactly and cleanly complete")
output = group / "analysis"
output.mkdir(exist_ok=True)
data, integrity = {}, []

for suite in SUITES:
    data[suite] = {}
    manifests = {}
    observation_mismatches = {}
    for method in METHODS:
        run = group / suite / method
        rows = [json.loads(x) for x in (run / "episodes.jsonl").read_text().splitlines() if x.strip()]
        manifest = json.loads((run / "manifest.json").read_text())
        done = json.loads((run / "COMPLETE.json").read_text())
        cases = [r["case"] for r in rows]
        valid = (len(rows) == 450 and done.get("episodes") == 450 and
                 len(cases) == len(set(cases)) and
                 all(r["status"] == "completed" and isinstance(r["success"], bool) for r in rows) and
                 not list(run.glob("ERROR_*.json")))
        if not valid:
            raise RuntimeError(f"Invalid or incomplete cell: {suite}/{method}")
        data[suite][method] = {r["case"]: r for r in rows}
        manifests[method] = manifest
    reference_cases = set(data[suite][REFERENCE])
    reference_manifest = manifests[REFERENCE]
    for method in METHODS:
        if set(data[suite][method]) != reference_cases:
            raise RuntimeError(f"Case mismatch: {suite}/{method}")
        for key in ("config", "code", "checkpoint", "environment"):
            if manifests[method][key] != reference_manifest[key]:
                raise RuntimeError(f"Manifest {key} mismatch: {suite}/{method}")
        for case in reference_cases:
            a, b = data[suite][REFERENCE][case], data[suite][method][case]
            for key in ("task_id", "initial_state_id", "repeat_id", "seed",
                        "initial_state_sha256"):
                if a[key] != b[key]:
                    raise RuntimeError(f"Pair mismatch: {suite}/{method}/{case}/{key}")
        observation_mismatches[method] = sum(
            data[suite][REFERENCE][case]["initial_observation_sha256"] !=
            data[suite][method][case]["initial_observation_sha256"]
            for case in reference_cases)
    integrity.append({"suite": suite, "methods": 9, "cases_per_method": 450,
                      "paired_case_set": True, "duplicates": 0, "errors": 0,
                      "manifest_identity": True,
                      "exact_initial_state_pairing": True,
                      "initial_observation_hash_mismatches_vs_hmm": observation_mismatches})

summary_rows, task_rows, pair_rows = [], [], []
rng = np.random.default_rng(20260907)
for suite in SUITES:
    for method in METHODS:
        rows = list(data[suite][method].values())
        wins = sum(r["success"] for r in rows)
        lo, hi = wilson(wins, len(rows))
        summary_rows.append({"suite": suite, "method": method, "n": len(rows),
            "successes": wins, "success_rate": wins/len(rows), "wilson95_low": lo,
            "wilson95_high": hi, "mean_steps": np.mean([r["steps"] for r in rows]),
            "mean_replans": np.mean([r["replans"] for r in rows]),
            "mean_selected_chunk": np.mean([r["mean_selected_chunk"] for r in rows]),
            "mean_executed_per_plan": np.mean([r["mean_executed_per_plan"] for r in rows]),
            "mean_policy_compute_s": np.mean([r["planning_s"]+r["filter_s"] for r in rows]),
            "mean_wall_s": np.mean([r["rollout_wall_s"] for r in rows])})
        for task in range(10):
            subset = [r for r in rows if r["task_id"] == task]
            task_rows.append({"suite": suite, "method": method, "task_id": task,
                              "n": len(subset), "successes": sum(r["success"] for r in subset),
                              "success_rate": np.mean([r["success"] for r in subset])})

    raw = []
    hmm = data[suite][REFERENCE]
    tasks = sorted({r["task_id"] for r in hmm.values()})
    for method in METHODS:
        if method == REFERENCE:
            continue
        other = data[suite][method]
        keys = sorted(hmm)
        b = sum(hmm[k]["success"] and not other[k]["success"] for k in keys)
        c = sum(other[k]["success"] and not hmm[k]["success"] for k in keys)
        diff = np.mean([hmm[k]["success"]-other[k]["success"] for k in keys])
        task_diffs = []
        for task in tasks:
            tk = [k for k in keys if hmm[k]["task_id"] == task]
            task_diffs.append(np.mean([hmm[k]["success"]-other[k]["success"] for k in tk]))
        draws = rng.choice(task_diffs, size=(BOOTSTRAPS, len(task_diffs)), replace=True).mean(axis=1)
        p = exact_mcnemar(b, c)
        item = {"suite": suite, "comparison": f"hmm_vs_{method}", "other": method,
                "primary": method == PRIMARY, "n_pairs": len(keys),
                "hmm_only_success": b, "other_only_success": c,
                "success_rate_difference": float(diff), "mcnemar_exact_p": p,
                "task_cluster_bootstrap95_low": float(np.quantile(draws, .025)),
                "task_cluster_bootstrap95_high": float(np.quantile(draws, .975)),
                "bootstrap_replicates": BOOTSTRAPS}
        pair_rows.append(item); raw.append((method, p))
    # The fixed_16 comparison is the sole prespecified primary comparison and
    # is reported unadjusted. Holm correction covers the seven secondary tests.
    adjusted = holm([(name, p) for name, p in raw if name != PRIMARY])
    for item in pair_rows:
        if item["suite"] == suite:
            item["holm_p_within_suite"] = (None if item["primary"] else adjusted[item["other"]])


def write_csv(name, rows):
    with (output / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


write_csv("summary.csv", summary_rows)
write_csv("per_task.csv", task_rows)
write_csv("pairwise_hmm.csv", pair_rows)
payload = {"group": str(group), "integrity": integrity, "summary": summary_rows,
           "pairwise_hmm": pair_rows,
           "analysis": {"reference": REFERENCE, "primary_comparator": PRIMARY,
                        "confidence_level": .95, "bootstrap_replicates": BOOTSTRAPS,
                        "bootstrap_unit": "task", "multiple_testing": "Holm within suite"}}
(output / "final_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

lookup = {(r["suite"], r["method"]): r for r in summary_rows}
lines = ["# Final paired test report", "", f"Source: `{group}`", "",
         "All 12,150 frozen test rollouts completed with zero runtime errors, zero duplicates, and exact case/simulator-state pairing.", "",
         "## Success rates", "", "| Suite | Method | Success | Rate | Wilson 95% CI | Mean replans |", "|---|---|---:|---:|---:|---:|"]
for suite in SUITES:
    for method in METHODS:
        r = lookup[suite, method]
        lines.append(f"| {suite} | {method} | {r['successes']}/{r['n']} | {100*r['success_rate']:.1f}% | {100*r['wilson95_low']:.1f}–{100*r['wilson95_high']:.1f}% | {r['mean_replans']:.1f} |")
lines += ["", "## Paired HMM comparisons", "",
          "Positive differences favor HMM. The prespecified primary comparison is HMM minus fixed_16; other comparisons are secondary.", "",
          "| Suite | Comparison | Difference | Task-cluster bootstrap 95% CI | McNemar p | Holm p |", "|---|---|---:|---:|---:|---:|"]
for r in pair_rows:
    label = "HMM vs " + r["other"] + (" (primary)" if r["primary"] else "")
    holm_text = "—" if r["holm_p_within_suite"] is None else f"{r['holm_p_within_suite']:.4g}"
    lines.append(f"| {r['suite']} | {label} | {100*r['success_rate_difference']:+.1f} pp | {100*r['task_cluster_bootstrap95_low']:+.1f} to {100*r['task_cluster_bootstrap95_high']:+.1f} pp | {r['mcnemar_exact_p']:.4g} | {holm_text} |")
lines += ["", "## Interpretation guardrails", "",
          "- Wilson intervals describe marginal success rates; paired comparisons use the matched cases.",
          "- Task-cluster bootstrap reflects variation across the 10 LIBERO tasks; only 10 task clusters are available per suite.",
          "- The HMM–fixed_16 primary McNemar test is unadjusted; Holm correction covers the seven secondary HMM comparisons within each suite.",
          "- Simulator state hashes, seeds, task IDs, initial-state IDs, and repeat IDs pair exactly. Spatial and Long contain some differing rendered initial-observation hashes despite identical simulator states; these are disclosed in `final_analysis.json` as a rendering reproducibility limitation.",
          "- Results apply to the three evaluated N1.5 checkpoints and do not establish checkpoint-seed or model-family generalization.",
          "- Validation data were used for protocol selection; only this frozen test run supports confirmatory claims."]
(output / "FINAL_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
(output / "SHA256SUMS.txt").write_text("\n".join(
    f"{sha(output/name)}  {name}" for name in
    ("summary.csv", "per_task.csv", "pairwise_hmm.csv", "final_analysis.json", "FINAL_REPORT.md"))+"\n", encoding="utf-8")
print(output)
