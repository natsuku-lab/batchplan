"""The comparison's tables, regenerated from saved runs (`compare.py run`, then `validate.py run`).

For each run directory: how many queries each planner claims to solve and how many the judge accepts
(with their failure reasons); end-to-end time per query, failures included at what they cost (median,
p95, p99); throughput as judge-valid results per second of pass wall time; and, over the problems every
compared configuration solves validly, path length and trajectory duration with 95% bootstrap intervals
resampled by problem (each problem's mean over its seeds).

    python scripts/report.py data/bench/runs/<run> [<run> ...] [--markdown]
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

KINODYNAMIC = {"velocity", "acceleration", "jerk"}


def load(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    records = [json.loads(l) for l in open(run / "records.jsonl")]
    passes = [json.loads(l) for l in open(run / "passes.jsonl")]
    verdicts = {}
    if (run / "verdicts.jsonl").exists():
        for l in open(run / "verdicts.jsonl"):
            v = json.loads(l)
            verdicts[(v["problem"], v["seed"])] = v
    return manifest, records, passes, verdicts


def label(manifest):
    s = manifest.get("settings", {})
    extra = s.get("mode") or ("ruckig" if s.get("smoothing") == "ruckig" else "")
    budget = f"{manifest['workers']} core" + ("s" if manifest["workers"] > 1 else "")
    return " ".join(x for x in [manifest["planner"], manifest["workload"], manifest["family"], extra, budget] if x)


def valid_within(v, tolerance):
    """A verdict re-read with a looser tolerance on the kinodynamic limits only."""
    if v is None:
        return False
    if v["valid"]:
        return True
    reasons = {r.strip() for r in v["reason"].split(",") if r.strip()}
    if not reasons <= KINODYNAMIC:
        return False
    return all(v.get(f"{r}_ratio", 0) <= 1 + tolerance for r in reasons)


def summarize(run):
    manifest, records, passes, verdicts = load(run)
    singles = [r for r in records if r.get("api", "single") == "single"]
    first = [r for r in singles if r["rep"] == 1]
    claimed = sum(r["status"] == "solved" for r in first)
    judged = [verdicts.get((r["problem"], r["seed"])) if r["status"] == "solved" else None for r in first]
    valid = sum(bool(v and v["valid"]) for v in judged)
    valid10 = sum(valid_within(v, 0.10) for v in judged)
    reasons = Counter(v["reason"] for v in judged if v and not v["valid"])
    statuses = Counter(r["status"] if r["status"] != "solved" else "solved" for r in first)
    times = np.array([r["phases"]["total"] for r in singles]) * 1e3
    walls = [p["wall"] for p in passes]
    answers = sum(p["problems"] for p in passes)
    valid_rate = valid / max(1, len(first))
    return {
        "label": label(manifest), "manifest": manifest, "queries": len(first), "claimed": claimed,
        "valid": valid, "valid10": valid10, "reasons": reasons, "statuses": statuses,
        "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        # Every pass answers every problem; judge-valid answers per second, scaled by the first
        # repetition's valid rate (outputs are judged for repetition 1).
        "answers_per_s": answers / sum(walls), "valid_per_s": valid_rate * answers / sum(walls),
        "verdicts": {(r["problem"], r["seed"]): v for r, v in zip(first, judged)},
        "batch_passes": [p for p in passes if "single_wall" in p],
    }


def bootstrap(values, rng, n=2000):
    values = np.asarray(values)
    if len(values) == 0:
        return float("nan"), float("nan")
    means = [rng.choice(values, len(values)).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def common(summaries, key):
    """Per configuration, the per-problem mean of `key` over its valid seeds, on the problems every
    configuration solves validly for at least one seed."""
    per = []
    for s in summaries:
        by_problem = defaultdict(list)
        for (problem, _), v in s["verdicts"].items():
            if v and v["valid"] and key in v:
                by_problem[problem].append(v[key])
        per.append({p: float(np.mean(x)) for p, x in by_problem.items()})
    shared = sorted(set.intersection(*(set(p) for p in per))) if per else []
    return shared, [np.array([p[k] for k in shared]) for p in per]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()
    summaries = [summarize(r) for r in args.runs]
    rng = np.random.default_rng(0)
    print(f"{'configuration':<40} {'queries':>7} {'claimed':>8} {'valid':>8} {'≤10%':>8} {'median ms':>10} {'p95 ms':>9} "
          f"{'p99 ms':>9} {'valid/s':>9}")
    for s in summaries:
        q = s["queries"]
        pct = lambda k: f"{100 * k / q:.1f}%"
        print(f"{s['label']:<40} {q:>7} {pct(s['claimed']):>8} {pct(s['valid']):>8} {pct(s['valid10']):>8} "
              f"{s['median_ms']:>10.2f} {s['p95_ms']:>9.1f} {s['p99_ms']:>9.1f} {s['valid_per_s']:>9.1f}")
    for s in summaries:
        if s["reasons"]:
            print(f"  {s['label']}: invalid " + ", ".join(f"{r} {n}" for r, n in s["reasons"].most_common()))
        for p in s["batch_passes"][:1]:
            print(f"  {s['label']}: native batch {p['problems'] / p['wall']:.0f} answers/s "
                  f"(singles {p['problems'] / p['single_wall']:.0f}/s), batch matches singles on {p['batch_matches_single']}/{p['problems']}")
    for key, unit in [("length", "rad"), ("duration", "s")]:
        shared, values = common(summaries, key)
        if not shared or any(len(v) == 0 for v in values):
            continue
        print(f"\n{key} over the {len(shared)} problems every configuration solves validly (mean, 95% CI):")
        for s, v in zip(summaries, values):
            lo, hi = bootstrap(v, rng)
            print(f"  {s['label']:<40} {v.mean():.3f} {unit} [{lo:.3f}, {hi:.3f}]  median {np.median(v):.3f}")


if __name__ == "__main__":
    main()
