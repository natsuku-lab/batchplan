"""The comparison's tables, regenerated from saved runs (`run_comparison.sh`, which runs and judges them).

Each argument is a configuration's directory of rounds (`<part>/<config>/round<k>`) or a single run
directory. Rounds and seeds are pooled:
- success: the share of queries a planner claims, and the share whose final output the judge accepts
  (and accepts with the kinodynamic limits loosened to 10%), with the reasons for the rest;
- time per query, failures included at what they cost: end-to-end median, p95 and p99, and the median
  without loading the scene (search, post-processing, timing and the pipeline's own validation);
- throughput: judge-valid results per second of each pass's wall time (a pass is every problem once
  with one seed), median and range over passes;
- against the first configuration, on the problems both solve validly: time per query (end-to-end),
  path length and trajectory duration, as per-problem means over valid outputs, paired, with 95%
  bootstrap intervals of the mean ratio (resampled by problem). Time is compared as the ratio of the
  per-problem medians' geometric mean.

    python scripts/report.py <config_dir> [<config_dir> ...] [--markdown] [--per-set]
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

KINODYNAMIC = {"velocity", "acceleration", "jerk"}
PLANNING = ["search", "post", "plan", "search+post", "search+post+timing+validation", "timing", "validation"]


def runs_of(path):
    path = Path(path)
    rounds = sorted(path.glob("round*"))
    return rounds if rounds else [path]


def label(manifest):
    s = manifest.get("settings", {})
    extra = s.get("mode") or ("Ruckig" if s.get("smoothing") == "ruckig" else "")
    family = "" if manifest["family"] == "defaults" else "matched"
    return " ".join(x for x in [manifest["planner"], manifest["workload"], extra, family] if x)


def valid_within(v, tolerance):
    """A verdict re-read with the kinodynamic limits loosened to `tolerance`."""
    if v is None:
        return False
    if v["valid"]:
        return True
    reasons = {r.strip() for r in v["reason"].split(",") if r.strip()}
    return reasons <= KINODYNAMIC and all(v.get(f"{r}_ratio", 0) <= 1 + tolerance for r in reasons)


def summarize(path):
    queries, passes, manifest = [], [], None
    for run in runs_of(path):
        manifest = json.loads((run / "manifest.json").read_text())
        verdicts = {}
        if (run / "verdicts.jsonl").exists():
            for l in open(run / "verdicts.jsonl"):
                v = json.loads(l)
                verdicts[(v["problem"], v["seed"])] = v
        records = [json.loads(l) for l in open(run / "records.jsonl")]
        valid_in_pass = Counter()
        for r in records:
            if r.get("api", "single") != "single" or r["rep"] != 1:
                continue
            v = verdicts.get((r["problem"], r["seed"])) if r["status"] == "solved" else None
            phases = r.get("phases", {})
            planning = sum(phases.get(k, 0.0) for k in PLANNING)
            queries.append({"problem": r["problem"], "seed": r["seed"], "round": run.name, "status": r["status"],
                            "verdict": v, "total": phases.get("total", 0.0), "planning": planning})
            valid_in_pass[(run.name, r["seed"])] += bool(v and v["valid"])
        for p in map(json.loads, open(run / "passes.jsonl")):
            passes.append({**p, "round": run.name, "valid": valid_in_pass[(run.name, p["seed"])]})
    q = len(queries)
    totals = np.array([x["total"] for x in queries]) * 1e3
    planning = np.array([x["planning"] for x in queries if x["status"] == "solved"]) * 1e3
    rates = [p["valid"] / p["wall"] for p in passes]
    return {
        "label": label(manifest), "manifest": manifest, "queries": queries, "n": q,
        "claimed": sum(x["status"] == "solved" for x in queries) / q,
        "valid": sum(bool(x["verdict"] and x["verdict"]["valid"]) for x in queries) / q,
        "valid10": sum(valid_within(x["verdict"], 0.10) for x in queries) / q,
        "reasons": Counter(x["verdict"]["reason"] for x in queries if x["verdict"] and not x["verdict"]["valid"]),
        "median": float(np.median(totals)), "p95": float(np.percentile(totals, 95)),
        "p99": float(np.percentile(totals, 99)),
        "planning": float(np.median(planning)) if len(planning) else float("nan"),
        "rate": float(np.median(rates)), "rate_range": (float(min(rates)), float(max(rates))),
        "passes": passes,
    }


def per_problem(summary, key):
    """Each problem's mean of `key` over its valid outputs (`time`: the median end-to-end time)."""
    values = defaultdict(list)
    for x in summary["queries"]:
        v = x["verdict"]
        if v and v["valid"]:
            if key == "time":
                values[x["problem"]].append(x["total"])
            elif key in v:
                values[x["problem"]].append(v[key])
    pick = np.median if key == "time" else np.mean
    return {p: float(pick(v)) for p, v in values.items()}


def paired_ratio(base, other, key, rng, n=4000):
    """Mean of other/base per problem both solve validly, with a 95% bootstrap interval."""
    a, b = per_problem(base, key), per_problem(other, key)
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    ratios = np.array([b[p] / a[p] for p in shared])
    if key == "time":
        # Times span orders of magnitude: the geometric mean of the ratios.
        logs = np.log(ratios)
        means = [np.exp(rng.choice(logs, len(logs)).mean()) for _ in range(n)]
        centre = float(np.exp(logs.mean()))
    else:
        means = [rng.choice(ratios, len(ratios)).mean() for _ in range(n)]
        centre = float(ratios.mean())
    return len(shared), centre, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), \
        float(np.median([a[p] for p in shared])), float(np.median([b[p] for p in shared]))


def per_set(summaries):
    sets = sorted({x["problem"].rsplit("/", 1)[0] for s in summaries for x in s["queries"]})
    print(f"\n{'set':<28}" + "".join(f"{s['label'][:16]:>17}" for s in summaries))
    for name in sets:
        row = []
        for s in summaries:
            xs = [x for x in s["queries"] if x["problem"].rsplit("/", 1)[0] == name]
            row.append(f"{100 * sum(bool(x['verdict'] and x['verdict']['valid']) for x in xs) / max(1, len(xs)):>16.1f}%")
        print(f"{name:<28}" + "".join(row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--per-set", action="store_true")
    args = parser.parse_args()
    summaries = [summarize(c) for c in args.configs]
    rng = np.random.default_rng(0)
    m = summaries[0]["manifest"]
    print(f"{m['split']} split, {m.get('set', 'problems')}, {m.get('scene', 'exact')} scenes, "
          f"{m['workers']} core{'s' if m['workers'] > 1 else ''}; queries = problems x seeds x rounds")
    head = ["configuration", "queries", "claimed", "valid", "valid ≤10%", "median ms", "p95 ms", "p99 ms",
            "without setup ms", "valid/s (range)"]
    rows = []
    for s in summaries:
        lo, hi = s["rate_range"]
        rows.append([s["label"], str(s["n"]), f"{100 * s['claimed']:.1f}%", f"{100 * s['valid']:.1f}%",
                     f"{100 * s['valid10']:.1f}%", f"{s['median']:.2f}", f"{s['p95']:.1f}", f"{s['p99']:.1f}",
                     f"{s['planning']:.2f}", f"{s['rate']:.0f} ({lo:.0f}–{hi:.0f})"])
    if args.markdown:
        print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
        for r in rows:
            print("| " + " | ".join(r) + " |")
    else:
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)]
        print("  ".join(h.rjust(w) if i else h.ljust(w) for i, (h, w) in enumerate(zip(head, widths))))
        for r in rows:
            print("  ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(r, widths))))
    for s in summaries:
        if s["reasons"]:
            print(f"  {s['label']}: invalid " + ", ".join(f"{r} {n}" for r, n in s["reasons"].most_common(6)))
        batch = [p for p in s["passes"] if "single_wall" in p]
        if batch:
            print(f"  {s['label']}: native batch {np.median([p['problems'] / p['wall'] for p in batch]):.0f} answers/s, "
                  f"one query at a time {np.median([p['problems'] / p['single_wall'] for p in batch]):.0f}/s; "
                  f"batch identical to singles on {sum(p['batch_matches_single'] for p in batch)}/{sum(p['problems'] for p in batch)}")
    base = summaries[0]
    for key, unit in [("time", "s"), ("length", "rad"), ("duration", "s")]:
        lines = []
        for s in summaries[1:]:
            r = paired_ratio(base, s, key, rng)
            if r:
                n, mean, lo, hi, a, b = r
                scale = 1e3 if key == "time" else 1.0
                shown = "ms" if key == "time" else unit
                lines.append(f"  {s['label']:<28} {mean:.3f}x [{lo:.3f}, {hi:.3f}] on {n} problems "
                             f"(medians {scale * a:.3f} {shown} for {base['label']}, {scale * b:.3f} {shown})")
        if lines:
            print(f"\n{key}, relative to {base['label']} on the problems both solve validly (mean ratio, 95% CI):")
            print("\n".join(lines))
    if args.per_set:
        per_set(summaries)


if __name__ == "__main__":
    main()
