#!/usr/bin/env python3
"""Analyse CST 435 sequential/OpenMP K-Means benchmark CSVs.

Examples:
  python scripts/analyze_results.py "results/raw/current/results_*.csv" --out analysis/current
  python scripts/analyze_results.py results/raw/final/results_10000_final.csv --out analysis/final

Requires matplotlib for charts: python -m pip install matplotlib
"""
from __future__ import annotations
import argparse, csv, glob, math, os, statistics
from collections import defaultdict
from pathlib import Path

NUMERIC_INT = ("dataset_size", "k", "threads", "repeat", "iterations")
NUMERIC_FLOAT = ("time_ms", "speedup", "efficiency_percent", "wcss")


def expand_files(specs):
    found=[]
    for spec in specs:
        matches=glob.glob(spec)
        found.extend(matches if matches else ([spec] if Path(spec).is_file() else []))
    result=sorted(set(str(Path(p).resolve()) for p in found))
    if not result:
        raise SystemExit("No input CSV files found. Run the benchmark suite first and pass results_*.csv.")
    return result


def load_rows(paths):
    rows=[]
    required={"implementation","dataset_size","k","threads","repeat","time_ms","iterations","wcss","converged"}
    for path in paths:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader=csv.DictReader(f)
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise SystemExit(f"{path} does not contain the expected benchmark columns.")
            for line_no, row in enumerate(reader, start=2):
                try:
                    for key in NUMERIC_INT: row[key]=int(row[key])
                    for key in NUMERIC_FLOAT: row[key]=float(row[key])
                    if not math.isfinite(row["time_ms"]) or row["time_ms"] <= 0:
                        raise ValueError("time_ms must be positive and finite")
                    if not math.isfinite(row["wcss"]): raise ValueError("WCSS must be finite")
                    row["converged"]=str(row["converged"]).strip().lower()=="true"
                    row["source_file"]=Path(path).name
                    rows.append(row)
                except (ValueError, TypeError, KeyError) as exc:
                    raise SystemExit(f"Invalid data at {path}:{line_no}: {exc}")
    if not rows: raise SystemExit("Input CSVs contain no measurement rows.")
    return rows


def med(values): return statistics.median(values)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="Benchmark CSV paths or glob patterns")
    parser.add_argument("--out", default="analysis", help="Output folder (default: analysis)")
    args=parser.parse_args()
    paths=expand_files(args.files); rows=load_rows(paths)
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # First calculate the median sequential baseline for each individual source file.
    seq_by_source=defaultdict(list)
    for r in rows:
        if r["implementation"].lower()=="sequential": seq_by_source[r["source_file"]].append(r["time_ms"])
    missing=[s for s in {r["source_file"] for r in rows} if s not in seq_by_source]
    if missing: raise SystemExit("Each benchmark file must contain sequential baseline rows: "+", ".join(missing))
    baseline={s:med(ts) for s,ts in seq_by_source.items()}

    groups=defaultdict(list)
    for r in rows:
        groups[(r["source_file"],r["dataset_size"],r["k"],r["implementation"].lower(),r["threads"])].append(r)
    summary=[]
    for (source,n,k,impl,threads), items in sorted(groups.items()):
        times=[r["time_ms"] for r in items]
        median_time=med(times); seq_med=baseline[source]
        speedup=seq_med/median_time if impl=="openmp" else 1.0
        efficiency=100.0*speedup/threads if impl=="openmp" and threads>0 else 100.0
        mean=statistics.mean(times)
        sd=statistics.stdev(times) if len(times)>1 else 0.0
        summary.append({
            "source_file":source,"dataset_size":n,"k":k,"implementation":impl,"threads":threads,
            "repeats":len(items),"median_time_ms":median_time,"mean_time_ms":mean,
            "stdev_time_ms":sd,"min_time_ms":min(times),"max_time_ms":max(times),
            "coefficient_of_variation_percent":100.0*sd/mean if mean else 0.0,
            "sequential_median_ms":seq_med,"speedup_vs_sequential_median":speedup,
            "efficiency_vs_sequential_median_percent":efficiency,
            "median_wcss":med([r["wcss"] for r in items]),
            "convergence_rate_percent":100.0*sum(r["converged"] for r in items)/len(items),
            "median_iterations":med([r["iterations"] for r in items]),
            "points_per_second":1000.0*n/median_time,
        })
    fields=list(summary[0].keys())
    with open(out/"summary.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summary)

    # A compact table suitable for direct inclusion in the report.
    report_fields=["source_file","dataset_size","implementation","threads","repeats","median_time_ms","stdev_time_ms","speedup_vs_sequential_median","efficiency_vs_sequential_median_percent","median_wcss","convergence_rate_percent"]
    with open(out/"report_table.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=report_fields); w.writeheader(); w.writerows([{k:r[k] for k in report_fields} for r in summary])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Wrote {out/'summary.csv'} and {out/'report_table.csv'}.")
        print("Install chart dependency with: python -m pip install matplotlib")
        return

    # One set of plots per workload avoids misleading comparisons across sizes.
    for source in sorted({r["source_file"] for r in summary}):
        items=[r for r in summary if r["source_file"]==source]
        n=items[0]["dataset_size"]
        parallel=sorted([r for r in items if r["implementation"]=="openmp"],key=lambda r:r["threads"])
        seq=[r for r in items if r["implementation"]=="sequential"]
        if not parallel: continue
        stem=Path(source).stem+f"_{n}rows"
        xs=[r["threads"] for r in parallel]
        plt.figure(figsize=(7.5,5)); plt.plot(xs,[r["median_time_ms"] for r in parallel],marker="o",label="OpenMP median")
        if seq: plt.axhline(seq[0]["median_time_ms"],linestyle="--",label="Sequential median")
        plt.xlabel("OpenMP thread count"); plt.ylabel("Median execution time (ms)"); plt.title(f"Runtime vs thread count (N={n:,})"); plt.xticks(xs); plt.grid(True,alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(out/f"{stem}_runtime.png",dpi=180); plt.close()
        plt.figure(figsize=(7.5,5)); plt.plot(xs,[r["speedup_vs_sequential_median"] for r in parallel],marker="o",label="Measured speedup")
        plt.plot(xs,xs,linestyle="--",label="Ideal linear speedup"); plt.axhline(1,linewidth=.8)
        plt.xlabel("OpenMP thread count"); plt.ylabel("Speedup (×)"); plt.title(f"Speedup vs thread count (N={n:,})"); plt.xticks(xs); plt.grid(True,alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(out/f"{stem}_speedup.png",dpi=180); plt.close()
        plt.figure(figsize=(7.5,5)); plt.plot(xs,[r["efficiency_vs_sequential_median_percent"] for r in parallel],marker="o",label="Measured efficiency")
        plt.axhline(100,linestyle="--",label="Ideal efficiency"); plt.xlabel("OpenMP thread count"); plt.ylabel("Efficiency (%)"); plt.title(f"Parallel efficiency (N={n:,})"); plt.xticks(xs); plt.grid(True,alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(out/f"{stem}_efficiency.png",dpi=180); plt.close()
        plt.figure(figsize=(7.5,5)); plt.bar([str(r["threads"]) for r in parallel],[r["coefficient_of_variation_percent"] for r in parallel])
        plt.xlabel("OpenMP thread count"); plt.ylabel("Timing variability (CV, %)"); plt.title(f"Run-to-run timing variability (N={n:,})"); plt.grid(True,axis="y",alpha=.3); plt.tight_layout(); plt.savefig(out/f"{stem}_variability.png",dpi=180); plt.close()

    # Cross-size scalability plot, one line per thread count.
    sizes=sorted({r["dataset_size"] for r in summary})
    thread_values=sorted({r["threads"] for r in summary if r["implementation"]=="openmp"})
    if len(sizes)>1 and thread_values:
        plt.figure(figsize=(8,5.5))
        for t in thread_values:
            chosen=[]
            for n in sizes:
                candidates=[r for r in summary if r["implementation"]=="openmp" and r["threads"]==t and r["dataset_size"]==n]
                if candidates: chosen.append((n,candidates[0]["median_time_ms"]))
            if chosen: plt.plot([x[0] for x in chosen],[x[1] for x in chosen],marker="o",label=f"{t} thread(s)")
        plt.xlabel("Number of points"); plt.ylabel("Median execution time (ms)"); plt.title("Workload scalability"); plt.xscale("log"); plt.yscale("log"); plt.grid(True,which="both",alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(out/"scalability_runtime.png",dpi=180); plt.close()
    print(f"Loaded {len(rows)} measurements from {len(paths)} file(s).")
    print(f"Summary: {(out/'summary.csv').resolve()}")
    print(f"Report table: {(out/'report_table.csv').resolve()}")
    print(f"Charts: {out.resolve()}")
    print("Use medians and timing variability in the report. Explain slowdowns honestly; do not selectively omit measurements.")

if __name__=="__main__": main()
