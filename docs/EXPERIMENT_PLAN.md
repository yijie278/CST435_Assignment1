# CST 435 Experimental Plan

## Research question

How do dataset size and OpenMP thread count affect K-Means execution time, speedup, parallel efficiency and clustering correctness on a shared-memory multicore CPU?

## Workloads

| Workload | Source | Points | Purpose |
|---|---|---:|---|
| Real application | Mall Customers | 200 | Demonstrate customer segmentation |
| Small | Synthetic | 10,000 | Observe parallel overhead |
| Medium | Synthetic | 100,000 | Measure moderate scaling |
| Large | Synthetic | 500,000 | Evaluate scalability |

Use K=5, the same seed, tolerance and iteration limit for matching comparisons. Test sequential execution and OpenMP with 1, 2, 4 and 8 threads where supported.

## Protocol

- Run 10 repetitions for every performance configuration.
- Use median execution time as the primary measurement.
- Calculate each member's speedup and efficiency using that member's own sequential median.
- Calculate arithmetic means across the four members after member-level metrics are available.
- Record CPU, core count, operating system, compiler, compiler flags and OpenMP status.
- Keep raw result CSVs unchanged.

## Metrics

```text
speedup = median sequential runtime / median parallel runtime
efficiency = speedup / thread count * 100
CV = sample standard deviation / mean runtime * 100
```

WCSS is calculated on standardised points against final centroids. Sequential and OpenMP WCSS should agree within the correctness tolerance.
