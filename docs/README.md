# CST 435: Sequential vs Parallel K-Means

This project compares sequential C++17 K-Means with a shared-memory OpenMP implementation. It includes a real customer-segmentation demonstration, synthetic scalability workloads, correctness tests and report-ready analysis.

## Structure

```text
src/main.cpp                 K-Means implementation and CLI
scripts/                     Tests, benchmark runners and Python analysis
data/                        Real input data
results/raw/                 Preserved benchmark CSVs
results/clusters/            Cluster assignment exports
results/centroids/           Centroid exports
analysis/                    Generated tables and charts
docs/                        Methodology and report documentation
build/                       Compiled executables
```

`results/raw/final/` contains the preserved ten-repeat synthetic benchmark set. `results/raw/preliminary/` contains earlier three-repeat measurements. Do not overwrite either set.

## Build and test

From the project root:

```powershell
$env:Path = "C:\msys64\ucrt64\bin;$env:Path"
g++ -std=c++17 -O2 -fopenmp src\main.cpp -o build\ParallelKMeans.exe
python scripts\test_project.py
```

## Run the workflow

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_experiments.ps1
```

New measurements are written under `results/raw/current/`, exports under the current result folders and charts under `analysis/current/`, protecting preserved evidence.

The synthetic procedure uses 10,000, 100,000 and 500,000 points, K=5, OpenMP thread counts 1, 2, 4 and 8, and 10 repeats per configuration.

Speedup is calculated from median times:

```text
speedup = median sequential time / median OpenMP time
efficiency = speedup / thread count * 100
```

Keep real-data segmentation separate from synthetic performance testing.
