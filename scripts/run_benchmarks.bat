@echo off
setlocal
cd /d "%~dp0\.."

set THREADS=1,2,4,8
set K=5
set REPEATS=10

if not exist results\raw\current mkdir results\raw\current
if not exist results\clusters\current mkdir results\clusters\current
if not exist results\centroids\current mkdir results\centroids\current

build\ParallelKMeans.exe --synthetic --points 10000 --k %K% --threads %THREADS% --repeats %REPEATS% --results results\raw\current\results_10000.csv --clusters results\clusters\current\clusters_10000.csv --centroids results\centroids\current\centroids_10000.csv
if errorlevel 1 goto :failed
build\ParallelKMeans.exe --synthetic --points 100000 --k %K% --threads %THREADS% --repeats %REPEATS% --results results\raw\current\results_100000.csv --clusters results\clusters\current\clusters_100000.csv --centroids results\centroids\current\centroids_100000.csv
if errorlevel 1 goto :failed
build\ParallelKMeans.exe --synthetic --points 500000 --k %K% --threads %THREADS% --repeats %REPEATS% --results results\raw\current\results_500000.csv --clusters results\clusters\current\clusters_500000.csv --centroids results\centroids\current\centroids_500000.csv
if errorlevel 1 goto :failed

echo.
echo Benchmarks completed. If Python and matplotlib are installed, run:
echo python scripts\analyze_results.py "results\raw\current\results_*.csv" --out analysis\current
exit /b 0

:failed
echo Benchmark failed. Check the error above.
exit /b 1
