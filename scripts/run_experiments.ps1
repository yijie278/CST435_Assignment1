$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$exe = Join-Path $root 'build\ParallelKMeans.exe'
$currentRaw = Join-Path $root 'results\raw\current'
$currentClusters = Join-Path $root 'results\clusters\current'
$currentCentroids = Join-Path $root 'results\centroids\current'
$currentAnalysis = Join-Path $root 'analysis\current'

New-Item -ItemType Directory -Force -Path `
    $currentRaw, $currentClusters, $currentCentroids, $currentAnalysis |
    Out-Null

Write-Host "=== CST 435 Parallel K-Means experiment workflow ===" -ForegroundColor Cyan

if (-not (Get-Command g++ -ErrorAction SilentlyContinue)) {
    $gccPath = 'C:\msys64\ucrt64\bin'
    if (Test-Path (Join-Path $gccPath 'g++.exe')) {
        $env:Path = "$gccPath;$env:Path"
    }
}
if (-not (Get-Command g++ -ErrorAction SilentlyContinue)) {
    throw 'g++ not found. Add C:\msys64\ucrt64\bin to PATH.'
}
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw 'Python not found. Install Python and enable Add Python to PATH.'
}
$python = 'C:\Python315\python.exe'
if (-not (Test-Path $python)) {
    $python = (Get-Command python).Source
}

Write-Host "[1/5] Compiling C++17 with OpenMP..." -ForegroundColor Yellow
g++ -std=c++17 -O2 -fopenmp src\main.cpp -o $exe
if ($LASTEXITCODE -ne 0) {
    throw 'Compilation failed.'
}

Write-Host "[2/5] Running correctness tests..." -ForegroundColor Yellow
& $python scripts\test_project.py
if ($LASTEXITCODE -ne 0) {
    throw 'Automated tests failed; fix these before benchmarking.'
}

$realData = Join-Path $root 'data\Mall_Customers.csv'
if (Test-Path $realData) {
    Write-Host "[3/5] Running real Mall Customers experiment..." -ForegroundColor Yellow
    & $exe --input $realData `
        --features "Annual Income (k$),Spending Score (1-100)" `
        --k 5 --threads 1,2,4 --repeats 10 --seed 2026 `
        --results "$currentRaw\results_mall.csv" `
        --clusters "$currentClusters\clustered_customers.csv" `
        --centroids "$currentCentroids\centroids_mall.csv"
    if ($LASTEXITCODE -ne 0) {
        throw 'Real dataset run failed.'
    }
} else {
    Write-Warning 'Mall_Customers.csv not found. Skipping real-data run.'
}

Write-Host "[4/5] Running synthetic scalability experiments..." -ForegroundColor Yellow
foreach ($n in @(10000, 100000, 500000)) {
    Write-Host "--- N=$n ---" -ForegroundColor Cyan
    & $exe --synthetic --points $n --k 5 --threads 1,2,4,8 `
        --repeats 10 --seed 2026 `
        --results "$currentRaw\results_$($n).csv" `
        --clusters "$currentClusters\clusters_$($n).csv" `
        --centroids "$currentCentroids\centroids_$($n).csv"
    if ($LASTEXITCODE -ne 0) {
        throw "Synthetic benchmark failed for N=$n."
    }
}

Write-Host "[5/5] Analysing results and creating charts..." -ForegroundColor Yellow
& $python -m pip show matplotlib *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Installing matplotlib for charts...' -ForegroundColor Yellow
    & $python -m pip install matplotlib
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not install matplotlib.'
    }
}

& $python scripts\analyze_results.py "$currentRaw\results_*.csv" `
    --out $currentAnalysis
if ($LASTEXITCODE -ne 0) {
    throw 'Analysis failed.'
}

$clusteredCustomers = Join-Path $currentClusters 'clustered_customers.csv'
if (Test-Path $clusteredCustomers) {
    & $python scripts\plot_clusters.py $clusteredCustomers `
        --x 'Annual Income (k$)' `
        --y 'Spending Score (1-100)' `
        --out "$currentAnalysis\customer_clusters.png"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'Cluster plot failed.'
    }
}

Write-Host "Completed. Open analysis\current for new tables and charts." `
    -ForegroundColor Green
