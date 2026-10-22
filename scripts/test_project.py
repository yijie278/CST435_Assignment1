#!/usr/bin/env python3
"""Integration tests for the CST 435 sequential/OpenMP K-Means project.

Run: python test_project.py
Requires GCC/G++ with OpenMP support.
"""
import csv, math, os, subprocess, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/'src'/'main.cpp'
EXE=ROOT/'build'/('ParallelKMeans_test.exe' if os.name=='nt' else 'ParallelKMeans_test')

def run(*args, expect_ok=True):
    p=subprocess.run([str(EXE),*map(str,args)],cwd=ROOT,text=True,capture_output=True)
    if expect_ok and p.returncode:
        raise AssertionError(f'Command failed: {args}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}')
    if not expect_ok and p.returncode==0:
        raise AssertionError(f'Command unexpectedly succeeded: {args}\n{p.stdout}')
    return p

def read_csv(path):
    with open(path,newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def main():
    try:
        EXE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['g++','-std=c++17','-O2','-fopenmp',str(SOURCE),'-o',str(EXE)],check=True,capture_output=True,text=True)
    except FileNotFoundError: raise SystemExit('g++ not found. Add the MSYS2 UCRT64 bin directory to PATH, then retry.')
    except subprocess.CalledProcessError as e: raise SystemExit('Compilation failed:\n'+e.stderr)
    try:
        with tempfile.TemporaryDirectory(prefix='kmeans_test_') as temp:
            td=Path(temp)
            out=run('--synthetic','--points',12000,'--k',5,'--threads','1,2,4','--repeats',2,'--seed',42,
                    '--results',td/'bench.csv','--clusters',td/'clusters.csv','--centroids',td/'centroids.csv').stdout
            if 'OpenMP: enabled' not in out: raise AssertionError('OpenMP disabled; parallel tests cannot be trusted.')
            rows=read_csv(td/'bench.csv')
            if len(rows)!=8: raise AssertionError(f'Expected 8 benchmark rows; got {len(rows)}')
            seq=[r for r in rows if r['implementation']=='sequential']; omp=[r for r in rows if r['implementation']=='openmp']
            if len(seq)!=2 or len(omp)!=6: raise AssertionError('Unexpected measurement row counts.')
            if any(abs(float(r['speedup'])-1)>1e-9 or abs(float(r['efficiency_percent'])-100)>1e-9 for r in seq):
                raise AssertionError('Sequential baseline should report speedup=1 and efficiency=100%.')
            for r in rows:
                for key in ('time_ms','wcss','speedup','efficiency_percent'):
                    if not math.isfinite(float(r[key])): raise AssertionError(f'Non-finite {key} value: {r}')
                if float(r['time_ms'])<=0: raise AssertionError('Runtime must be positive.')
                if r['converged'].lower() not in ('true','false'): raise AssertionError('Invalid convergence flag.')
            wc=[float(r['wcss']) for r in rows]
            if max(wc)-min(wc)>max(1e-5,max(wc)*1e-5): raise AssertionError(f'Sequential/OpenMP WCSS mismatch: {wc}')
            clusters=read_csv(td/'clusters.csv')
            if len(clusters)!=12000 or 'cluster' not in clusters[0]: raise AssertionError('Cluster output row count/column is wrong.')
            labels={int(r['cluster']) for r in clusters}
            if not labels.issubset(set(range(5))): raise AssertionError('Cluster labels outside 0..K-1.')
            centers=read_csv(td/'centroids.csv')
            if len(centers)!=5: raise AssertionError('Expected K centroid rows.')
            for row in centers:
                for key,value in row.items():
                    if key!='cluster' and not math.isfinite(float(value)): raise AssertionError('Non-finite centroid.')
            # Real CSV header and feature selection test.
            sample=td/'sample.csv'
            sample.write_text('CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)\n1,Male,19,15,39\n2,Female,21,16,81\n3,Female,20,17,6\n4,Male,23,18,77\n5,Female,31,19,40\n6,Male,22,20,76\n',encoding='utf-8')
            run('--input',sample,'--features','Annual Income (k$),Spending Score (1-100)','--k',2,'--threads','1,2','--repeats',1,
                '--results',td/'real.csv','--clusters',td/'real_clusters.csv','--centroids',td/'real_centroids.csv')
            real=read_csv(td/'real_clusters.csv')
            if len(real)!=6: raise AssertionError('CSV feature selection did not preserve all sample rows.')
            # Invalid K should fail clearly rather than silently producing bad output.
            run('--synthetic','--points',5,'--k',6,'--threads','1','--repeats',1,expect_ok=False)
        print('PASS: OpenMP enabled; repeated measurements; finite timing/WCSS; sequential/OpenMP objective agreement;')
        print('      cluster labels and row counts; finite centroids; real-CSV feature selection; invalid K rejection.')
    finally:
        EXE.unlink(missing_ok=True)

if __name__=='__main__': main()
