#!/usr/bin/env python3
"""Audit the supplied, archived NPU log. Does not execute an NPU."""
import argparse
import collections
import csv
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMATS = ('E4M3FN', 'E5M2')
PLANES = {'E4M3FN': 3, 'E5M2': 5}
EXP = {'E4M3FN': 9, 'E5M2': 16}
MAX = {'E4M3FN': 229376, 'E5M2': 3758096384}
KS = (256, 1024, 4096, 8192)
PAT = re.compile(r'^(E4M3FN|E5M2) x (E4M3FN|E5M2) K=(\d+) pattern=(\d+) tiles=(\d+) partial_mismatch=(\d+)/(\d+) final_mismatch=(\d+)/(\d+) exponent=(-\d+) transfer_launch_seconds=([\d.]+) checked_wall_seconds=([\d.]+) (PASS|FAIL)\n  C\[0,0\] exact = (-?\d+) \* 2\^\((-\d+)\)', re.M)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def audit(path):
    text = path.read_text()
    rows = []
    keys = set()
    require(text.count('NPU device 0; max 20 Cube blocks per launch; CPU decode/reconstruction') == 4, 'expected four NPU run headers')
    require(text.count('FINAL: PASS (all checked integer results exact)') == 4, 'expected four successful runs')
    require(text.count('decode E4M3FN finite=254 special_rejected=2 mismatch=0') == 4, 'E4M3FN decode count')
    require(text.count('decode E5M2 finite=248 special_rejected=8 mismatch=0') == 4, 'E5M2 decode count')
    for a,b,k,p,tiles,pm,pc,fm,fc,e,t,w,status,coef,ce in PAT.findall(text):
        k,p,tiles,pm,pc,fm,fc,e,coef,ce = map(int,(k,p,tiles,pm,pc,fm,fc,e,coef,ce))
        key = (k,a,b,p)
        require(key not in keys, f'duplicate: {key}')
        keys.add(key)
        require(tiles == ((k+31)//32)*PLANES[a]*PLANES[b], f'tile count: {key}')
        require(pc == tiles*1024 and fc == 1024, f'element count: {key}')
        require(pm == fm == 0 and status == 'PASS', f'mismatch: {key}')
        require(e == ce == -EXP[a]-EXP[b], f'exponent: {key}')
        # Independently validate printed deterministic scalar samples.
        if p == 1:
            require(coef == k*MAX[a]*MAX[b], f'maximum-input coefficient: {key}')
        if p == 2:
            expected = (k//3) + (MAX[a]*MAX[b] if k%3 else 0) + (1 if k%3 == 2 else 0)
            require(coef == expected, f'cancellation coefficient: {key}')
        if p == 3:
            require(coef == 0, f'zero coefficient: {key}')
        require(0 < float(t) <= float(w), f'timing: {key}')
        rows.append(dict(K=k, A=a, B=b, pattern=p, tiles=tiles, partial_mismatch=pm, partial_checked=pc, final_mismatch=fm, final_checked=fc, exponent=e, transfer_launch_seconds=float(t), checked_wall_seconds=float(w), status=status, c00_coefficient=coef))
    expected_keys = {(k,a,b,p) for k in KS for a in FORMATS for b in FORMATS for p in range(4)}
    require(keys == expected_keys, 'missing or unexpected test cases')
    require(len(re.findall(r'^E[45]M[23](?:FN)? x ',text,re.M)) == len(rows), 'unparsed test lines')
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('log',type=Path,nargs='?',default=ROOT/'results/npu_device0_2026-11-25.log')
    p.add_argument('--csv',type=Path)
    args=p.parse_args()
    rows=audit(args.log)
    if args.csv:
        with args.csv.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Archived log audit (not a new hardware run)')
    for k in KS:
        group=[r for r in rows if r['K']==k]
        print(f'K={k:4d} cases={len(group)} partial={sum(r["partial_checked"] for r in group):,} final={sum(r["final_checked"] for r in group):,} mismatch=0')
    print(f'TOTAL cases={len(rows)} partial={sum(r["partial_checked"] for r in rows):,} final={sum(r["final_checked"] for r in rows):,} mismatch=0')
    print('Median timings (all four patterns; transfer+launch ms / checked wall ms)')
    for k in KS:
        for a in FORMATS:
            for b in FORMATS:
                g=[r for r in rows if (r['K'],r['A'],r['B'])==(k,a,b)]
                print(f'{k:4d} {a:7s} x {b:7s}: {1000*statistics.median(r["transfer_launch_seconds"] for r in g):.4f} / {1000*statistics.median(r["checked_wall_seconds"] for r in g):.4f}')
    print('AUDIT PASS')

if __name__=='__main__':
    main()
