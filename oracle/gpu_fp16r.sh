#!/bin/bash
# gpu_fp16r.sh — re-measure the residual-stream-fp32 posture after the cast fix (H3): e2e vs the oracle, the perf
# arms, and the whole-frame bench on the 27 cells with SSIMULACRA2. Idle GPU required.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd); cd "$HERE"
BENCH=${1:?benchDir}; OUT=${2:?outDir}; mkdir -p "$OUT"
SMOKE=.build/out/Products/Release/heart-smoke; W=oracle/weights; G=oracle/goldens
VG=/Volumes/Satechi/Development/mlxengine-forge/Tools/vosrgate/.build/release/vosrgate
util() { ioreg -r -d 1 -c AGXAccelerator | grep -o '"Device Utilization %"=[0-9]*' | grep -o '[0-9]*$'; }
echo "GPU util before: $(util) %"
echo "── e2e fp16 residual-fp32 (fixed) vs the torch fp32 oracle"
$SMOKE gate $W $G --gpu --fp16 --residual-fp32 --only e2e > oracle/reports/h3-gpu-fp16r.log 2>&1 || true
grep PSNR oracle/reports/h3-gpu-fp16r.log | cut -c1-200
echo "── perf: fp16-64 vs fp16r-64 (5 rounds, interleaved)"
$SMOKE perf $W --sizes 256,512 --rounds 5 --arms fp16-64,fp16r-64 --json oracle/reports/h4-perf-fp16r.json 2>&1 | tee oracle/reports/h4-perf-fp16r.log | grep -v "^loaded"
echo "── bench: whole-frame arms on the 27 cells"
MLX_ENABLE_TF32=0 $SMOKE bench $W "$BENCH" "$OUT" --arms fp32-64-whole,fp16-64-whole,fp16r-64-whole --json oracle/reports/h3-bench-fp16r.json 2>&1 | tee oracle/reports/h3-bench-fp16r.log | grep -v "^loaded" | tail -6
"$VG" score "$OUT" oracle/reports/h3-score-fp16r.csv 2>&1 | tail -1
echo "GPU util after: $(util) %"; echo "done $(date '+%H:%M:%S')"
