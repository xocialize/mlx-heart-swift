#!/bin/bash
# gpu_window.sh — everything the HEART port needs from an IDLE GPU, in one pass (PORTING-SPEC S2b, H3, H4, H5, CAN live).
# Run only inside an announced GPU window (memory `gpu-contention-hold`); it reads the AGX utilization counter before and
# after each stage and every timing stage interleaves its arms. ~25–40 min on M5 Max.
#   ./oracle/gpu_window.sh <benchDir with *__low.png + *__reference.png> <scratchOutDir>
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd); cd "$HERE"
BENCH=${1:?benchDir}; OUT=${2:?outDir}; mkdir -p "$OUT" oracle/reports
SMOKE=.build/out/Products/Release/heart-smoke
W=oracle/weights; G=oracle/goldens
VG=/Volumes/Satechi/Development/mlxengine-forge/Tools/vosrgate/.build/release/vosrgate
util() { ioreg -r -d 1 -c AGXAccelerator | grep -o '"Device Utilization %"=[0-9]*' | grep -o '[0-9]*$'; }
idle() { echo "GPU utilization samples (10 × 1 s):"; for i in $(seq 1 10); do printf ' %s' "$(util)"; sleep 1; done; echo; }
stage() { echo; echo "════ $1 ════ $(date '+%H:%M:%S')"; }

stage "S2b · GPU parity gate, fp32 lane, TF32 OFF (the honest fp32) then default (TF32 on)"; idle
MLX_ENABLE_TF32=0 $SMOKE gate $W $G --gpu --sizes 128 > oracle/reports/s2b-gpu-fp32-notf32.log 2>&1 || true
grep -E "summary|PSNR" oracle/reports/s2b-gpu-fp32-notf32.log | tail -12
$SMOKE gate $W $G --gpu --sizes 128 > oracle/reports/s2b-gpu-fp32-tf32.log 2>&1 || true
grep -E "summary|PSNR" oracle/reports/s2b-gpu-fp32-tf32.log | tail -6

stage "S2b · fp16 lane on the GPU vs the torch fp32 oracle (128² and 512², all variants), residual fp16 then fp32"
$SMOKE gate $W $G --gpu --fp16 --only e2e > oracle/reports/h3-gpu-fp16.log 2>&1 || true
grep -E "PSNR" oracle/reports/h3-gpu-fp16.log
$SMOKE gate $W $G --gpu --fp16 --residual-fp32 --only e2e --variants fidelity,sharp > oracle/reports/h3-gpu-fp16r.log 2>&1 || true
grep -E "PSNR" oracle/reports/h3-gpu-fp16r.log

stage "S2b · eyeballs: one real image per variant, fp16 (shipping) and fp32 (TF32 off)"
IMG=$(ls "$BENCH"/*__jpeg__low.png | head -1)
for v in fidelity sharp clean clean2x; do
  $SMOKE run "$IMG" "$OUT/eyeball_${v}_fp16.png" $W --variant $v 2>&1 | tail -1
done
MLX_ENABLE_TF32=0 $SMOKE run "$IMG" "$OUT/eyeball_fidelity_fp32.png" $W --variant fidelity --fp32 2>&1 | tail -1

stage "H4 · perf bracket (arms interleaved: fp32/fp16 × head 64/40) at 256² and 512²"; idle
$SMOKE perf $W --sizes 256,512 --rounds 5 --arms fp32-64,fp32-40,fp16-64,fp16-40,fp16r-64,fp16-64-seams --json oracle/reports/h4-perf.json 2>&1 | tee oracle/reports/h4-perf.log | grep -v "^loaded"
idle

stage "H3 + H5 · the 27 bench cells: dtype arms (fp32 / fp16 / fp16 residual-fp32) and tile arms (256/384/512 × overlap 32/64)"; idle
MLX_ENABLE_TF32=0 $SMOKE bench $W "$BENCH" "$OUT/bench" \
  --arms fp32-64-whole,fp16-64-whole,fp16r-64-whole,fp16-64-t256o32,fp16-64-t384o32,fp16-64-t512o32,fp16-64-t256o64,fp16-64-t384o64,fp16-64-t512o64 \
  --json oracle/reports/h5-bench.json 2>&1 | tee oracle/reports/h5-bench.log | grep -v "^loaded"
"$VG" score "$OUT/bench" oracle/reports/h5-score.csv 2>&1 | tail -3

stage "H5 · whole-frame memory curve (fp16 and fp32) at 512², 960×540, 1280×720, 1920×1080"
$SMOKE bench $W --memory-curve --arms fp16-64-whole,fp32-64-whole 2>&1 | tee oracle/reports/h5-memory.log

stage "H5 · the ×2 route: heart_2x (clean2x) vs .fidelity ×4 + downsample — outputs for scoring"
mkdir -p "$OUT/x2"
$SMOKE bench $W "$BENCH" "$OUT/x2" --variant clean2x --arms fp16-64-whole --json oracle/reports/h5-x2-clean2x.json 2>&1 | grep -v "^loaded" | tail -3

stage "CAN · live mid-run cancel probe through the engine (1024² tiled, cancel at 1.5 s)"
$SMOKE cancel $W --after 1.5 --side 1024 2>&1 | tee oracle/reports/can-live.log | tail -2

stage "H6 · engine footprints (MLX peak / floor) at five sizes, fp16 and fp32, through the REAL engine"
for side in 256 512 1024; do
  sips -z $side $side "$IMG" --out "$OUT/in_${side}.png" > /dev/null 2>&1
done
sips -z 540 960 "$IMG" --out "$OUT/in_960x540.png" > /dev/null 2>&1
sips -z 1080 1920 "$IMG" --out "$OUT/in_1920x1080.png" > /dev/null 2>&1
for f in in_256 in_512 in_960x540 in_1024 in_1920x1080; do
  $SMOKE engine "$OUT/$f.png" "$OUT/${f}__engine_fp16.png" $W 2>&1 | tail -1 | tee -a oracle/reports/h6-footprints.log
done
for f in in_512 in_1920x1080; do
  $SMOKE engine "$OUT/$f.png" "$OUT/${f}__engine_fp32.png" $W --fp32 2>&1 | tail -1 | tee -a oracle/reports/h6-footprints.log
done
echo; echo "GPU utilization after: $(util) %"; echo "done $(date '+%H:%M:%S')"
