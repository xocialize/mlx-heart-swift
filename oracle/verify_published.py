#!/usr/bin/env python
"""verify_published.py — the published-artifact check (the VOSR2 way, model-registry.md):
1. every LFS file's Hub sha256 == the staged file's sha256 (both lanes, all files);
2. a FRESH download of one variant per lane into hub-verify/ is byte-identical to the staged file;
3. (the caller then runs `heart-smoke keys hub-verify/<lane>` and an engine run on the downloaded bytes vs the staged
   run — see the spec's H7 row).

    PYTHONPATH=pydeps python verify_published.py --weights weights --out hub-verify
"""
import argparse, hashlib, os, shutil
from huggingface_hub import HfApi, hf_hub_download
p = argparse.ArgumentParser(); p.add_argument("--weights", required=True); p.add_argument("--out", required=True); a = p.parse_args()
LANES = {"fp16": "mlx-community/HEART-fp16", "fp32": "mlx-community/HEART-fp32"}
STEMS = ["heart_4x_otf_v2", "heart_4x_otf_gan", "heart_4x_pretrain", "heart_2x"]
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
api = HfApi(); ok = True
for lane, repo in LANES.items():
    info = api.model_info(repo, files_metadata=True)
    hub = {s.rfilename: (s.lfs.sha256 if s.lfs else None, s.size) for s in info.siblings}
    for f in [f"{s}_{lane}.safetensors" for s in STEMS] + ["config.json", "README.md"]:
        staged = os.path.join(a.weights, f)
        if f == "README.md": print(f"  {repo}/{f}: present {f in hub}"); ok &= f in hub; continue
        local = sha(staged); remote = hub.get(f, (None, None))[0]
        same = (remote == local) if remote else (hub.get(f, (None, None))[1] == os.path.getsize(staged))
        print(f"  {repo}/{f}: hub {'sha256 ' + remote[:12] if remote else 'size ' + str(hub.get(f, (None, None))[1])} == staged {'sha256 ' + local[:12]} → {'OK' if same else 'MISMATCH'}")
        ok &= same
    # fresh download of the default variant + config into out/<lane>/
    dst = os.path.join(a.out, lane); shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
    for f in [f"heart_4x_otf_v2_{lane}.safetensors", "config.json"]:
        got = hf_hub_download(repo, f, cache_dir=os.path.join(a.out, "cache"), force_download=True)
        shutil.copyfile(got, os.path.join(dst, f))
        same = sha(os.path.join(dst, f)) == sha(os.path.join(a.weights, f))
        print(f"  fresh download {repo}/{f} → {dst}: byte-identical to staged {'OK' if same else 'MISMATCH'}")
        ok &= same
print("PUBLISHED-ARTIFACT CHECK:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
