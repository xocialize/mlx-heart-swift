#!/usr/bin/env python
"""upload_weights.py — publish the converted HEART weight lanes to mlx-community.

    python upload_weights.py --weights <dir> [--lanes fp16,fp32] [--dry-run]

⚠️ OUTWARD ACTION (public model repos). mlx-community grammar: `<UpstreamRepoName>-<quant>`, one repo per lane, every
variant's file inside so a package pulls only the checkpoint it uses:
  mlx-community/HEART-fp16  heart_{4x_otf_v2,4x_otf_gan,4x_pretrain,2x}_fp16.safetensors + config.json  (the shipping lane)
  mlx-community/HEART-fp32  the same four files in fp32 + config.json                                 (the parity lane)
Each repo gets the model card from MODEL_CARD.md with the lane filled in. Verify after upload with verify_published.py.
"""
import argparse, os
p = argparse.ArgumentParser(); p.add_argument("--weights", required=True); p.add_argument("--lanes", default="fp16,fp32")
p.add_argument("--dry-run", action="store_true"); a = p.parse_args()
LANES = {"fp16": "mlx-community/HEART-fp16", "fp32": "mlx-community/HEART-fp32"}
STEMS = ["heart_4x_otf_v2", "heart_4x_otf_gan", "heart_4x_pretrain", "heart_2x"]
here = os.path.dirname(os.path.abspath(__file__))
card_t = open(os.path.join(here, "MODEL_CARD.md")).read()
for lane in a.lanes.split(","):
    repo = LANES[lane]
    files = [f"{s}_{lane}.safetensors" for s in STEMS] + ["config.json"]
    for f in files:
        path = os.path.join(a.weights, f); assert os.path.exists(path), path
        print(f"  {repo}: {f:36} {os.path.getsize(path)/1e6:7.1f} MB")
    card = card_t.replace("{{LANE}}", lane).replace("{{REPO}}", repo)
    if a.dry_run: continue
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo)
    for f in files:
        print("uploading", repo, f, flush=True)
        api.upload_file(path_or_fileobj=os.path.join(a.weights, f), path_in_repo=f, repo_id=repo)
    print("done:", f"https://huggingface.co/{repo}", flush=True)
if a.dry_run: print("dry run — nothing uploaded")
