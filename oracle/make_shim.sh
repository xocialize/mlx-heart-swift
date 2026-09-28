#!/bin/bash
# make_shim.sh — rebuild the minimal traiNNer package HEART's heart_arch.py imports, from pinned upstream sources
# (both Apache-2.0). Nothing third-party is vendored in this repo; this script re-fetches it byte-for-byte.
#   traiNNer-redux  686293190aca3e432a42b6155dbc94ea51c4bf9f   (hat_iln_arch.py + registry.py verbatim; ONLY the iLN class, lines 21-72, of arch_util.py)
#   Phips/HEART     868878ce4c253a8061300f923b620fdc6edf090a   (heart_arch.py verbatim)
set -euo pipefail
OUT=${1:-shim}; mkdir -p "$OUT/traiNNer/archs" "$OUT/traiNNer/utils"
touch "$OUT/traiNNer/__init__.py" "$OUT/traiNNer/archs/__init__.py" "$OUT/traiNNer/utils/__init__.py"
raw() { curl -fsSL "https://raw.githubusercontent.com/the-database/traiNNer-redux/686293190aca3e432a42b6155dbc94ea51c4bf9f/$1"; }
raw traiNNer/archs/hat_iln_arch.py > "$OUT/traiNNer/archs/hat_iln_arch.py"
raw traiNNer/utils/registry.py > "$OUT/traiNNer/utils/registry.py"
{ echo "# Shim: ONLY the iLN class from traiNNer-redux traiNNer/archs/arch_util.py (686293190aca3e432a42b6155dbc94ea51c4bf9f), lines 21-72, verbatim."
  echo "import torch"; echo "from torch import nn"; echo; raw traiNNer/archs/arch_util.py | sed -n '21,72p'; } > "$OUT/traiNNer/archs/arch_util.py"
curl -fsSL "https://huggingface.co/Phips/HEART/resolve/868878ce4c253a8061300f923b620fdc6edf090a/heart_arch.py" > "$OUT/traiNNer/archs/heart_arch.py"
echo "shim -> $OUT (also needs: pip install --target <dir> --no-deps spandrel==0.4.2 onnxruntime==1.30.0)"
