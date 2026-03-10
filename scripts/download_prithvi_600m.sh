#!/bin/bash
# Pre-download Prithvi-EO v2 600M-TL weights to the HuggingFace cache
# so SLURM compute nodes can load them without internet access.
#
# Run ONCE on the login node before submitting 600m jobs:
#   bash scripts/download_prithvi_600m.sh

set -euo pipefail

module load eth_proxy

cd "$(dirname "$0")/.."
echo "Downloading Prithvi-EO-2.0-600M-TL weights..."
uv run python - <<'EOF'
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL",
    filename="Prithvi_EO_V2_600M_TL.pt",
)
print(f"Weights cached at: {path}")
EOF
