#!/bin/bash
# Deploy Sleeping LLM on a Vast.ai H100 instance
#
# Usage:
#   1. Create a Vast.ai instance (H100 80GB, PyTorch template)
#   2. SSH in: ssh -p PORT root@HOST
#   3. Clone repo: git clone <repo-url> /workspace/j && cd /workspace/j
#   4. Run: bash deploy-h100.sh
#   5. Access UI: http://HOST:8000 (or tunnel via ssh -L 8000:localhost:8000)

set -euo pipefail

echo "=== Sleeping LLM — H100 Deploy ==="

# 1. Install dependencies
echo "[1/4] Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements-torch.txt

# 2. Accept Llama license + download model
echo "[2/4] Downloading Llama-3.1-70B-Instruct..."
echo "NOTE: You need a HuggingFace token with Llama access."
echo "      Run: huggingface-cli login"
echo ""

# Check if already logged in
if ! python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    echo "Not logged in to HuggingFace. Run:"
    echo "  huggingface-cli login"
    echo "Then re-run this script."
    exit 1
fi

# Pre-download model (so first startup isn't slow)
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_id = 'meta-llama/Llama-3.1-70B-Instruct'
print(f'Downloading tokenizer for {model_id}...')
AutoTokenizer.from_pretrained(model_id)
print('Downloading model (4-bit)...')
AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    ),
    device_map='auto',
    low_cpu_mem_usage=True,
)
print('Model downloaded and verified.')
"

# 3. Create data directories
echo "[3/4] Setting up data directories..."
mkdir -p data/conversations data/core_identity data/benchmarks data/memit data/adapters models/fused

# 4. Start
echo "[4/4] Starting web UI on port 8000..."
echo ""
echo "Access via: http://$(hostname -I | awk '{print $1}'):8000"
echo "Or tunnel:  ssh -L 8000:localhost:8000 root@<host> -p <port>"
echo ""
python3 -m src.main --web --port 8000 --config config-70b.yaml
