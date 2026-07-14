#!/bin/bash
# Full H100 benchmark battery for LiquidGEMM. Run detached: nohup bash bench/h100_run.sh &
set -x
cd /root/LiquidGEMM
export PATH=/root/.local/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda PYTHONPATH=/root/LiquidGEMM
set -a; source .env; set +a           # HF_TOKEN for gated Llama
source /root/venv/bin/activate
# meta-llama repo is gated and this token isn't authorized -> ungated mirror (same weights)
MODEL=NousResearch/Meta-Llama-3.1-8B-Instruct

echo "=========== 1. KERNEL BENCH (RS vs bf16 vs int8-wgmma) ==========="
python bench/rs_bench.py 2>&1

echo "=========== 2. SMOKE Llama-3.1-8B w4-RS ==========="
LIQUIDGEMM_W4=1 python bench/vllm_smoke.py --model $MODEL 2>&1 | grep -E "OUTPUT:|Error|Traceback"

echo "=========== 3. SERVING Llama-3.1-8B (cuda graphs) ==========="
for mode in w4 int8 bf16; do
  for b in 1 30; do
    case $mode in
      w4)   env LIQUIDGEMM_W4=1 python bench/vllm_serving.py --model $MODEL --quant liquidgemm --batch $b --out-len 128 --eager 0 2>&1 | grep -E "RESULT|Model loading took" | sed "s/^/[w4] /";;
      int8) env LIQUIDGEMM_W4=0 python bench/vllm_serving.py --model $MODEL --quant liquidgemm --batch $b --out-len 128 --eager 0 2>&1 | grep -E "RESULT|Model loading took" | sed "s/^/[int8] /";;
      bf16) python bench/vllm_serving.py --model $MODEL --quant none --batch $b --out-len 128 --eager 0 2>&1 | grep -E "RESULT|Model loading took" | sed "s/^/[bf16] /";;
    esac
  done
done

echo "=========== 4. ACCURACY Llama-3.1-8B (wikitext2 ppl) ==========="
uv pip install -q datasets accelerate
python bench/accuracy.py --model $MODEL --max-samples 40 2>&1 | tail -12

echo "H100_RUN_COMPLETE"
