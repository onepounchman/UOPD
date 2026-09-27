#!/usr/bin/env bash
#SBATCH --job-name=uopd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=uopd-%j.log
set -euo pipefail
# Activate the installed Python environment before sbatch; it is inherited.
# Usage: sbatch [cluster options] scripts/train.slurm webshop 3b uopd /shared/run-dir
env_name=${1:?environment required}
model_size=${2:?student size required}
method=${3:?method required}
output=${4:?absolute output directory required}
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD:${WEBSHOP_ROOT:-}:${PYTHONPATH:-}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES
# On an exclusive allocation this head belongs to this job. Do not stop another
# user's Ray service: refuse to start when the chosen port is already occupied.
ray_port=${UOPD_RAY_PORT:-6379}
node_ip=$(hostname -I | awk '{print $1}')
ray start --head --node-ip-address="$node_ip" --port="$ray_port" --num-gpus=8
trap 'ray stop' EXIT
export RAY_ADDRESS="$node_ip:$ray_port"
python scripts/run.py train --env "$env_name" --size "$model_size" --method "$method" --output "$output"
