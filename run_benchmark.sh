#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --array=0-999
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

source .venv/bin/activate
export CUDA_HOME=/afs/math/software/nvidia/cuda/13.0.1

DATA_DIR="data/MIPLIB2017_benchmark_set"

mapfile -t INSTANCES < <(find "$DATA_DIR" -maxdepth 1 -type f | sort)
NUM_INSTANCES=${#INSTANCES[@]}

if [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_INSTANCES" ]; then
    echo "No instance for array task $SLURM_ARRAY_TASK_ID; found only $NUM_INSTANCES instances."
    exit 0
fi

INSTANCE="${INSTANCES[$SLURM_ARRAY_TASK_ID]}"
echo "Running benchmark for instance: $INSTANCE"

python src/benchmark.py "$INSTANCE"
