#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source .venv/bin/activate
export CUDA_HOME=/afs/math/software/nvidia/cuda/13.0.1
python src/benchmark.py