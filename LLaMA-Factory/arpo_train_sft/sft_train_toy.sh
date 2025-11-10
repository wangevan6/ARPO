#!/bin/bash

#================== Basic Configuration ==================#
# Use only first GPU for toy testing
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd):$PYTHONPATH
export USE_LIBUV=0
# Disable Weights & Biases
export WANDB_DISABLED=true

#================== Training Parameter Configuration ==================#
# Single GPU training configuration
NNODES=1                 # Single node
NODE_RANK=0              # Node rank 0
PROC_PER_NODE=1          # Single GPU
MASTER_ADDR="127.0.0.1"  # Master address
MASTER_PORT=29501        # Different port to avoid conflicts

# Output directory for toy model
OUTPUT_DIR="./checkpoints/qwen0.5b_toy"
# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Path to the training script
TRAIN_SCRIPT="../src/llamafactory/launcher.py"

# Path to the toy training configuration file
TRAIN_ARGS="yaml/qwen_toy.yaml"

echo "=========================================="
echo "Starting TOY SFT Training (Local Testing)"
echo "=========================================="
echo "Model: Qwen2.5-0.5B-Instruct"
echo "Dataset: arpo_sft_toy (500 samples)"
echo "GPUs: 1 (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "Output: ${OUTPUT_DIR}"
echo "Config: ${TRAIN_ARGS}"
echo "=========================================="
echo ""

# Command to launch training - ensure USE_LIBUV=0 is properly set
# For single GPU training, use direct Python execution with environment variable
USE_LIBUV=0 PYTHONPATH=$(pwd):$PYTHONPATH python ${TRAIN_SCRIPT} ${TRAIN_ARGS} 2>&1 | tee ${OUTPUT_DIR}/training.log

echo ""
echo "=========================================="
echo "Toy training completed!"
echo "Check logs at: ${OUTPUT_DIR}/training.log"
echo "Checkpoints saved to: ${OUTPUT_DIR}/"
echo "=========================================="
