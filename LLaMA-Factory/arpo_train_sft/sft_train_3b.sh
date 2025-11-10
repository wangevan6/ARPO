#!/bin/bash

#================== Basic Configuration ==================#
# Use all 8 GPUs for production training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$(pwd):$PYTHONPATH
export USE_LIBUV=0

# Disable Weights & Biases (or set WANDB_API_KEY for logging)
export WANDB_DISABLED=true

#================== Training Parameter Configuration ==================#
# Multi-GPU distributed training configuration
NNODES=1                 # Single node with 8 GPUs
NODE_RANK=0              # Node rank 0
PROC_PER_NODE=8          # 8 GPUs
MASTER_ADDR="127.0.0.1"  # Master address
MASTER_PORT=29500        # Master port

# Output directory for production model
OUTPUT_DIR="<your_output_dir>"  # e.g., /data/checkpoints/qwen2.5-3b-sft
# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Path to the training script
TRAIN_SCRIPT="../src/llamafactory/launcher.py"

# Path to the production training configuration file
TRAIN_ARGS="yaml/qwen_3b.yaml"

echo "=================================================="
echo "Starting PRODUCTION SFT Training (8 GPUs)"
echo "=================================================="
echo "Model: Qwen2.5-3B-Instruct"
echo "Dataset: ARPO-SFT-54K (full dataset)"
echo "GPUs: 8 (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "Output: ${OUTPUT_DIR}"
echo "Config: ${TRAIN_ARGS}"
echo "Effective Batch Size: 1 (per_device) × 2 (grad_accum) × 8 (GPUs) = 16"
echo "Expected Duration: 8-16 hours"
echo "=================================================="
echo ""

# Print GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Command to launch training
torchrun --nnodes ${NNODES} \
         --node_rank ${NODE_RANK} \
         --nproc_per_node ${PROC_PER_NODE} \
         --master_addr ${MASTER_ADDR} \
         --master_port ${MASTER_PORT} \
         ${TRAIN_SCRIPT} \
         ${TRAIN_ARGS} 2>&1 | tee ${OUTPUT_DIR}/training.log

echo ""
echo "=================================================="
echo "Production training completed!"
echo "Check logs at: ${OUTPUT_DIR}/training.log"
echo "Checkpoints saved to: ${OUTPUT_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Verify final checkpoint exists"
echo "  2. Test model inference"
echo "  3. Use checkpoint for ARPO/AEPO RL training"
echo "=================================================="
