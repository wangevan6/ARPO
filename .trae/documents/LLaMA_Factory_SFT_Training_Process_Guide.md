# Complete ARPO/AEPO RL Training Guide

## From Local Toy Testing to Production Deployment

**Target**: Train ARPO and AEPO on Qwen2.5-7B with Reasoning dataset (10K samples)

**Strategy**: Local toy test (16GB GPU) → Cluster production (8x 40GB GPUs)

***

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [STAGE 1: Local Toy RL Training](#stage-1-local-toy-rl-training)
3. [STAGE 2: Cluster ARPO Production](#stage-2-cluster-arpo-production)
4. [STAGE 3: Cluster AEPO Production](#stage-3-cluster-aepo-production)
5. [STAGE 4: Verification](#stage-4-verification)
6. [Troubleshooting](#troubleshooting)

***

## Prerequisites

### Hardware

* **Local**: 1 GPU with 16GB VRAM (WSL2 on Windows)

* **Cluster**: 8x NVIDIA A100 40GB GPUs (Linux)

### Software

* WSL2 with CUDA support (local)

* Conda/Miniconda

* Git with LFS

### Required Accounts

* HuggingFace account (for model downloads)

* Bright Data account (for Bing Search API)

* WandB account (optional, for training monitoring)

### Assumptions

* SFT training already completed

* SFT checkpoints available (or using base models)

***

## STAGE 1: Local Toy RL Training

**Goal**: Verify the entire RL pipeline works before deploying to cluster

**Duration**: 1-2 hours

**Strategy**: Try 3B model first, fallback to 0.5B if OOM

### 1.1 Environment Setup

#### Step 1: Create Conda Environment

```bash
# Navigate to ARPO directory
cd C:/Users/user/Projects/ARPO/ARPO

# Create conda environment
conda create -n arpo python==3.10 -y
conda activate arpo

# Verify Python version
python --version  # Should be 3.10.x
```

#### Step 2: Install PyTorch (CUDA 12.8)

```bash
# Install PyTorch with CUDA 12.8 (per CLAUDE.md requirements)
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# Verify CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

Expected output:

```
CUDA Available: True, GPUs: 1
```

#### Step 3: Install Flash Attention

```bash
pip install flash-attn --no-build-isolation
```

This may take 5-10 minutes to compile.

#### Step 4: Install RL Dependencies

```bash
# Make sure you're in ARPO/ARPO directory
pip install -r requirements.txt
```

#### Step 5: Install VERL Framework

```bash
cd verl_arpo_entropy
pip install -e .
cd ..

# Verify installation
python -c "import verl; print('VERL installed successfully')"
```

#### Step 6: Test Ray

```bash
# Start Ray cluster
ray start --head --port=6379

# Verify
python -c "import ray; ray.init(address='auto'); print(ray.cluster_resources())"

# Stop Ray
ray stop
```

### 1.2 Download Models

#### Download 3B Model (Try First)

```bash
# Create models directory
mkdir -p models

# Download Qwen2.5-3B-Instruct
hf download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/Qwen2.5-3B-Instruct

# Verify download
ls -lh ./models/Qwen2.5-3B-Instruct/
# Should see: config.json, model.safetensors, tokenizer files
```

#### Download 0.5B Model (Fallback)

```bash
# Download Qwen2.5-0.5B-Instruct as backup
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./models/Qwen2.5-0.5B-Instruct

# Verify
ls -lh ./models/Qwen2.5-0.5B-Instruct/
```

### 1.3 Prepare Toy Dataset

#### Download Full Dataset

```bash
# Create dataset directory
mkdir -p rl_datasets

# Download ARPO-RL-Reasoning-10K dataset
hf download dongguanting/ARPO-RL-Reasoning-10K train_10k.parquet \
  --repo-type dataset \
  --local-dir ./temp_rl

# Move to rl_datasets
mv temp_rl/train_10k.parquet rl_datasets/train.parquet

# Download validation set
hf download dongguanting/ARPO-RL-Reasoning-10K test.parquet \
  --repo-type dataset \
  --local-dir ./temp_rl

mv temp_rl/test.parquet rl_datasets/valid.parquet

# Clean up
rm -rf temp_rl
```

#### Create Toy Subset (100 Samples)

Create a Python script to generate toy dataset:

**File**: `scripts/create_toy_dataset.py`

```python
import pandas as pd
import sys

def create_toy_dataset(input_file, output_file, num_samples=100):
    """Create a toy dataset with limited samples for testing."""
    print(f"Reading dataset from {input_file}...")
    df = pd.read_parquet(input_file)

    print(f"Original dataset size: {len(df)} samples")

    # Take first N samples
    df_toy = df.head(num_samples)

    print(f"Toy dataset size: {len(df_toy)} samples")

    # Save to parquet
    df_toy.to_parquet(output_file, index=False)

    print(f"Toy dataset saved to {output_file}")

    # Display sample
    print("\nSample prompt:")
    print(df_toy.iloc[0]['prompt'][:200] + "...")
    print(f"\nColumns: {df_toy.columns.tolist()}")

if __name__ == "__main__":
    create_toy_dataset(
        input_file='rl_datasets/train.parquet',
        output_file='rl_datasets/train_toy.parquet',
        num_samples=100
    )
```

Run the script:

```bash
python scripts/create_toy_dataset.py
```

Expected output:

```
Reading dataset from rl_datasets/train.parquet...
Original dataset size: 10000 samples
Toy dataset size: 100 samples
Toy dataset saved to rl_datasets/train_toy.parquet
```

### 1.4 Configure Search API

#### Step 1: Get Bright Data Credentials

1. Sign up at <https://brightdata.com/>
2. Navigate to **Zones** → **Web Scraper API**
3. Create a new zone
4. Note down:

   * **API Key** (e.g., `brd-customer-abc-zone-xyz`)

   * **Zone name** (e.g., `residential`)

#### Step 2: Create Search Cache

```bash
mkdir -p search_cache
echo '{}' > search_cache/search_cache.json
chmod 644 search_cache/search_cache.json
```

#### Step 3: Update Configuration Files

You need to update API keys in **3 locations**:

**Location 1:** **`scripts/config/ppo_trainer.yaml`**

Find the search tool section:

```yaml
actor_rollout_ref:
  rollout:
    tools:
      tool_instances:
        search:
          class_path: verl.workers.rollout.tools.search_tool.BingSearchTool
          params:
            api_key: YOUR_API_KEY_HERE        # ← UPDATE THIS
            zone: YOUR_ZONE_HERE                # ← UPDATE THIS
            max_results: 10
            result_length: 1000
            location: us
            cache_file: search_cache/search_cache.json
```

**Location 2:** **`verl_arpo_entropy/verl/workers/rollout/tools/config_example.yaml`**

Copy and edit:

```bash
cp verl_arpo_entropy/verl/workers/rollout/tools/config_example.yaml \
   verl_arpo_entropy/verl/workers/rollout/tools/config.yaml

nano verl_arpo_entropy/verl/workers/rollout/tools/config.yaml
```

Update:

```yaml
tools:
  tool_instances:
    python:
      class_path: verl.workers.rollout.tools.python_tool.PythonTool
      params:
        conda_path: /home/user/anaconda3         # ← UPDATE with your conda path
        conda_env: arpo
        timeout: 60

    search:
      class_path: verl.workers.rollout.tools.search_tool.BingSearchTool
      params:
        api_key: YOUR_API_KEY_HERE               # ← UPDATE THIS
        zone: YOUR_ZONE_HERE                      # ← UPDATE THIS
        max_results: 10
        result_length: 1000
        location: us
        cache_file: search_cache/search_cache.json
```

**Location 3: Training script** (we'll create this next)

#### Find Your Conda Path

```bash
which conda
# Example output: /home/user/anaconda3/bin/conda
# Use: /home/user/anaconda3 (without /bin/conda)
```

Or:

```bash
echo $CONDA_PREFIX
# While conda env is activated
```

### 1.5 Create Toy ARPO Training Scripts

#### Script 1: 3B Model (Try First)

**File**: `scripts/ARPO_3B_Toy_Local.sh`

```bash
#!/bin/bash

# ============================================================================
# ARPO Local Toy Training Script - Qwen2.5-3B
# Purpose: Test RL pipeline on local 16GB GPU before cluster deployment
# Memory Strategy: Aggressive optimization to fit 3B model in 16GB
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"
echo "Working directory: $PARENT_DIR"

# ============ Environment Variables ============
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VERL_LOGGING_LEVEL=INFO
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export RAY_memory_usage_threshold=0.9
export RAY_memory_monitor_refresh_ms=0

# Python path
export PYTHONPATH="${PARENT_DIR}/verl_arpo_entropy:$PYTHONPATH"

# ============ Basic Configuration ============
PROJECT_NAME="local_toy_test"
EXPERIMENT_NAME="ARPO_3B_toy_local"

CONFIG_PATH="${PARENT_DIR}/scripts/config"
CONFIG_NAME="ppo_trainer.yaml"

# ============ Hardware Settings ============
NNODES=1
N_GPUS_PER_NODE=1  # Single GPU

# ============ Data Configuration ============
PROMPT_KEY="prompt"
TRAIN_BATCH_SIZE=4          # Very small for 16GB
PPO_MINI_BATCH_SIZE=2       # Minimal
MAX_PROMPT_LENGTH=512       # Short sequences
MAX_RESPONSE_LENGTH=1024    # Short responses

TRAIN_FILES="${PARENT_DIR}/rl_datasets/train_toy.parquet"
VALID_FILES="${PARENT_DIR}/rl_datasets/train_toy.parquet"  # Use same for validation

# ============ Model Configuration ============
ACTOR_MODEL_PATH="${PARENT_DIR}/models/Qwen2.5-3B-Instruct"

# ============ Rollout Configuration ============
ROLLOUT_NAME="vllm"
ROLLOUT_MODE="sync_with_tool"
ROLLOUT_N=4              # Minimal rollouts
INITIAL_ROLLOUTS=2       # Start with 2
BEAM_SIZE=2              # Branch to 2
BRANCH_PROBABILITY=0.5
ENTROPY_WEIGHT=0.2

# ============ Tool Configuration ============
SEARCH_CACHE_PATH="${PARENT_DIR}/search_cache/search_cache.json"

# UPDATE THIS with your Bright Data credentials
BING_API_KEY="YOUR_API_KEY_HERE"
BING_ZONE="YOUR_ZONE_HERE"

# UPDATE THIS with your conda path
CONDA_PATH="/home/user/anaconda3"  # Change to your conda installation

# ============ Reward Configuration ============
REWARD_MANAGER="naive"
CUSTOM_REWARD_FUNCTION_PATH="${PARENT_DIR}/verl_arpo_entropy/verl/utils/reward_score/math.py"
CUSTOM_REWARD_FUNCTION_NAME="compute_score"

# ============ Training Configuration ============
TOTAL_EPOCHS=1  # Just 1 epoch for testing
SAVE_FREQ=5
TEST_FREQ=5

# ============ Paths ============
SAVE_PATH="${PARENT_DIR}/checkpoints_toy/${EXPERIMENT_NAME}"
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"

# ============ WandB (Optional) ============
WANDB_API_KEY=""  # Leave empty to disable

# ============ Preparation ============
if [ "$WANDB_API_KEY" != "" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

mkdir -p $SAVE_PATH
mkdir -p $ROLLOUT_SAVE_PATH

echo "============================================"
echo "ARPO 3B Local Toy Training"
echo "============================================"
echo "Model: ${ACTOR_MODEL_PATH}"
echo "Dataset: ${TRAIN_FILES}"
echo "Batch Size: ${TRAIN_BATCH_SIZE}"
echo "Rollout Budget: ${ROLLOUT_N} (initial: ${INITIAL_ROLLOUTS})"
echo "Save Path: ${SAVE_PATH}"
echo "============================================"
echo

# ============ Start Training ============
python3 -m verl.trainer.main_ppo \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VALID_FILES} \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.initial_rollouts=${INITIAL_ROLLOUTS} \
    actor_rollout_ref.rollout.beam_size=${BEAM_SIZE} \
    actor_rollout_ref.rollout.branch_probability=${BRANCH_PROBABILITY} \
    actor_rollout_ref.rollout.entropy_weight=${ENTROPY_WEIGHT} \
    actor_rollout_ref.rollout.tools.tool_instances.search.params.cache_file=${SEARCH_CACHE_PATH} \
    actor_rollout_ref.rollout.tools.tool_instances.search.params.api_key=${BING_API_KEY} \
    actor_rollout_ref.rollout.tools.tool_instances.search.params.zone=${BING_ZONE} \
    actor_rollout_ref.rollout.tools.tool_instances.python.params.conda_path=${CONDA_PATH} \
    actor_rollout_ref.rollout.tools.tool_instances.python.params.conda_env=arpo \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=${REWARD_MANAGER} \
    custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH} \
    custom_reward_function.name=${CUSTOM_REWARD_FUNCTION_NAME} \
    trainer.critic_warmup=0 \
    trainer.logger="[console]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_PATH} \
    hydra.run.dir=${SAVE_PATH}/outputs 2>&1 | tee ${SAVE_PATH}/run.log

echo
echo "============================================"
echo "Training Complete!"
echo "Logs: ${SAVE_PATH}/run.log"
echo "Checkpoints: ${SAVE_PATH}"
echo "============================================"
```

#### Script 2: 0.5B Model (Fallback if 3B OOMs)

**File**: `scripts/ARPO_0.5B_Toy_Local.sh`

```bash
#!/bin/bash

# ============================================================================
# ARPO Local Toy Training Script - Qwen2.5-0.5B (FALLBACK)
# Purpose: Fallback if 3B model doesn't fit in 16GB
# Memory Strategy: Relaxed settings since 0.5B is small
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"
echo "Working directory: $PARENT_DIR"

# ============ Environment Variables ============
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VERL_LOGGING_LEVEL=INFO
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export RAY_memory_usage_threshold=0.9
export RAY_memory_monitor_refresh_ms=0

export PYTHONPATH="${PARENT_DIR}/verl_arpo_entropy:$PYTHONPATH"

# ============ Basic Configuration ============
PROJECT_NAME="local_toy_test"
EXPERIMENT_NAME="ARPO_0.5B_toy_local_fallback"

CONFIG_PATH="${PARENT_DIR}/scripts/config"
CONFIG_NAME="ppo_trainer.yaml"

# ============ Hardware Settings ============
NNODES=1
N_GPUS_PER_NODE=1

# ============ Data Configuration ============
PROMPT_KEY="prompt"
TRAIN_BATCH_SIZE=8          # Can afford larger with 0.5B
PPO_MINI_BATCH_SIZE=4
MAX_PROMPT_LENGTH=1024      # Longer sequences OK
MAX_RESPONSE_LENGTH=2048

TRAIN_FILES="${PARENT_DIR}/rl_datasets/train_toy.parquet"
VALID_FILES="${PARENT_DIR}/rl_datasets/train_toy.parquet"

# ============ Model Configuration ============
ACTOR_MODEL_PATH="${PARENT_DIR}/models/Qwen2.5-0.5B-Instruct"  # 0.5B model

# ============ Rollout Configuration ============
ROLLOUT_NAME="vllm"
ROLLOUT_MODE="sync_with_tool"
ROLLOUT_N=6              # More rollouts possible
INITIAL_ROLLOUTS=3
BEAM_SIZE=2
BRANCH_PROBABILITY=0.5
ENTROPY_WEIGHT=0.2

# ============ Tool Configuration ============
SEARCH_CACHE_PATH="${PARENT_DIR}/search_cache/search_cache.json"

# UPDATE THESE
BING_API_KEY="YOUR_API_KEY_HERE"
BING_ZONE="YOUR_ZONE_HERE"
CONDA_PATH="/home/user/anaconda3"

# ============ Reward Configuration ============
REWARD_MANAGER="naive"
CUSTOM_REWARD_FUNCTION_PATH="${PARENT_DIR}/verl_arpo_entropy/verl/utils/reward_score/math.py"
CUSTOM_REWARD_FUNCTION_NAME="compute_score"

# ============ Training Configuration ============
TOTAL_EPOCHS=1
SAVE_FREQ=5
TEST_FREQ=5

# ============ Paths ============
SAVE_PATH="${PARENT_DIR}/checkpoints_toy/${EXPERIMENT_NAME}"
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"

WANDB_API_KEY=""

# ============ Preparation ============
if [ "$WANDB_API_KEY" != "" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

mkdir -p $SAVE_PATH
mkdir -p $ROLLOUT_SAVE_PATH

echo "============================================"
echo "ARPO 0.5B Local Toy Training (FALLBACK)"
echo "============================================"
echo "Model: ${ACTOR_MODEL_PATH}"
echo "Dataset: ${TRAIN_FILES}"
echo "Batch Size: ${TRAIN_BATCH_SIZE}"
echo "Rollout Budget: ${ROLLOUT_N} (initial: ${INITIAL_ROLLOUTS})"
echo "Save Path: ${SAVE_PATH}"
echo "============================================"
echo

# ============ Start Training ============
python3 -m verl.trainer.main_ppo \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VALID_FILES} \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.initial_rollouts=${INITIAL_ROLLOUTS} \
    actor_rollout_ref.rollout.beam_size=${BEAM_SIZE} \
    actor_rollout_ref.rollout.branch_probability=${BRANCH_PROBABILITY} \
    actor_rollout_ref.rollout.entropy_weight=${ENTROPY_WEIGHT} \
    actor_rollout_ref.rollout.tools.tool_instances.search.params.cache_file=${SEARCH_CACHE_PATH} \
    actor_rollout_ref.rollout.tools.tool_instances.search.params.api_key=${BING_API_KEY} \
    actor_rollout_ref.rollout.tools.tool_instances.search.params.zone=${BING_ZONE} \
    actor_rollout_ref.rollout.tools.tool_instances.python.params.conda_path=${CONDA_PATH} \
    actor_rollout_ref.rollout.tools.tool_instances.python.params.conda_env=arpo \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    reward_model.reward_manager=${REWARD_MANAGER} \
    custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH} \
    custom_reward_function.name=${CUSTOM_REWARD_FUNCTION_NAME} \
    trainer.critic_warmup=0 \
    trainer.logger="[console]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_PATH} \
    hydra.run.dir=${SAVE_PATH}/outputs 2>&1 | tee ${SAVE_PATH}/run.log

echo
echo "============================================"
echo "Training Complete!"
echo "Logs: ${SAVE_PATH}/run.log"
echo "Checkpoints: ${SAVE_PATH}"
echo "============================================"
```

### 1.6 Update Configuration Files

Before running, you MUST update the API keys and conda path in the scripts:

```bash
# Edit 3B script
nano scripts/ARPO_3B_Toy_Local.sh
# Update:
# - BING_API_KEY
# - BING_ZONE
# - CONDA_PATH

# Edit 0.5B script
nano scripts/ARPO_0.5B_Toy_Local.sh
# Update same variables
```

Make scripts executable:

```bash
chmod +x scripts/ARPO_3B_Toy_Local.sh
chmod +x scripts/ARPO_0.5B_Toy_Local.sh
```

### 1.7 Run Local Toy Training

#### Attempt 1: Try 3B Model

```bash
# Navigate to ARPO directory
cd C:/Users/user/Projects/ARPO/ARPO

# Activate environment
conda activate arpo

# Run 3B training
bash scripts/ARPO_3B_Toy_Local.sh
```

**Monitor GPU Memory:**

In another terminal:

```bash
watch -n 1 nvidia-smi
```

**Watch for OOM Indicators:**

❌ **Out of Memory Errors**:

```
RuntimeError: CUDA out of memory
RuntimeError: vLLM OOM
Process killed
```

✅ **Success Indicators**:

```
[INFO] Initializing Ray cluster...
[INFO] Loading model from ./models/Qwen2.5-3B-Instruct
[INFO] Creating Actor, Rollout, Ref workers...
[INFO] Starting training for 1 epochs...
Epoch 0 | Step 1 | Reward: 0.XX | Loss: 0.XXX
```

#### Attempt 2: Fallback to 0.5B (If 3B OOMs)

If you see OOM errors, immediately:

```bash
# Stop training (Ctrl+C if still running)
# Clean up Ray
ray stop --force

# Run 0.5B fallback
bash scripts/ARPO_0.5B_Toy_Local.sh
```

Expected: Should complete in 20-40 minutes

### 1.8 Verification Checklist

After training completes, verify:

```bash
# Check logs
tail -50 checkpoints_toy/ARPO_*/run.log

# Check checkpoints
ls -lh checkpoints_toy/ARPO_*/global_step_*/

# Check rollout data
ls -lh checkpoints_toy/ARPO_*/rollout/
```

**Success Criteria:**

* ✅ Ray cluster initialized successfully

* ✅ Model loaded without errors

* ✅ Rollout generated responses with `<python>` and/or `<search>` tags

* ✅ Python tool executed code successfully

* ✅ Search tool returned results or cache hits

* ✅ Reward function computed scores (values between 0-1)

* ✅ Policy update completed without OOM

* ✅ Checkpoint saved

* ✅ Training log shows progress through 1 epoch

* ✅ No crashes or hangs

**Record which model worked:**

* [ ] 3B model worked

* [ ] 0.5B fallback used

This information will help configure cluster training.

***

## STAGE 2: Cluster ARPO Production

**Goal**: Train production ARPO model with Qwen2.5-7B on 8x A100 40GB GPUs

**Duration**: 4-6 hours

**Dataset**: Full 10K Reasoning dataset

### 2.1 Transfer to Cluster

#### Step 1: SSH to Cluster

```bash
ssh user@your-cluster-ip
```

#### Step 2: Clone Repository

```bash
cd ~
git clone https://github.com/dongguanting/ARPO.git
cd ARPO/ARPO
```

#### Step 3: Setup Environment

```bash
# Create conda environment
conda create -n arpo python==3.10 -y
conda activate arpo

# Install PyTorch (CUDA 12.4 for cluster)
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install flash-attn
pip3 install flash-attn --no-build-isolation

# Install requirements
pip install -r requirements.txt

# Install VERL
cd verl_arpo_entropy
pip install -e .
cd ..
```

#### Step 4: Verify 8 GPUs

```bash
nvidia-smi
# Should show 8 GPUs with ~40GB memory each

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
# Should output: CUDA: True, GPUs: 8
```

#### Step 5: Transfer Configuration

Copy your verified API keys and settings from local machine:

```bash
# On local machine, copy API key and zone
# Transfer via SCP or manually edit on cluster
```

### 2.2 Download Production Model

Choose one:

**Option A: Use Your SFT Checkpoint**

```bash
# If you have an SFT checkpoint from earlier training
ACTOR_MODEL_PATH=/path/to/your/sft/checkpoint
```

**Option B: Use Base 7B Model**

```bash
# Download Qwen2.5-7B-Instruct
mkdir -p /data/models
hf download Qwen/Qwen2.5-7B-Instruct --local-dir /data/models/Qwen2.5-7B-Instruct

# Verify
ls -lh /data/models/Qwen2.5-7B-Instruct/
```

### 2.3 Download Full Dataset

```bash
cd ARPO/ARPO

# Download full dataset
hf download dongguanting/ARPO-RL-Reasoning-10K \
  --repo-type dataset \
  --local-dir ./rl_datasets

# Verify files
ls -lh rl_datasets/
# Should have:
# - train_10k.parquet (10,000 samples)
# - test.parquet (300 samples)
```

### 2.4 Create Search Cache

```bash
mkdir -p search_cache
echo '{}' > search_cache/search_cache.json
```

### 2.5 Create Production ARPO Script

**File**: `scripts/ARPO_7B_Production_Cluster.sh`

```bash
#!/bin/bash

# ============================================================================
# ARPO Production Training Script - Qwen2.5-7B
# Environment: 8x A100 40GB GPUs
# Dataset: 10K Reasoning samples
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"
echo "Working directory: $PARENT_DIR"

# ============ Environment Variables ============
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VERL_LOGGING_LEVEL=INFO
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export RAY_memory_usage_threshold=0.8
export RAY_memory_monitor_refresh_ms=0

export PYTHONPATH="${PARENT_DIR}/verl_arpo_entropy:$PYTHONPATH"

# ============ Basic Configuration ============
PROJECT_NAME="reasoning_tasks"
EXPERIMENT_NAME="ARPO_Qwen2.5_7B_Reasoning"

CONFIG_PATH="${PARENT_DIR}/scripts/config"  # MUST be absolute path
CONFIG_NAME="ppo_trainer.yaml"

# ============ Hardware Settings ============
NNODES=1
N_GPUS_PER_NODE=8  # 8 GPUs

# ============ Data Configuration ============
PROMPT_KEY="prompt"
TRAIN_BATCH_SIZE=128        # Production batch size
PPO_MINI_BATCH_SIZE=16
MAX_PROMPT_LENGTH=1536      # Production sequence lengths
MAX_RESPONSE_LENGTH=4096

TRAIN_FILES="${PARENT_DIR}/rl_datasets/train_10k.parquet"
VALID_FILES="${PARENT_DIR}/rl_datasets/test.parquet"

# ============ Model Configuration ============
# UPDATE THIS: Point to your SFT checkpoint or base 7B model
ACTOR_MODEL_PATH="/data/models/Qwen2.5-7B-Instruct"  # ← CHANGE THIS

# ============ Rollout Configuration ============
ROLLOUT_NAME="vllm"
ROLLOUT_MODE="sync_with_tool"
ROLLOUT_N=16                # Full rollout budget
INITIAL_ROLLOUTS=8
BEAM_SIZE=2
BRANCH_PROBABILITY=0.5
ENTROPY_WEIGHT=0.2

# ============ Tool Configuration ============
SEARCH_CACHE_PATH="${PARENT_DIR}/search_cache/search_cache.json"

# UPDATE THESE with your Bright Data credentials
BING_API_KEY="YOUR_API_KEY_HERE"  # ← CHANGE THIS
BING_ZONE="YOUR_ZONE_HERE"         # ← CHANGE THIS

# UPDATE THIS with cluster conda path
CONDA_PATH="/home/user/anaconda3"  # ← CHANGE THIS

# ============ Reward Configuration ============
REWARD_MANAGER="naive"
CUSTOM_REWARD_FUNCTION_PATH="${PARENT_DIR}/verl_arpo_entropy/verl/utils/reward_score/math.py"
CUSTOM_REWARD_FUNCTION_NAME="compute_score"

# ============ Training Configuration ============
TOTAL_EPOCHS=2  # Production: 2 epochs
SAVE_FREQ=5     # Save every 5 steps
TEST_FREQ=5     # Validate every 5 steps

# ============ Paths ============
SAVE_PATH="/data/checkpoints/${EXPERIMENT_NAME}"  # ← CHANGE THIS if needed
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"

# ============ WandB Configuration ============
WANDB_API_KEY=""  # ← ADD YOUR WANDB KEY (optional)

# ============ Preparation ============
if [ "$WANDB_API_KEY" != "" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

mkdir -p $SAVE_PATH
mkdir -p $ROLLOUT_SAVE_PATH

echo "============================================"
echo "ARPO Production Training - Qwen2.5-7B"
echo "============================================"
echo "Model: ${ACTOR_MODEL_PATH}"
echo "Dataset: ${TRAIN_FILES}"
echo "Samples: 10,000"
echo "Batch Size: ${TRAIN_BATCH_SIZE}"
echo "GPUs: ${N_GPUS_PER_NODE}"
echo "Rollout Budget: ${ROLLOUT_N} (initial: ${INITIAL_ROLLOUTS}, beam: ${BEAM_SIZE})"
echo "Epochs: ${TOTAL_EPOCHS}"
echo "Save Path: ${SAVE_PATH}"
echo "============================================"
echo

# ============ Start Training ============
python3 -m verl.trainer.main_ppo \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VALID_FILES} \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.initial_rollouts=${INITIAL_ROLLOUTS} \
    actor_rollout_ref.rollout.beam_size=${BEAM_SIZE} \
    actor_rollout_ref.rollout.branch_probability=${BRANCH_PROBABILITY} \
    actor_rollout_ref.rollout.entropy_weight=${ENTROPY_WEIGHT} \
    actor_rollout_ref.rollout.tools.tool_instances.search.params.cache_file=${SEARCH_CACHE_PATH} \
    actor_rollout_ref.rollout.tools.tool_instances.search.params.api_key=${BING_API_KEY} \
    actor_rollout_ref.rollout.tools.tool_instances.search.params.zone=${BING_ZONE} \
    actor_rollout_ref.rollout.tools.tool_instances.python.params.conda_path=${CONDA_PATH} \
    actor_rollout_ref.rollout.tools.tool_instances.python.params.conda_env=arpo \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=${REWARD_MANAGER} \
    custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH} \
    custom_reward_function.name=${CUSTOM_REWARD_FUNCTION_NAME} \
    trainer.critic_warmup=0 \
    trainer.logger="[console, wandb]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_PATH} \
    hydra.run.dir=${SAVE_PATH}/outputs 2>&1 | tee ${SAVE_PATH}/run.log

echo
echo "============================================"
echo "ARPO Training Complete!"
echo "============================================"
echo "Logs: ${SAVE_PATH}/run.log"
echo "Checkpoints: ${SAVE_PATH}/global_step_*"
echo "============================================"
```

### 2.6 Update Production Script

```bash
nano scripts/ARPO_7B_Production_Cluster.sh
```

Update:

* `ACTOR_MODEL_PATH`: Your 7B model path

* `BING_API_KEY`: Your Bright Data API key

* `BING_ZONE`: Your Bright Data zone

* `CONDA_PATH`: Cluster conda installation path

* `SAVE_PATH`: Where to save checkpoints

* `WANDB_API_KEY`: (Optional) Your WandB key

Make executable:

```bash
chmod +x scripts/ARPO_7B_Production_Cluster.sh
```

### 2.7 Run Production Training

```bash
# Use tmux for persistent session
tmux new -s arpo_training

# Activate environment
conda activate arpo

# Navigate to directory
cd ~/ARPO/ARPO

# Launch training
bash scripts/ARPO_7B_Production_Cluster.sh

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t arpo_training
```

### 2.8 Monitor Training

**In another terminal/SSH session:**

```bash
# Monitor GPU usage (all 8 should be active)
watch -n 1 nvidia-smi

# Watch training log
tail -f /data/checkpoints/ARPO_Qwen2.5_7B_Reasoning/run.log

# Check specific metrics
grep "Reward" /data/checkpoints/ARPO_Qwen2.5_7B_Reasoning/run.log

# If WandB is configured, view dashboard:
# https://wandb.ai/your-username/reasoning_tasks
```

**Expected Progress:**

```
Epoch 0 | Step 1/78 | Reward: 0.25 | KL: 0.00 | Loss: 0.45
Epoch 0 | Step 5/78 | Reward: 0.32 | KL: 0.00 | Loss: 0.38
Epoch 0 | Step 10/78 | Reward: 0.41 | KL: 0.00 | Loss: 0.31
...
Epoch 0 | Step 78/78 | Reward: 0.58 | KL: 0.00 | Loss: 0.22
Epoch 1 | Step 1/78 | Reward: 0.60 | KL: 0.00 | Loss: 0.20
...
Epoch 1 | Step 78/78 | Reward: 0.72 | KL: 0.00 | Loss: 0.15
```

**Training Time:**

* \~3-4 minutes per step

* \~78 steps per epoch (10000 / 128)

* 2 epochs total

* **Total: 4-6 hours**

### 2.9 Convert ARPO Checkpoint to HuggingFace Format

After training completes:

```bash
cd ~/ARPO/ARPO/merge_ckpt

# Edit conversion script
nano convert_checkpoint_from_verl_to_hf_qwen3.sh
```

Update the script:

```bash
#!/bin/bash

# Source VERL checkpoint (find latest global_step_XXX)
VERL_CHECKPOINT="/data/checkpoints/ARPO_Qwen2.5_7B_Reasoning/global_step_156"  # ← UPDATE

# Output directory for HuggingFace format
OUTPUT_DIR="/data/hf_checkpoints/qwen2.5-7b-arpo"  # ← UPDATE

mkdir -p $OUTPUT_DIR

python convert_verl_to_hf.py \
    --verl_checkpoint $VERL_CHECKPOINT \
    --output_dir $OUTPUT_DIR \
    --model_type qwen2

echo "Conversion complete: $OUTPUT_DIR"
```

Run conversion:

```bash
bash convert_checkpoint_from_verl_to_hf_qwen3.sh
```

Verify:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/data/hf_checkpoints/qwen2.5-7b-arpo",
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "/data/hf_checkpoints/qwen2.5-7b-arpo",
    trust_remote_code=True
)

print("✅ ARPO model loaded successfully!")
print(f"Model type: {model.config.model_type}")
print(f"Vocab size: {len(tokenizer)}")
```

***

## STAGE 3: Cluster AEPO Production

**Goal**: Train AEPO with entropy-balancing mechanisms

**Duration**: 4-6 hours

**Reuses**: Same dataset and search cache from ARPO

### 3.1 Setup AEPO Environment

```bash
cd ~/ARPO/AEPO

# Create new conda environment
conda create -n aepo python==3.10 -y
conda activate aepo

# Install PyTorch
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install flash-attn
pip3 install flash-attn --no-build-isolation

# Install requirements
pip install -r requirements.txt

# Install VERL
cd verl_aepo_entropy
pip install -e .
cd ..
```

### 3.2 Create Production AEPO Script

**File**: `scripts/AEPO_7B_Production_Cluster.sh`

```bash
#!/bin/bash

# ============================================================================
# AEPO Production Training Script - Qwen2.5-7B
# Environment: 8x A100 40GB GPUs
# Dataset: 10K Reasoning samples (shared with ARPO)
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"
echo "Working directory: $PARENT_DIR"

# ============ Environment Variables ============
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VERL_LOGGING_LEVEL=INFO
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export RAY_memory_usage_threshold=0.8
export RAY_memory_monitor_refresh_ms=0

export PYTHONPATH="${PARENT_DIR}/verl_aepo_entropy:$PYTHONPATH"

# ============ Basic Configuration ============
PROJECT_NAME="reasoning_tasks"
EXPERIMENT_NAME="AEPO_Qwen2.5_7B_Reasoning"

CONFIG_PATH="${PARENT_DIR}/scripts/config"  # MUST be absolute path
CONFIG_NAME="ppo_trainer_dr.yaml"  # AEPO config

# ============ Hardware Settings ============
NNODES=1
N_GPUS_PER_NODE=8

# ============ AEPO-Specific Flags ============
ENABLE_DYNAMIC_ROLLOUTS=False
ENABLE_ENTROPY_BALANCED_CLIPPING=True
ENABLE_ENTROPY_BALANCED_ADVANTAGE=True

# ============ Data Configuration ============
PROMPT_KEY="prompt"
TRAIN_BATCH_SIZE=128
PPO_MINI_BATCH_SIZE=16
MAX_PROMPT_LENGTH=1536
MAX_RESPONSE_LENGTH=4096

# REUSE ARPO DATASET
TRAIN_FILES="${PARENT_DIR}/../ARPO/rl_datasets/train_10k.parquet"
VALID_FILES="${PARENT_DIR}/../ARPO/rl_datasets/test.parquet"

# ============ Model Configuration ============
# UPDATE THIS: Same model as ARPO
ACTOR_MODEL_PATH="/data/models/Qwen2.5-7B-Instruct"  # ← CHANGE THIS

# ============ Rollout Configuration ============
ROLLOUT_NAME="vllm"
ROLLOUT_MODE="sync_with_tool"
ROLLOUT_N=16
INITIAL_ROLLOUTS=8
BEAM_SIZE=2
BRANCH_PROBABILITY=0.5
ENTROPY_WEIGHT=0.2

# ============ Tool Configuration ============
# REUSE ARPO SEARCH CACHE
SEARCH_CACHE_PATH="${PARENT_DIR}/../ARPO/search_cache/search_cache.json"

# UPDATE THESE
BING_API_KEY="YOUR_API_KEY_HERE"  # ← CHANGE THIS
BING_ZONE="YOUR_ZONE_HERE"         # ← CHANGE THIS
CONDA_PATH="/home/user/anaconda3"  # ← CHANGE THIS

# ============ Reward Configuration ============
REWARD_MANAGER="naive"
CUSTOM_REWARD_FUNCTION_PATH="${PARENT_DIR}/verl_aepo_entropy/verl/utils/reward_score/math.py"
CUSTOM_REWARD_FUNCTION_NAME="compute_score"

# ============ Training Configuration ============
TOTAL_EPOCHS=2
SAVE_FREQ=5
TEST_FREQ=5

# ============ Paths ============
SAVE_PATH="/data/checkpoints/${EXPERIMENT_NAME}"  # ← CHANGE THIS if needed
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"

# ============ WandB Configuration ============
WANDB_API_KEY=""  # ← ADD YOUR WANDB KEY (optional)

# ============ Preparation ============
if [ "$WANDB_API_KEY" != "" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

mkdir -p $SAVE_PATH
mkdir -p $ROLLOUT_SAVE_PATH

echo "============================================"
echo "AEPO Production Training - Qwen2.5-7B"
echo "============================================"
echo "Model: ${ACTOR_MODEL_PATH}"
echo "Dataset: ${TRAIN_FILES}"
echo "Batch Size: ${TRAIN_BATCH_SIZE}"
echo "GPUs: ${N_GPUS_PER_NODE}"
echo "AEPO Modules:"
echo "  - Dynamic Rollouts: ${ENABLE_DYNAMIC_ROLLOUTS}"
echo "  - Entropy Clipping: ${ENABLE_ENTROPY_BALANCED_CLIPPING}"
echo "  - Entropy Advantage: ${ENABLE_ENTROPY_BALANCED_ADVANTAGE}"
echo "Epochs: ${TOTAL_EPOCHS}"
echo "Save Path: ${SAVE_PATH}"
echo "============================================"
echo

# ============ Start Training ============
python3 -m verl.trainer.main_ppo \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VALID_FILES} \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.enable_entropy_balanced_clipping=${ENABLE_ENTROPY_BALANCED_CLIPPING} \
    actor_rollout_ref.actor.enable_entropy_balanced_advantage=${ENABLE_ENTROPY_BALANCED_ADVANTAGE} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.enable_dynamic_rollouts=${ENABLE_DYNAMIC_ROLLOUTS} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.initial_rollouts=${INITIAL_ROLLOUTS} \
    actor_rollout_ref.rollout.beam_size=${BEAM_SIZE} \
    actor_rollout_ref.rollout.branch_probability=${BRANCH_PROBABILITY} \
    actor_rollout_ref.rollout.entropy_weight=${ENTROPY_WEIGHT} \
    ++actor_rollout_ref.rollout.tools.tool_instances.search.params.cache_file=${SEARCH_CACHE_PATH} \
    ++actor_rollout_ref.rollout.tools.tool_instances.search.params.api_key=${BING_API_KEY} \
    ++actor_rollout_ref.rollout.tools.tool_instances.search.params.zone=${BING_ZONE} \
    ++actor_rollout_ref.rollout.tools.tool_instances.python.params.conda_path=${CONDA_PATH} \
    ++actor_rollout_ref.rollout.tools.tool_instances.python.params.conda_env=aepo \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=${REWARD_MANAGER} \
    custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH} \
    custom_reward_function.name=${CUSTOM_REWARD_FUNCTION_NAME} \
    trainer.critic_warmup=0 \
    trainer.logger="[console, wandb]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_PATH} \
    hydra.run.dir=${SAVE_PATH}/outputs 2>&1 | tee ${SAVE_PATH}/run.log

echo
echo "============================================"
echo "AEPO Training Complete!"
echo "============================================"
echo "Logs: ${SAVE_PATH}/run.log"
echo "Checkpoints: ${SAVE_PATH}/global_step_*"
echo "============================================"
```

### 3.3 Update AEPO Script

```bash
cd ~/ARPO/AEPO
nano scripts/AEPO_7B_Production_Cluster.sh
```

Update same variables as ARPO script:

* `ACTOR_MODEL_PATH`

* `BING_API_KEY`

* `BING_ZONE`

* `CONDA_PATH`

* `SAVE_PATH`

* `WANDB_API_KEY`

Make executable:

```bash
chmod +x scripts/AEPO_7B_Production_Cluster.sh
```

### 3.4 Run AEPO Training

```bash
# Use tmux
tmux new -s aepo_training

# Activate environment
conda activate aepo

# Navigate to directory
cd ~/ARPO/AEPO

# Launch training
bash scripts/AEPO_7B_Production_Cluster.sh

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t aepo_training
```

### 3.5 Monitor AEPO Training

Same monitoring as ARPO:

```bash
watch -n 1 nvidia-smi
tail -f /data/checkpoints/AEPO_Qwen2.5_7B_Reasoning/run.log
```

**Expected**: Similar or better final reward than ARPO (0.6-0.8+)

### 3.6 Convert AEPO Checkpoint

```bash
cd ~/ARPO/AEPO/merge_ckpt

# Edit conversion script (similar to ARPO)
nano convert_checkpoint_from_verl_to_hf_qwen3.sh
```

Update paths:

```bash
VERL_CHECKPOINT="/data/checkpoints/AEPO_Qwen2.5_7B_Reasoning/global_step_156"
OUTPUT_DIR="/data/hf_checkpoints/qwen2.5-7b-aepo"
```

Run conversion:

```bash
bash convert_checkpoint_from_verl_to_hf_qwen3.sh
```

***

## STAGE 4: Verification

**Goal**: Test both models generate tool-augmented responses

### 4.1 Test ARPO Model

Create test script: `scripts/test_arpo_model.py`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def test_model(model_path, model_name):
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}\n")

    # Load model
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    print(f"✅ Model loaded successfully!")
    print(f"Model type: {model.config.model_type}")
    print(f"Vocab size: {len(tokenizer)}\n")

    # Test prompts
    test_prompts = [
        "What is the derivative of x^3 + 2x^2 - 5x + 3?",
        "Calculate 12345 * 67890",
        "What is the capital of France? Search online if needed."
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {prompt}\n")

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0
            )

        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=False
        )

        print(f"Response: {response}\n")

        # Check for tool usage
        has_python = "<python>" in response
        has_search = "<search>" in response
        has_answer = "<answer>" in response

        print(f"Tool usage:")
        print(f"  - Python: {'✅' if has_python else '❌'}")
        print(f"  - Search: {'✅' if has_search else '❌'}")
        print(f"  - Answer: {'✅' if has_answer else '❌'}")

if __name__ == "__main__":
    # Test ARPO model
    test_model(
        model_path="/data/hf_checkpoints/qwen2.5-7b-arpo",
        model_name="ARPO Model"
    )
```

Run test:

```bash
python scripts/test_arpo_model.py
```

### 4.2 Test AEPO Model

```python
# Same script, just change model_path
test_model(
    model_path="/data/hf_checkpoints/qwen2.5-7b-aepo",
    model_name="AEPO Model"
)
```

### 4.3 Compare Models

Create comparison script: `scripts/compare_models.py`

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

def load_training_logs(arpo_log, aepo_log):
    """Parse training logs and extract metrics."""

    def parse_log(log_file):
        rewards = []
        losses = []
        with open(log_file, 'r') as f:
            for line in f:
                if "Reward:" in line:
                    # Parse: Epoch X | Step Y | Reward: 0.XX | Loss: 0.XX
                    parts = line.split('|')
                    for part in parts:
                        if "Reward:" in part:
                            reward = float(part.split(':')[1].strip())
                            rewards.append(reward)
                        if "Loss:" in part:
                            loss = float(part.split(':')[1].strip())
                            losses.append(loss)
        return rewards, losses

    arpo_rewards, arpo_losses = parse_log(arpo_log)
    aepo_rewards, aepo_losses = parse_log(aepo_log)

    return {
        'arpo': {'rewards': arpo_rewards, 'losses': arpo_losses},
        'aepo': {'rewards': aepo_rewards, 'losses': aepo_losses}
    }

def plot_comparison(metrics):
    """Plot reward and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot rewards
    ax1.plot(metrics['arpo']['rewards'], label='ARPO', linewidth=2)
    ax1.plot(metrics['aepo']['rewards'], label='AEPO', linewidth=2)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Reward Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot losses
    ax2.plot(metrics['arpo']['losses'], label='ARPO', linewidth=2)
    ax2.plot(metrics['aepo']['losses'], label='AEPO', linewidth=2)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('arpo_aepo_comparison.png', dpi=300)
    print("Comparison plot saved to: arpo_aepo_comparison.png")

def print_summary(metrics):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)

    for model in ['arpo', 'aepo']:
        rewards = metrics[model]['rewards']
        losses = metrics[model]['losses']

        print(f"\n{model.upper()}:")
        print(f"  Final Reward: {rewards[-1]:.4f}")
        print(f"  Max Reward: {max(rewards):.4f}")
        print(f"  Avg Reward (last 10 steps): {sum(rewards[-10:])/10:.4f}")
        print(f"  Final Loss: {losses[-1]:.4f}")
        print(f"  Min Loss: {min(losses):.4f}")

if __name__ == "__main__":
    metrics = load_training_logs(
        arpo_log="/data/checkpoints/ARPO_Qwen2.5_7B_Reasoning/run.log",
        aepo_log="/data/checkpoints/AEPO_Qwen2.5_7B_Reasoning/run.log"
    )

    print_summary(metrics)
    plot_comparison(metrics)
```

Run comparison:

```bash
python scripts/compare_models.py
```

***

## Troubleshooting

### Local Toy Training Issues

**Issue: 3B Model OOM**

Symptoms:

```
RuntimeError: CUDA out of memory
torch.cuda.OutOfMemoryError
```

Solution:

1. Use 0.5B fallback script
2. Or further reduce batch size in 3B script:

   ```bash
   TRAIN_BATCH_SIZE=2
   PPO_MINI_BATCH_SIZE=1
   ROLLOUT_N=2
   ```

**Issue: vLLM OOM**

Symptoms:

```
RuntimeError: vLLM OOM during model loading
```

Solution:

```bash
# Reduce vLLM memory
gpu_memory_utilization=0.3  # From 0.4
```

**Issue: Tool Execution Timeout**

Symptoms:

```
WARNING: Python tool timeout (60s exceeded)
```

Solution:

```bash
# Increase timeout in config
timeout: 120  # From 60
```

### Cluster Training Issues

**Issue: Not All GPUs Active**

Check:

```bash
nvidia-smi
# Should show all 8 GPUs with activity
```

Solution:

```bash
# Verify N_GPUS_PER_NODE=8 in script
# Check CUDA_VISIBLE_DEVICES not set
```

**Issue: Ray Cluster Fails to Start**

Solution:

```bash
# Kill existing Ray
ray stop --force

# Clear Ray temp files
rm -rf /tmp/ray

# Restart training
```

**Issue: Reward Stays at 0**

Check:

```bash
# Verify reward function path
tail -100 /data/checkpoints/ARPO*/run.log | grep "reward"
```

Solution:

```bash
# Test reward function manually
python -c "
from verl.utils.reward_score.math import compute_score
samples = [{'output': '<answer>42</answer>', 'answer': '42', 'data_source': 'gsm8k'}]
print(compute_score(samples))  # Should be [1.0]
"
```

**Issue: Training Diverges (Loss → NaN)**

Solution:

```bash
# Reduce learning rate
actor_rollout_ref.actor.optim.lr=5e-7  # From 1e-6

# Enable gradient clipping
actor_rollout_ref.actor.max_grad_norm=1.0

# Reduce clip ratio
actor_rollout_ref.actor.clip_ratio=0.1  # From 0.2
```

***

## Summary Checklist

### Local Toy Training ✓

* [ ] Environment setup complete

* [ ] Models downloaded (3B and 0.5B)

* [ ] Toy dataset created (100 samples)

* [ ] Search API configured

* [ ] Training completed successfully

* [ ] Noted which model worked (3B or 0.5B)

### Cluster ARPO Training ✓

* [ ] Environment setup on cluster

* [ ] 7B model downloaded/SFT checkpoint available

* [ ] Full dataset downloaded (10K samples)

* [ ] Search API configured

* [ ] Training completed (2 epochs)

* [ ] Checkpoint converted to HF format

* [ ] Model loads and generates successfully

### Cluster AEPO Training ✓

* [ ] AEPO environment setup

* [ ] AEPO script configured

* [ ] Training completed (2 epochs)

* [ ] Checkpoint converted to HF format

* [ ] Model loads and generates successfully

### Verification ✓

* [ ] Both models tested with sample prompts

* [ ] Tool usage verified (Python and Search)

* [ ] Training curves compared

* [ ] Final rewards compared

* [ ] Models ready for full evaluation

***

## Next Steps

With both ARPO and AEPO models trained and verified, you can now:

1. **Evaluate on 13 Benchmarks**: Use the evaluation pipeline to test on AIME, MATH500, GSM8K, HotpotQA, 2Wiki, Musique, Bamboogle, GAIA, HLE, etc.

2. **Fine-tune Further**: If performance is below target, consider:

   * More training epochs

   * Different hyperparameters

   * Larger model (14B)

3. **Deploy**: Use the HF checkpoints for inference or API deployment

***

## Quick Reference Commands

**Local Toy Training:**

```bash
cd ARPO/ARPO
conda activate arpo
bash scripts/ARPO_3B_Toy_Local.sh  # Try 3B first
# If OOM:
bash scripts/ARPO_0.5B_Toy_Local.sh  # Fallback to 0.5B
```

**Cluster ARPO Training:**

```bash
ssh user@cluster
cd ~/ARPO/ARPO
conda activate arpo
tmux new -s arpo_training
bash scripts/ARPO_7B_Production_Cluster.sh
```

**Cluster AEPO Training:**

```bash
cd ~/ARPO/AEPO
conda activate aepo
tmux new -s aepo_training
bash scripts/AEPO_7B_Production_Cluster.sh
```

**Monitor Training:**

```bash
watch -n 1 nvidia-smi
tail -f /data/checkpoints/ARPO*/run.log
tmux attach -t arpo_training
```

**Convert Checkpoints:**

```bash
cd ARPO/merge_ckpt
bash convert_checkpoint_from_verl_to_hf_qwen3.sh
```

**Test Models:**

```bash
python scripts/test_arpo_model.py
python scripts/compare_models.py
```

***

**Congratulations!** You now have a complete end-to-end RL training pipeline from local testing to production deployment. Both ARPO and AEPO models are ready for evaluation and deployment.
