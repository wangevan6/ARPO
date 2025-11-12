#!/bin/bash
# ARPO Toy Training Script - Qwen2.5-3B-Instruct
# For local single GPU testing (16GB VRAM)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"
echo "Switched to parent directory: $PARENT_DIR"

# ============================ Environment Setup ============================
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VERL_LOGGING_LEVEL=DEBUG
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export RAY_memory_usage_threshold=0.8
export RAY_memory_monitor_refresh_ms=0
export USE_LIBUV=0  # Windows compatibility

# Set Python path (absolute path for Windows)
export PYTHONPATH="C:/Users/user/Projects/ARPO/ARPO/verl_arpo_entropy:$PYTHONPATH"

# ============================ Basic Configuration ============================
PROJECT_NAME="toy_testing"
EXPERIMENT_NAME="ARPO_3B_Toy_Local"

# Configuration file path (absolute path for Windows)
CONFIG_PATH="C:/Users/user/Projects/ARPO/ARPO/scripts/config"
CONFIG_NAME="ppo_trainer.yaml"

# Distributed training settings (single GPU for toy)
NNODES=1
N_GPUS_PER_NODE=1

# ============================ Data Configuration ============================
PROMPT_KEY="prompt"
TRAIN_BATCH_SIZE=16                 # Reduced for single GPU
PPO_MINI_BATCH_SIZE=4               # Reduced for single GPU
MAX_PROMPT_LENGTH=1536
MAX_RESPONSE_LENGTH=4096

# Data file paths (absolute paths for Windows)
TRAIN_FILES="C:/Users/user/Projects/ARPO/ARPO/rl_datasets/train_toy.parquet"
VALID_FILES="C:/Users/user/Projects/ARPO/ARPO/rl_datasets/valid.parquet"

# ============================ Model Configuration ============================
# Actor model path (absolute path for Windows)
ACTOR_MODEL_PATH="C:/Users/user/Projects/ARPO/ARPO/models/Qwen2.5-3B-Instruct"

# ============================ Rollout Configuration ==========================
ROLLOUT_NAME="vllm"
ROLLOUT_MODE="sync_with_tool"
ROLLOUT_N=4                          # Reduced for toy (was 16)
INITIAL_ROLLOUTS=2                  # Reduced for toy (was 8)
BEAM_SIZE=2
BRANCH_PROBABILITY=0.5
Entropy_weight=0.2

# ============================ Rollout Tools Configuration ==========================
SEARCH_CACHE_PATH="C:/Users/user/Projects/ARPO/ARPO/search_cache/search_cache.json"

# ============================ Bright Data API Configuration ==========================
# Bright Data credentials (working and tested)
BING_API_KEY="2e1ce5909f8fe768882fcb1a1ca38052e2eecffc560ec94e0b96e1e7d2bfc731"
BING_ZONE="serp_api1"

# ============================ Conda Configuration ==========================
# Path to conda installation (for Python tool)
CONDA_PATH="C:/Users/user/miniconda3"

# ============================ Reward Model Configuration ==========================
REWARD_MANAGER="naive"
CUSTOM_REWARD_FUNCTION_PATH="C:/Users/user/Projects/ARPO/ARPO/verl_arpo_entropy/verl/utils/reward_score/deep_research.py"
CUSTOM_REWARD_FUNCTION_NAME="compute_score"

# ============================ Training Configuration ============================
TOTAL_EPOCHS=1                      # Just 1 epoch for toy
SAVE_FREQ=50                       # Save every 50 steps
TEST_FREQ=50

# ============================ Path Configuration ============================
SAVE_PATH="C:/Users/user/Projects/ARPO/ARPO/checkpoints_toy/${EXPERIMENT_NAME}"
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"

# ============================ WandB Configuration ============================
WANDB_API_KEY=""  # Leave empty to disable WandB logging
SEARCH_CLASS_PATH="verl.workers.rollout.tools.search_tool.BingSearchTool"

# ============================ Preparation ============================
# Create save directories
if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p "$SAVE_PATH"
fi

if [ ! -d "$ROLLOUT_SAVE_PATH" ]; then
    mkdir -p "$ROLLOUT_SAVE_PATH"
fi

# Create search cache directory and file
mkdir -p "C:/Users/user/Projects/ARPO/ARPO/search_cache"
if [ ! -f "$SEARCH_CACHE_PATH" ]; then
    echo '{}' > "$SEARCH_CACHE_PATH"
fi

# ============================ Start Training ============================
echo "==========================================="
echo "Starting ARPO Toy Training - 3B Model"
echo "==========================================="
echo "Model: $ACTOR_MODEL_PATH"
echo "Dataset: $TRAIN_FILES"
echo "Save Path: $SAVE_PATH"
echo "GPUs: $N_GPUS_PER_NODE"
echo "Batch Size: $TRAIN_BATCH_SIZE"
echo "Rollouts: $ROLLOUT_N (initial: $INITIAL_ROLLOUTS)"
echo "==========================================="

/c/Users/user/miniconda3/envs/arpo/python.exe -m verl.trainer.main_ppo \
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
    actor_rollout_ref.rollout.entropy_weight=${Entropy_weight} \
    actor_rollout_ref.rollout.tools.tool_instances.search.params.cache_file=${SEARCH_CACHE_PATH} \
    actor_rollout_ref.rollout.tools.tool_instances.search.class_path=${SEARCH_CLASS_PATH} \
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

echo "==========================================="
echo "Training Complete!"
echo "Checkpoints saved to: $SAVE_PATH"
echo "==========================================="
