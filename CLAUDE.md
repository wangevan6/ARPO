# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ARPO (Agentic Reinforced Policy Optimization) is a reinforcement learning framework for training multi-turn LLM-based agents with tool integration. This repository contains two main RL algorithms:

- **ARPO**: Encourages adaptive branch sampling during high-entropy tool-call rounds for efficient tool-use alignment
- **AEPO** (Agentic Entropy-Balanced Policy Optimization): Balances entropy in both rollout and policy update phases

The training pipeline consists of three stages: SFT (Supervised Fine-Tuning), RL Training (ARPO/AEPO), and Evaluation across 13 benchmarks.

## Repository Structure

```
ARPO/
├── LLaMA-Factory/          # SFT stage using LLaMA-Factory framework
├── ARPO/                   # ARPO RL training implementation
│   ├── verl_arpo_entropy/  # VERL framework for ARPO
│   ├── scripts/            # Training scripts (7B, 8B, 14B models)
│   ├── rl_datasets/        # RL training data (parquet files)
│   └── search_cache/       # Shared search result cache
├── AEPO/                   # AEPO RL training implementation
│   ├── verl_aepo_entropy/  # VERL framework for AEPO
│   └── scripts/            # Training scripts with entropy balancing
└── evaluation/             # Multi-benchmark evaluation framework
    ├── src/                # Evaluation engine with tool support
    ├── data/               # 13 test benchmarks (GAIA, HLE, MATH, etc.)
    └── vllm_scripts/       # vLLM inference server scripts
```

## Key Architecture Concepts

### Training Pipeline Flow

1. **SFT Stage** (Optional): Cold-start fine-tuning on 54K multi-tool conversation samples using LLaMA-Factory with DeepSpeed ZeRO-3
2. **RL Stage**: Distributed RL training using Ray with tool-augmented rollouts (Python executor + Bing Search API)
3. **Evaluation**: Multi-turn generation with tool calls across 13 benchmarks using vLLM inference

### ARPO Core Mechanism

ARPO implements **entropy-based branching** during rollout:
- Monitors entropy after each tool-call round
- Triggers branching when entropy exceeds threshold: `entropy > entropy_weight * base_entropy`
- Creates `beam_size` branches with probability `branch_probability`
- Branches share computation until tool execution diverges

Key parameters:
- `initial_rollouts`: Base number of samples per prompt (e.g., 8)
- `beam_size`: Branch factor when high entropy detected (e.g., 2)
- `n`: Total rollout budget (e.g., 16 = 8 initial + 8 from branching)
- `entropy_weight`: Threshold multiplier (e.g., 0.2)

### AEPO Enhancements

AEPO adds three components to ARPO:

1. **Dynamic Entropy-Balanced Rollout** (`enable_dynamic_rollouts`): Adaptively allocates sampling budget with branch penalty on consecutive high-entropy steps
2. **Entropy Clipping-Balanced Mechanism** (`enable_entropy_balanced_clipping`): Stop-gradient on high-entropy clipping term to preserve gradients
3. **Entropy-aware Advantage Estimation** (`enable_entropy_balanced_advantage`): Prioritizes learning on high-uncertainty tokens

### Distributed Training Architecture

- **Ray-based orchestration** with worker groups:
  - `ActorRolloutRefWorker`: Policy model + vLLM generation + reference policy
  - `CriticWorker`: Value function for GAE advantage estimation
- **FSDP sharding** for large models (7B-14B parameters)
- **Tool execution**: Concurrent execution with shared cache (async writes to `search_cache.json`)

### Tool System

Two primary tools integrated into rollout:

1. **Python Tool**: Executes code in isolated conda environment
   - Preprocessing: Converts expressions to print statements
   - Timeout: 120s default
   - Tag: `<python>...</python>`

2. **Search Tool** (Bright Data Bing API):
   - Shared cache with async writes to prevent blocking
   - Configurable result length and location
   - Tag: `<search>...</search>`

## Common Development Commands

### Environment Setup

**SFT Environment:**
```bash
cd LLaMA-Factory
conda create -n sft python=3.10
conda activate sft
pip install -r requirements.txt
```

**RL Environment (ARPO/AEPO):**
```bash
conda create -n arpo python==3.10
conda activate arpo
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
cd ARPO  # or AEPO
pip install -r requirements.txt
```

**Evaluation Environment:**
```bash
conda create -n evaluation python=3.10
conda activate evaluation
cd evaluation
pip install -r requirements.txt
```

### Training Commands

**SFT Training:**
```bash
cd LLaMA-Factory/arpo_train_sft
# Edit yaml/sft_config.yaml with model path and output directory
bash sft_train.sh
```

**ARPO RL Training:**
```bash
cd ARPO/scripts
# Edit ARPO_7B_Reasoning_1node.sh with paths:
#   - ACTOR_MODEL_PATH: Your SFT checkpoint
#   - TRAIN_FILES/VALID_FILES: Dataset paths
#   - SAVE_PATH: Checkpoint output directory
#   - SEARCH_CACHE_PATH: search_cache.json path
#   - API keys in config/ppo_trainer.yaml
bash ARPO_7B_Reasoning_1node.sh
```

**AEPO RL Training:**
```bash
cd AEPO/scripts
# Edit AEPO_Qwen3_14B_DeepResearch.sh with paths (similar to ARPO)
# Configure AEPO modules:
#   - ENABLE_DYNAMIC_ROLLOUTS=True/False
#   - ENABLE_ENTROPY_BALANCED_CLIPPING=True/False
#   - ENABLE_ENTROPY_BALANCED_ADVANTAGE=True/False
bash AEPO_Qwen3_14B_DeepResearch.sh
```

**Convert RL Checkpoint to HuggingFace Format:**
```bash
cd ARPO/merge_ckpt  # or AEPO/merge_ckpt
bash convert_checkpoint_from_verl_to_hf_qwen3.sh
```

### Evaluation Commands

**Start vLLM Inference Servers:**
```bash
cd evaluation/vllm_scripts
# Edit vllm_launch_reasoning_model_cuda4-7.sh with MODEL_PATH and MODEL_NAME
bash vllm_launch_reasoning_model_cuda4-7.sh

# Start summarization model (for webpage extraction)
bash vllm_launch_summarize_model_cuda0-3_<your_model>.sh
```

**Run Evaluation:**
```bash
cd evaluation
# Edit infer_local_sds.sh:
#   - data_names: Select benchmarks (gaia, hle, math500, etc.)
#   - EXP_NAME, MODEL_PATH, OUTPUT_PATH
#   - BING_API_KEY, BING_ZONE
#   - CONDA_PATH, CONDA_ENV
bash infer_local_sds.sh
```

**Calculate Metrics:**
```bash
# Start evaluation model (Qwen2.5-72B-Instruct)
bash evaluation/deploy_qwen2.5_72B_instruct.sh

# Edit evaluate_passk.sh with OUTPUT_DIR
bash evaluation/evaluate_passk.sh
```

## Configuration Management

### Hydra Configuration System (RL Training)

Both ARPO and AEPO use Hydra for configuration management:

**Base Config:** `ARPO/scripts/config/ppo_trainer.yaml` or `AEPO/scripts/config/ppo_trainer_dr.yaml`

**CLI Overrides** in training scripts:
```bash
python3 -m verl.trainer.main_ppo \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    data.train_batch_size=128 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.initial_rollouts=8
```

### Critical Configuration Paths to Update

When setting up training, you MUST update these paths:

1. **Training Script Variables:**
   - `ACTOR_MODEL_PATH`: Path to SFT checkpoint or base model
   - `TRAIN_FILES`/`VALID_FILES`: Dataset paths (`.parquet` files)
   - `SAVE_PATH`: Checkpoint output directory
   - `SEARCH_CACHE_PATH`: Shared search cache file
   - `CUSTOM_REWARD_FUNCTION_PATH`: Reward function script path
   - `WANDB_API_KEY`: Weights & Biases key (optional)

2. **Tool Configuration** (in `ppo_trainer.yaml`):
   - `actor_rollout_ref.rollout.tools.tool_instances.python.params.conda_path`
   - `actor_rollout_ref.rollout.tools.tool_instances.python.params.conda_env`
   - `actor_rollout_ref.rollout.tools.tool_instances.search.params.api_key`
   - `actor_rollout_ref.rollout.tools.tool_instances.search.params.zone`
   - `actor_rollout_ref.rollout.tools.tool_instances.search.params.cache_file`

3. **Evaluation Configuration** (in `infer_local_sds.sh`):
   - Search API credentials match training configuration
   - Conda environment paths for tool execution

## Important Implementation Details

### Data Format

**RL Training Input (Parquet):**
```python
{
  "prompt": "What is the capital of France?",
  "answer": "Paris",
  "data_source": "bamboogle",  # Used for reward function selection
  "tools": ["search"]
}
```

**Multi-turn Conversation Format:**
```json
{
  "conversations": [
    {"from": "user", "value": "Calculate 2+2"},
    {"from": "assistant", "value": "<think>I need to calculate</think><python>print(2+2)</python>"},
    {"from": "user", "value": "<result>4</result>"},
    {"from": "assistant", "value": "The answer is 4"}
  ]
}
```

### Reward Functions

Reward functions are located in `verl_arpo_entropy/verl/utils/reward_score/`:

- **`deep_research.py`**: For GAIA, HLE (format validation 20% + correctness 80%)
- **`math.py`**: For AIME, MATH500, GSM8K (exact match after normalization)

Custom reward functions must implement:
```python
def compute_score(data: List[Dict]) -> List[float]:
    """
    Args:
        data: List of {output, answer, data_source}
    Returns:
        scores: List of floats [0.0, 1.0]
    """
```

### Memory Optimization

For training large models (14B+):

1. Enable gradient checkpointing: `enable_gradient_checkpointing: True`
2. Use FSDP parameter offloading for reference policy: `ref.fsdp_config.param_offload: True`
3. Adjust GPU memory utilization: `rollout.gpu_memory_utilization: 0.6-0.7`
4. Use dynamic batch sizing: `actor.use_dynamic_bsz: True` with `ppo_max_token_len_per_gpu`

### Search Cache Mechanism

The search cache (`search_cache.json`) uses async writes to prevent blocking during rollout:
- Cache hits are served immediately
- Cache misses trigger API call + async background write
- Shared across all training runs and evaluation
- Location MUST match in training config, tool config, and evaluation scripts

## Git Workflow

This repository is a fork. When working with changes:

**Push to Fork:**
```bash
git push fork main
```

**Create Pull Request to Upstream:**
```bash
# After pushing to fork, create PR from fork (wangevan6/ARPO) to upstream (RUC-NLPIR/ARPO)
```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure Bright Data API keys are set in THREE locations: `scripts/config/ppo_trainer.yaml`, `verl_*/verl/workers/rollout/tools/config_example.yaml`, and evaluation scripts

2. **CUDA Out of Memory**:
   - Reduce `train_batch_size`, `ppo_mini_batch_size`
   - Increase `tensor_model_parallel_size` for rollout
   - Enable parameter offloading for ref policy

3. **Tool Execution Timeout**: Increase `tools.timeout` in config (default 120s)

4. **Checkpoint Conversion Errors**: Ensure source checkpoint path in `convert_checkpoint_from_verl_to_hf_qwen3.sh` matches the saved RL checkpoint

5. **Ray Worker Timeout**: Increase `ray_wait_register_center_timeout` if initialization is slow

## Model Support

Supported base models:
- Qwen2.5 series (3B, 7B)
- Qwen3 series (8B, 14B)
- LLaMA 3.1 (8B)

When using different model families, update:
- Template in SFT config: `template: qwen` or `template: llama3`
- Checkpoint conversion script for model architecture
- Notice using hf instead of huggingface-cli which is outdated, so for instacne when you downlo ARPO-sft dataset, you use ""hf download dongguanting/ARPO-SFT-54K final_5w4_still.jsonl \
  --repo-type dataset \
  --local-dir ./temp_dataset ""
- conda activate sft to activate the correct environment