# ARPO/AEPO RL Training - Complete Documentation Package

**Created**: 2025

**Purpose**: Comprehensive guide for training ARPO and AEPO models from local testing to production deployment

---

## 📁 Documentation Files Created

This package includes the following documentation and scripts:

### 1. Main Guides

| File | Description |
|------|-------------|
| **Complete_RL_Training_Guide.md** | Comprehensive step-by-step guide with detailed explanations, configuration examples, and troubleshooting |
| **RL_Training_Quickstart.md** | Quick reference for executing the training pipeline with minimal explanations |
| **README_RL_Training.md** | This file - overview and getting started |

### 2. Helper Scripts

Location: `ARPO/ARPO/scripts/`

| Script | Purpose |
|--------|---------|
| **create_toy_dataset.py** | Create 100-sample toy dataset from full 10K dataset for local testing |
| **test_model.py** | Test trained models with sample prompts to verify tool usage |
| **compare_models.py** | Compare ARPO and AEPO training metrics and generate comparison plots |

### 3. Training Scripts (To Be Created)

**Note**: Due to their length, the full training scripts are documented in `Complete_RL_Training_Guide.md`. You need to create them based on the templates provided:

| Script | Location | Purpose |
|--------|----------|---------|
| **ARPO_3B_Toy_Local.sh** | `ARPO/ARPO/scripts/` | Local toy training with 3B model (16GB GPU) |
| **ARPO_0.5B_Toy_Local.sh** | `ARPO/ARPO/scripts/` | Fallback toy training with 0.5B model |
| **ARPO_7B_Production_Cluster.sh** | `ARPO/ARPO/scripts/` | Production ARPO training (8x 40GB GPUs) |
| **AEPO_7B_Production_Cluster.sh** | `ARPO/AEPO/scripts/` | Production AEPO training (8x 40GB GPUs) |

---

## 🚀 Getting Started

### Which Guide Should I Use?

**For first-time setup with detailed explanations:**
→ Read `Complete_RL_Training_Guide.md`

**For quick execution (already familiar with the process):**
→ Use `RL_Training_Quickstart.md`

### Recommended Workflow

1. **Read the Complete Guide First**
   - Understand the overall architecture
   - Learn about ARPO and AEPO mechanisms
   - Review prerequisites and requirements

2. **Set Up Local Toy Training**
   - Follow Section 1 of either guide
   - Verify the RL pipeline works on your 16GB GPU
   - This saves time by catching configuration errors early

3. **Scale to Cluster Production**
   - Follow Sections 2-3 for ARPO and AEPO
   - Train on full 10K dataset with 7B model
   - Monitor training progress and metrics

4. **Verify and Compare**
   - Test both models with provided scripts
   - Compare ARPO vs AEPO performance
   - Prepare for full benchmark evaluation

---

## 📋 Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    RL TRAINING PIPELINE                      │
└─────────────────────────────────────────────────────────────┘

STAGE 1: Local Toy Testing (1-2 hours)
├── Setup environment (conda, PyTorch, VERL)
├── Download models (3B + 0.5B fallback)
├── Create toy dataset (100 samples)
├── Configure API keys (Bright Data)
├── Run toy training
└── Verify pipeline works
        ↓
STAGE 2: Cluster ARPO Training (4-6 hours)
├── Transfer setup to cluster
├── Download 7B model + full dataset (10K samples)
├── Configure production training
├── Run ARPO training (2 epochs)
├── Convert checkpoint to HuggingFace format
└── Verify model generates tool-augmented responses
        ↓
STAGE 3: Cluster AEPO Training (4-6 hours)
├── Setup AEPO environment
├── Reuse ARPO dataset and search cache
├── Configure AEPO-specific modules
├── Run AEPO training (2 epochs)
├── Convert checkpoint to HuggingFace format
└── Verify model performance
        ↓
STAGE 4: Verification (30 min)
├── Test both models with sample prompts
├── Compare training curves and metrics
├── Analyze tool usage patterns
└── Prepare for benchmark evaluation

TOTAL TIME: 10-15 hours (mostly unattended training)
```

---

## 🔑 Key Configuration Points

Before starting, ensure you have:

### Required Credentials
- [x] **HuggingFace account** - For model downloads
- [x] **Bright Data account** - For Bing Search API (API key + zone)
- [x] **WandB account** (optional) - For training monitoring

### Hardware Requirements

**Local Testing:**
- 1 GPU with 16GB VRAM
- WSL2 with CUDA support

**Production Training:**
- 8x NVIDIA A100 40GB GPUs (or equivalent)
- Linux cluster environment

### Software Prerequisites
- Python 3.10
- CUDA 12.1+ (for vLLM compatibility)
- Conda/Miniconda
- Git with LFS

---

## 📝 Critical Configuration Checklist

### Must Update in Scripts

When creating training scripts from the templates, update these:

**All Scripts:**
- [ ] `BING_API_KEY` - Your Bright Data API key
- [ ] `BING_ZONE` - Your Bright Data zone
- [ ] `CONDA_PATH` - Your conda installation path
- [ ] `ACTOR_MODEL_PATH` - Path to your model/SFT checkpoint

**Production Scripts Only:**
- [ ] `SAVE_PATH` - Where to save checkpoints
- [ ] `WANDB_API_KEY` - Your WandB key (optional)

### Must Update in Config Files

**3 Locations for API Keys:**

1. `ARPO/ARPO/scripts/config/ppo_trainer.yaml`
2. `ARPO/ARPO/verl_arpo_entropy/verl/workers/rollout/tools/config_example.yaml`
3. Training scripts (as above)

**Important**: All 3 locations must have matching API keys!

---

## 🎯 Training Parameters

### Local Toy (3B Model, 16GB GPU)

```bash
TRAIN_BATCH_SIZE=4              # Small for memory
ROLLOUT_N=4                     # Minimal rollouts
MAX_PROMPT_LENGTH=512           # Short sequences
gpu_memory_utilization=0.4      # Conservative vLLM
param_offload=True              # Offload to save memory
```

### Production (7B Model, 8x 40GB GPUs)

```bash
TRAIN_BATCH_SIZE=128            # Full batch
ROLLOUT_N=16                    # Full rollout budget
MAX_PROMPT_LENGTH=1536          # Production length
gpu_memory_utilization=0.7      # Utilize most GPU memory
param_offload=False             # Keep on GPU
```

### AEPO-Specific Flags

```bash
ENABLE_DYNAMIC_ROLLOUTS=False           # Dynamic entropy-balanced rollout
ENABLE_ENTROPY_BALANCED_CLIPPING=True   # Stop-gradient on high entropy
ENABLE_ENTROPY_BALANCED_ADVANTAGE=True  # Entropy-aware advantage
```

---

## 📊 Expected Results

### Training Metrics

**ARPO (7B, Reasoning Dataset):**
- Initial Reward: ~0.25-0.35
- Final Reward: ~0.65-0.75
- Training Time: 4-6 hours (2 epochs)
- Steps per Epoch: ~78 (10K / 128 batch)

**AEPO (7B, Reasoning Dataset):**
- Initial Reward: ~0.25-0.35
- Final Reward: ~0.70-0.80 (may exceed ARPO)
- Training Time: 4-6 hours (2 epochs)
- Advantage: Smoother convergence

### Model Outputs

Successfully trained models should generate:

**Example Input:**
```
What is the derivative of x^3 + 2x^2 - 5x + 3?
```

**Example Output:**
```xml
<think>
I need to find the derivative using the power rule.
For each term ax^n, the derivative is n*ax^(n-1).
</think>

<python>
from sympy import symbols, diff
x = symbols('x')
expr = x**3 + 2*x**2 - 5*x + 3
derivative = diff(expr, x)
print(derivative)
</python>

<result>
3*x**2 + 4*x - 5
</result>

<answer>
The derivative is 3x^2 + 4x - 5
</answer>
```

---

## 🐛 Common Issues

### Local Training

| Issue | Solution |
|-------|----------|
| 3B model OOM | Use 0.5B fallback script |
| vLLM OOM | Reduce `gpu_memory_utilization=0.3` |
| Tool timeout | Increase `timeout: 120` in tool config |

### Cluster Training

| Issue | Solution |
|-------|----------|
| Not all GPUs active | Verify `N_GPUS_PER_NODE=8` |
| Ray fails to start | `ray stop --force && rm -rf /tmp/ray` |
| Reward stays at 0 | Check reward function path and output format |
| Training diverges (NaN) | Reduce learning rate to `5e-7` |

**Full troubleshooting guide**: See `Complete_RL_Training_Guide.md` Section "Troubleshooting"

---

## 📁 File Organization

After completing the training, your directory structure should look like:

```
ARPO/
├── ARPO/
│   ├── models/
│   │   ├── Qwen2.5-3B-Instruct/          # Local toy model
│   │   └── Qwen2.5-0.5B-Instruct/        # Fallback model
│   ├── rl_datasets/
│   │   ├── train.parquet                  # Full 10K dataset
│   │   ├── train_toy.parquet              # 100-sample toy dataset
│   │   └── valid.parquet                  # Validation set
│   ├── search_cache/
│   │   └── search_cache.json              # Shared search cache
│   ├── checkpoints_toy/
│   │   └── ARPO_*_toy_local/              # Local toy checkpoints
│   ├── scripts/
│   │   ├── create_toy_dataset.py          # ✅ Created
│   │   ├── test_model.py                  # ✅ Created
│   │   ├── compare_models.py              # ✅ Created
│   │   ├── ARPO_3B_Toy_Local.sh           # ⚠️ Create from guide
│   │   ├── ARPO_0.5B_Toy_Local.sh         # ⚠️ Create from guide
│   │   └── ARPO_7B_Production_Cluster.sh  # ⚠️ Create from guide
│   └── verl_arpo_entropy/
│
├── AEPO/
│   ├── scripts/
│   │   └── AEPO_7B_Production_Cluster.sh  # ⚠️ Create from guide
│   └── verl_aepo_entropy/
│
└── .trae/
    └── documents/
        ├── Complete_RL_Training_Guide.md   # ✅ Created
        ├── RL_Training_Quickstart.md       # ✅ Created
        └── README_RL_Training.md           # ✅ This file
```

**On Cluster:**
```
/data/
├── models/
│   └── Qwen2.5-7B-Instruct/              # Production model
├── checkpoints/
│   ├── ARPO_Qwen2.5_7B_Reasoning/        # ARPO training
│   └── AEPO_Qwen2.5_7B_Reasoning/        # AEPO training
└── hf_checkpoints/
    ├── qwen2.5-7b-arpo/                  # Converted ARPO
    └── qwen2.5-7b-aepo/                  # Converted AEPO
```

---

## ✅ Success Criteria

### Local Toy Training
- [ ] Environment setup without errors
- [ ] Toy dataset created (100 samples)
- [ ] Training completed 1 epoch
- [ ] Model generates tool calls (`<python>`, `<search>`)
- [ ] Checkpoint saved successfully

### Production ARPO
- [ ] Training completed 2 epochs
- [ ] Final reward > 0.65
- [ ] All 8 GPUs utilized
- [ ] Checkpoint converted to HuggingFace format
- [ ] Model passes verification tests

### Production AEPO
- [ ] Training completed 2 epochs
- [ ] Final reward ≥ ARPO reward
- [ ] Checkpoint converted to HuggingFace format
- [ ] Model passes verification tests

### Overall
- [ ] Both models tested with sample prompts
- [ ] Training curves compared
- [ ] Tool usage verified (Python + Search)
- [ ] Ready for benchmark evaluation

---

## 🎓 Next Steps After Training

### 1. Benchmark Evaluation

Evaluate your trained models on 13 benchmarks:
- **Math**: AIME24, AIME25, MATH500, GSM8K
- **Knowledge**: HotpotQA, 2Wiki, Musique, Bamboogle
- **Search**: GAIA, HLE, SimpleDeepSearch, WebDancer

See the main README.md for evaluation instructions.

### 2. Model Comparison Analysis

- Compare ARPO vs AEPO on specific tasks
- Analyze tool usage patterns
- Identify strengths and weaknesses
- Generate performance reports

### 3. Fine-Tuning (Optional)

If performance is below target:
- Train for more epochs (3-5)
- Adjust hyperparameters
- Try different reward functions
- Scale up to 14B model

### 4. Deployment

Deploy your models for:
- API inference services
- Interactive demo applications
- Further research experiments

---

## 🔗 Additional Resources

### Documentation
- **ARPO Paper**: https://arxiv.org/abs/2507.19849
- **AEPO Paper**: https://arxiv.org/abs/2510.14545
- **VERL Framework**: https://github.com/volcengine/verl
- **LLaMA-Factory**: https://github.com/hiyouga/LLaMA-Factory

### Model Checkpoints
- **ARPO Models**: https://huggingface.co/collections/dongguanting/arpo-688229ff8a6143fe5b4ad8ae
- **AEPO Models**: https://huggingface.co/collections/dongguanting/aepo-68ef6832c99697ee03d5e1c7

### Datasets
- **ARPO-SFT-54K**: https://huggingface.co/datasets/dongguanting/ARPO-SFT-54K
- **ARPO-RL-Reasoning-10K**: https://huggingface.co/datasets/dongguanting/ARPO-RL-Reasoning-10K
- **ARPO-RL-DeepSearch-1K**: https://huggingface.co/datasets/dongguanting/ARPO-RL-DeepSearch-1K

---

## 📞 Support

For issues or questions:

1. **Check Troubleshooting Section** in `Complete_RL_Training_Guide.md`
2. **Review CLAUDE.md** in the repository root
3. **Existing RL Training Guide** at `.trae/documents/RL_Training_Guide.md`
4. **GitHub Issues**: https://github.com/RUC-NLPIR/ARPO/issues

---

## 📄 License

This documentation follows the project license (MIT). See LICENSE file in repository root.

---

**Good luck with your RL training!** 🚀

Remember to start with the local toy example to verify everything works before investing time in full cluster training.
