# ARPO SFT Training Directory

This directory contains all configuration files and scripts for **Supervised Fine-Tuning (SFT)** as part of the ARPO training pipeline.

## Quick Navigation

- **[QUICKSTART.md](QUICKSTART.md)** - Essential commands to get started (recommended for most users)
- **[SETUP_SFT.md](SETUP_SFT.md)** - Comprehensive setup guide with detailed explanations
- **[CLAUDE.md](../../../CLAUDE.md)** - Project-wide guidance for Claude Code

## Training Approach

We use a **two-stage validation approach**:

### Stage 1: Local Toy Testing
- **Model**: Qwen2.5-0.5B-Instruct
- **Dataset**: 500 samples (toy subset)
- **Hardware**: 1 GPU
- **Duration**: 10-30 minutes
- **Purpose**: Validate environment setup before scaling up

### Stage 2: Production Training
- **Model**: Qwen2.5-3B-Instruct
- **Dataset**: ARPO-SFT-54K (54,000 samples)
- **Hardware**: 8 GPUs
- **Duration**: 8-16 hours
- **Purpose**: Full SFT training for ARPO/AEPO RL

## Directory Structure

```
arpo_train_sft/
├── README.md                   # This file
├── QUICKSTART.md               # Quick start commands
├── SETUP_SFT.md               # Detailed setup guide
├── create_toy_dataset.py      # Script to create toy dataset subset
├── yaml/
│   ├── qwen.yaml              # Original config (reference)
│   ├── qwen_toy.yaml          # Stage 1: Toy training config
│   └── qwen_3b.yaml           # Stage 2: Production training config
├── dataset_info/
│   └── dataset_info.json      # Dataset registry (includes toy + full)
├── sft_train.sh               # Original training script (reference)
├── sft_train_toy.sh           # Stage 1: Toy training launcher
└── sft_train_3b.sh            # Stage 2: Production training launcher
```

## Quick Start

### For Stage 1 (Local Toy Testing):
```bash
# 1. Create toy dataset
python create_toy_dataset.py --num_samples 500

# 2. Edit config with model path
nano yaml/qwen_toy.yaml

# 3. Launch training
bash sft_train_toy.sh
```

### For Stage 2 (Production Training):
```bash
# 1. Edit config with model and output paths
nano yaml/qwen_3b.yaml
nano sft_train_3b.sh

# 2. Launch training
bash sft_train_3b.sh
```

See [QUICKSTART.md](QUICKSTART.md) for complete step-by-step commands.

## Key Files to Modify

Before training, you **must** update these files with your specific paths:

### Stage 1 (Toy):
- `yaml/qwen_toy.yaml`:
  - `model_name_or_path` → Path to Qwen2.5-0.5B-Instruct

### Stage 2 (Production):
- `yaml/qwen_3b.yaml`:
  - `model_name_or_path` → Path to Qwen2.5-3B-Instruct
  - `output_dir` → Your checkpoint directory
- `sft_train_3b.sh`:
  - `OUTPUT_DIR` → Match yaml config

## Configuration Comparison

| Parameter | Toy (Stage 1) | Production (Stage 2) |
|-----------|---------------|---------------------|
| Model | Qwen2.5-0.5B | Qwen2.5-3B |
| Dataset | arpo_sft_toy (500) | arpo_sft_54k (54K) |
| GPUs | 1 | 8 |
| Batch Size | 2 | 1 |
| Cutoff Length | 4096 | 15000 |
| Epochs | 1 | 3 |
| DeepSpeed | ZeRO-2 | ZeRO-3 + offload |
| Save Steps | 50 | 2000 |

## Expected Outputs

### Stage 1 Success Indicators:
- ✓ Training completes in 10-30 minutes
- ✓ Checkpoint saved to `./checkpoints/qwen0.5b_toy/`
- ✓ No CUDA OOM errors
- ✓ Loss decreases consistently

### Stage 2 Success Indicators:
- ✓ All 8 GPUs showing activity
- ✓ Training completes 3 epochs (~8-16 hours)
- ✓ Checkpoints saved every 2000 steps
- ✓ Final checkpoint at specified output directory

## Troubleshooting

**Stage 1 fails**: Check [SETUP_SFT.md](SETUP_SFT.md#stage-1-troubleshooting)

**Stage 2 fails**: Check [SETUP_SFT.md](SETUP_SFT.md#stage-2-troubleshooting)

**GPU OOM**: Reduce `per_device_train_batch_size` and/or `cutoff_len`

**Dataset errors**: Verify `dataset_info.json` and dataset file paths

## Next Steps

After successful SFT training:

1. Verify final checkpoint exists and is loadable
2. Test model inference quality
3. Proceed to **ARPO/AEPO RL training**:
   - Location: `../../ARPO/` or `../../AEPO/`
   - Use SFT checkpoint as `actor_model_path`
   - See RL training documentation

## Resources

- **ARPO Paper**: https://arxiv.org/abs/2507.19849
- **AEPO Paper**: https://arxiv.org/abs/2510.14545
- **LLaMA-Factory**: https://github.com/hiyouga/LLaMA-Factory
- **Qwen Models**: https://huggingface.co/Qwen
- **Dataset**: https://huggingface.co/datasets/dongguanting/ARPO-SFT-54K

## Questions?

1. Check [SETUP_SFT.md](SETUP_SFT.md) for detailed explanations
2. Review [QUICKSTART.md](QUICKSTART.md) for command reference
3. See project root [CLAUDE.md](../../../CLAUDE.md) for architecture overview
4. Check training logs for specific error messages
