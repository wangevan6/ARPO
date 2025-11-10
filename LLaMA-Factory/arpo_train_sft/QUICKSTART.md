# ARPO SFT Training - Quick Start Guide

**Two-Stage Approach**: Test locally with toy dataset → Scale up to production on server

---

## STAGE 1: Local Toy Testing (Single GPU, 0.5B Model)

### 1. Setup Environment
```bash
cd ARPO/LLaMA-Factory
conda create -n sft python=3.10 -y
conda activate sft
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
huggingface-cli login
huggingface-cli download dongguanting/ARPO-SFT-54K --local-dir ./temp_dataset
mv ./temp_dataset/*.json ./data/final_sft_edition9.json
```

### 3. Create Toy Dataset
```bash
python ./arpo_train_sft/create_toy_dataset.py --num_samples 500
```h` |
| `sft_train_3b.sh` | Production script (8 GPUs) | `sft_train_3b.sh` |
| `dataset_info.json` | Dataset registry | `dataset_info/dataset_info.json` |
| `create_toy_dataset.py` | Create toy subset | `create_toy_dataset.py` |

---

## Quick Troubleshooting

### Stage 1 Issues

**OOM Error**:
```bash
# Edit yaml/qwen_toy.yaml
per_device_train_batch_size: 1
cutoff_len: 2048
```

**Dataset Not Found**:
```bash
ls -lh ../data/final_sft_edition9_toy.jsonl  # Should exist
python create_toy_dataset.py --num_samples 500  # Re-create if missing
```

### Stage 2 Issues

**OOM Error**:
```bash
# Edit yaml/qwen_3b.yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
cutoff_len: 12000
```

**Only Using 1 GPU**:
```bash
# Verify in sft_train_3b.sh
PROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

**Training Interrupted**:
```bash
# Resume from checkpoint
# Edit yaml/qwen_3b.yaml, add:
resume_from_checkpoint: /data/checkpoints/qwen2.5-3b-sft/checkpoint-XXXX
```

---

## Next Steps

After successful SFT training:

1. **Test inference** with the trained model
2. **Note checkpoint path** for RL training
3. **Proceed to ARPO/AEPO RL training** using this checkpoint as initialization

See `SETUP_SFT.md` for detailed documentation.


### 4. Download Model
```bash
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./models/Qwen2.5-0.5B-Instruct
```

### 5. Configure Training
```bash
# Edit yaml/qwen_toy.yaml
nano yaml/qwen_toy.yaml
# Set: model_name_or_path: ./models/Qwen2.5-0.5B-Instruct
```

### 6. Launch Training
```bash
chmod +x sft_train_toy.sh
bash sft_train_toy.sh
```

**Expected**: Completes in 10-30 minutes on 1 GPU

### 7. Verify Success
```bash
ls -lh ./checkpoints/qwen0.5b_toy/checkpoint-*/
tail -20 ./checkpoints/qwen0.5b_toy/training.log
```

---

## STAGE 2: Production Training (8 GPUs, 3B Model)

### 1. Setup Server Environment
```bash
ssh user@your-server
cd ARPO/LLaMA-Factory
conda create -n sft python=3.10 -y
conda activate sft
#pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124(Might cause error )
conda run -n sft pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### 2. Verify Hardware
```bash
nvidia-smi  # Should show 8 GPUs
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### 3. Download Dataset
```bash
cd arpo_train_sft
huggingface-cli login
huggingface-cli download dongguanting/ARPO-SFT-54K --local-dir ../temp_dataset
mv ../temp_dataset/*.json ../data/final_sft_edition9.json
```

### 4. Download Model
```bash
hf download Qwen/Qwen2.5-3B-Instruct --local-dir /data/models/Qwen2.5-3B-Instruct
```

### 5. Configure Training
```bash
# Edit yaml/qwen_3b.yaml
cd arpo_train_sft
nano yaml/qwen_3b.yaml
# Set: model_name_or_path: /data/models/Qwen2.5-3B-Instruct
# Set: output_dir: /data/checkpoints/qwen2.5-3b-sft

# Edit sft_train_3b.sh
nano sft_train_3b.sh
# Set: OUTPUT_DIR="/data/checkpoints/qwen2.5-3b-sft"
```

### 6. (Optional) Dry Run Test
```bash
# Temporarily edit qwen_3b.yaml: max_samples: 1000
bash sft_train_3b.sh
# Verify all 8 GPUs active, no OOM errors
# Ctrl+C to stop, revert max_samples: 1000000
```

### 7. Launch Full Training
```bash
# Recommended: Use tmux
tmux new -s sft_training
bash sft_train_3b.sh
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t sft_training
```

**Expected**: Completes in 8-16 hours on 8 GPUs

### 8. Monitor Progress
```bash
# Watch GPUs
watch -n 1 nvidia-smi

# Watch training log
tail -f /data/checkpoints/qwen2.5-3b-sft/training.log

# Check checkpoints
ls -lh /data/checkpoints/qwen2.5-3b-sft/checkpoint-*
```

### 9. Verify Completion
```bash
# Check final checkpoint
ls -lh /data/checkpoints/qwen2.5-3b-sft/*.safetensors
grep "Training completed" /data/checkpoints/qwen2.5-3b-sft/training.log
```

---

## Key Configuration Files

| File | Purpose | Location |
|------|---------|----------|
| `qwen_toy.yaml` | Toy training config (0.5B, 500 samples) | `yaml/qwen_toy.yaml` |
| `qwen_3b.yaml` | Production config (3B, 54K samples) | `yaml/qwen_3b.yaml` |
| `sft_train_toy.sh` | Toy training script (1 GPU) | `sft_train_toy.s