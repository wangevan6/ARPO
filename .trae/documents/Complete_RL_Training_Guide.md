# ARPO/AEPO RL Training - Quickstart Guide

**Quick Reference**: Step-by-step commands to execute the complete RL training pipeline

**Full Documentation**: See `Complete_RL_Training_Guide.md` for detailed explanations

***

## Prerequisites

* [x] SFT training completed (or using base models)

* [x] Bright Data API credentials (API key + zone)

* [x] HuggingFace account with `hf` CLI configured

* [x] WSL2 with 16GB GPU (local) + Linux cluster with 8x 40GB GPUs

***

## STAGE 1: Local Toy Testing (1-2 hours)

**Goal**: Verify RL pipeline works before cluster deployment

### 1.1 Setup Environment

```bash
cd C:/Users/user/Projects/ARPO/ARPO

# Create and activate conda environment
conda create -n arpo python==3.10 -y
conda activate arpo

# Install PyTorch (CUDA 12.8 for WSL2)
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# Install dependencies
pip install flash-attn --no-build-isolation
pip install -r requirements.txt

# Install VERL
cd verl_arpo_entropy && pip install -e . && cd ..

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### 1.2 Download Models

```bash
# Download 3B model (try first)
hf download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/Qwen2.5-3B-Instruct

# Download 0.5B model (fallback if 3B OOMs)
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./models/Qwen2.5-0.5B-Instruct
```

### 1.3 Download Dataset & Create Toy Subset

```bash
# Download full dataset
mkdir -p rl_datasets
hf download dongguanting/ARPO-RL-Reasoning-10K train_10k.parquet \
  --repo-type dataset --local-dir ./temp_rl
mv temp_rl/train_10k.parquet rl_datasets/train.parquet

# Create 100-sample toy dataset
python scripts/create_toy_dataset.py --num_samples 100
```

### 1.4 Configure API Keys

**Get Bright Data credentials** at <https://brightdata.com/>

**Update API keys in 3 locations:**

1. `scripts/config/ppo_trainer.yaml`
2. `verl_arpo_entropy/verl/workers/rollout/tools/config_example.yaml`
3. Training scripts (see below)

Find your conda path:

```bash
which conda
# Example: /home/user/anaconda3/bin/conda → Use: /home/user/anaconda3
```

### 1.5 Create Toy Training Scripts

**Note**: Due to the length of the training scripts, refer to the complete guide in `Complete_RL_Training_Guide.md` Section 1.5-1.6 for the full scripts.

**Short version - Update these variables in the scripts:**

In `scripts/ARPO_3B_Toy_Local.sh`:

```bash
BING_API_KEY="YOUR_API_KEY_HERE"     # ← UPDATE
BING_ZONE="YOUR_ZONE_HERE"           # ← UPDATE
CONDA_PATH="/home/user/anaconda3"    # ← UPDATE
```

Same for `scripts/ARPO_0.5B_Toy_Local.sh`.

**Make scripts executable:**

```bash
chmod +x scripts/ARPO_3B_Toy_Local.sh
chmod +x scripts/ARPO_0.5B_Toy_Local.sh
```

### 1.6 Run Local Training

**Try 3B first:**

```bash
bash scripts/ARPO_3B_Toy_Local.sh
```

**Monitor GPU memory** (in another terminal):

```bash
watch -n 1 nvidia-smi
```

**If 3B OOMs (CUDA out of memory):**

```bash
# GPU usage (all 8 should be active)
watch -n 1 nvidia-smi

# Training progress
tail -f /data/checkpoints/ARPO_Qwen2.5_7B_Reasoning/run.log

# Search for reward
grep "Reward" /data/checkpoints/ARPO_Qwen2.5_7B_Reasoning/run.log
```

**Expected**: Training completes 1 epoch in 20-60 minutes

### 1.7 Verify Success

```bash
# Start tmux session
tmux new -s arpo_training

# Activate environment
conda activate arpo

# Launch training
bash scripts/ARPO_7B_Production_Cluster.sh

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t arpo_training
```

**Success criteria:**

* ✅ Training completed without crashes

* ✅ Rollout generated responses with `<python>` or `<search>` tags

* ✅ Reward scores computed (0-1 range)

* ✅ Checkpoint saved

***

## STAGE 2: Cluster ARPO Training (4-6 hours)

**Goal**: Train production ARPO with 7B model on full dataset

### 2.1 Transfer to Cluster

```bash
# SSH to cluster
ssh user@your-cluster-ip

# Clone repo
cd ~
git clone https://github.com/dongguanting/ARPO.git
cd ARPO/ARPO

# Setup environment
conda create -n arpo python==3.10 -y
conda activate arpo

# Install PyTorch (CUDA 12.4 for cluster)
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip install -r requirements.txt

# Install VERL
cd verl_arpo_entropy && pip install -e . && cd ..

# Verify 8 GPUs
nvidia-smi
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### 2.2 Download Model & Dataset

```bash
# Hardware
NNODES=1
N_GPUS_PER_NODE=8

# Data
TRAIN_BATCH_SIZE=128
PPO_MINI_BATCH_SIZE=16
TOTAL_EPOCHS=2

# Model
ACTOR_MODEL_PATH="/data/models/Qwen2.5-7B-Instruct"  # ← UPDATE

# Paths
SAVE_PATH="/data/checkpoints/ARPO_Qwen2.5_7B_Reasoning"  # ← UPDATE

# API Keys
BING_API_KEY="YOUR_API_KEY_HERE"  # ← UPDATE
BING_ZONE="YOUR_ZONE_HERE"         # ← UPDATE
CONDA_PATH="/home/user/anaconda3"  # ← UPDATE
WANDB_API_KEY=""                   # ← Optional
```

### 2.3 Create Production ARPO Script

**Note**: See `Complete_RL_Training_Guide.md` Section 2.5 for the full production script.

**Short version - Create** **`scripts/ARPO_7B_Production_Cluster.sh`** **with these key settings:**

```bash
# Download 7B model (or use your SFT checkpoint)
mkdir -p /data/models
hf download Qwen/Qwen2.5-7B-Instruct --local-dir /data/models/Qwen2.5-7B-Instruct

# Download full dataset
hf download dongguanting/ARPO-RL-Reasoning-10K \
  --repo-type dataset --local-dir ./rl_datasets

# Create search cache
mkdir -p search_cache
echo '{}' > search_cache/search_cache.json
```

Make executable:

```bash
# SSH to cluster
ssh user@your-cluster-ip

# Clone repo
cd ~
git clone https://github.com/dongguanting/ARPO.git
cd ARPO/ARPO

# Setup environment
conda create -n arpo python==3.10 -y
conda activate arpo

# Install PyTorch (CUDA 12.4 for cluster)
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip install -r requirements.txt

# Install VERL
cd verl_arpo_entropy && pip install -e . && cd ..

# Verify 8 GPUs
nvidia-smi
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### 2.4 Run Production Training

```bash
# Check logs
tail -50 checkpoints_toy/ARPO_*/run.log

# Check checkpoints
ls -lh checkpoints_toy/ARPO_*/global_step_*/

# Record which model worked
echo "Used model: [3B or 0.5B]" >> training_notes.txt
```

### 2.5 Monitor Training

```bash
ray stop --force  # Clean up
bash scripts/ARPO_0.5B_Toy_Local.sh  # Use fallback
```

**Expected**:

* Duration: 4-6 hours (2 epochs, \~78 steps/epoch)

* Reward: 0.3 → 0.6-0.8

* All 8 GPUs active

### 2.6 Convert Checkpoint to HuggingFace

```bash
cd ~/ARPO/ARPO/merge_ckpt

# Edit convert script with correct paths
nano convert_checkpoint_from_verl_to_hf_qwen3.sh

# Update:
# VERL_CHECKPOINT="/data/checkpoints/ARPO_Qwen2.5_7B_Reasoning/global_step_156"
# OUTPUT_DIR="/data/hf_checkpoints/qwen2.5-7b-arpo"

# Run conversion
bash convert_checkpoint_from_verl_to_hf_qwen3.sh

# Verify
python -c "from transformers import AutoModelForCausalLM; \
  model = AutoModelForCausalLM.from_pretrained('/data/hf_checkpoints/qwen2.5-7b-arpo'); \
  print('✅ ARPO model loaded successfully!')"
```

***

## STAGE 3: Cluster AEPO Training (4-6 hours)

**Goal**: Train AEPO with entropy-balancing mechanisms

### 3.1 Setup AEPO Environment

```bash
cd ~/ARPO/AEPO

# Create new environment
conda create -n aepo python==3.10 -y
conda activate aepo

# Install dependencies (same as ARPO)
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip install -r requirements.txt

# Install VERL
cd verl_aepo_entropy && pip install -e . && cd ..
```

### 3.2 Create Production AEPO Script

**Note**: See `Complete_RL_Training_Guide.md` Section 3.2 for the full script.

**Short version - Create** **`scripts/AEPO_7B_Production_Cluster.sh`** **with:**

```bash
chmod +x scripts/ARPO_3B_Toy_Local.sh
chmod +x scripts/ARPO_0.5B_Toy_Local.sh
```

Make executable:

```bash
BING_API_KEY="YOUR_API_KEY_HERE"     # ← UPDATE
BING_ZONE="YOUR_ZONE_HERE"           # ← UPDATE
CONDA_PATH="/home/user/anaconda3"    # ← UPDATE
```

### 3.3 Run AEPO Training

```bash
which conda
# Example: /home/user/anaconda3/bin/conda → Use: /home/user/anaconda3
```

### 3.4 Convert AEPO Checkpoint

Same process as ARPO:

```bash
# Download full dataset
mkdir -p rl_datasets
hf download dongguanting/ARPO-RL-Reasoning-10K train_10k.parquet \
  --repo-type dataset --local-dir ./temp_rl
mv temp_rl/train_10k.parquet rl_datasets/train.parquet

# Create 100-sample toy dataset
python scripts/create_toy_dataset.py --num_samples 100
```

***

## STAGE 4: Verification (30 min)

### 4.1 Test Models

```bash
# Download 3B model (try first)
hf download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/Qwen2.5-3B-Instruct

# Download 0.5B model (fallback if 3B OOMs)
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./models/Qwen2.5-0.5B-Instruct
```

### 4.2 Compare Models

```bash
cd C:/Users/user/Projects/ARPO/ARPO

# Create and activate conda environment
conda create -n arpo python==3.10 -y
conda activate arpo

# Install PyTorch (CUDA 12.8 for WSL2)
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# Install dependencies
pip install flash-attn --no-build-isolation
pip install -r requirements.txt

# Install VERL
cd verl_arpo_entropy && pip install -e . && cd ..

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

***

## Quick Troubleshooting

| Issue                   | Solution                                           |
| ----------------------- | -------------------------------------------------- |
| **3B OOM locally**      | Use 0.5B fallback script                           |
| **vLLM OOM**            | Reduce `gpu_memory_utilization=0.3`                |
| **Not all GPUs active** | Verify `N_GPUS_PER_NODE=8` in script               |
| **Ray fails to start**  | `ray stop --force && rm -rf /tmp/ray`              |
| **Reward always 0**     | Check reward function path and model output format |
| **Training diverges**   | Reduce learning rate to `5e-7`                     |

***

## Checklist

**Local Toy** (verify before cluster):

* [ ] Environment setup complete

* [ ] Models downloaded

* [ ] Toy dataset created

* [ ] API keys configured

* [ ] Training completed (3B or 0.5B)

**Cluster ARPO**:

* [ ] 7B model downloaded

* [ ] Full dataset (10K) downloaded

* [ ] All 8 GPUs verified

* [ ] Training completed (2 epochs)

* [ ] Checkpoint converted to HF

**Cluster AEPO**:

* [ ] Environment setup

* [ ] Training completed (2 epochs)

* [ ] Checkpoint converted to HF

**Verification**:

* [ ] Both models tested

* [ ] Tool usage verified

* [ ] Performance compared

***

## Timeline

| Stage        | Duration                            |
| ------------ | ----------------------------------- |
| Local Toy    | 1-2 hours                           |
| Cluster ARPO | 4-6 hours                           |
| Cluster AEPO | 4-6 hours                           |
| Verification | 30 min                              |
| **Total**    | **10-15 hours** (mostly unattended) |

***

## Next Steps

After successful training:

1. **Evaluation**: Test on 13 benchmarks (AIME, MATH500, GSM8K, GAIA, HLE, etc.)
2. **Deployment**: Use HF checkpoints for inference
3. **Analysis**: Compare ARPO vs AEPO performance

**Congratulations!** You've successfully completed the complete RL training pipeline.

For detailed explanations and troubleshooting, see: `Complete_RL_Training_Guide.md`
