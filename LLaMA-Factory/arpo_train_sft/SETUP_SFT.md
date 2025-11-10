# ARPO SFT Training Setup Guide

## Overview

This guide provides step-by-step instructions for setting up and executing Supervised Fine-Tuning (SFT) for the ARPO project using a **two-stage approach**:

1. **Stage 1 (Local Toy Testing)**: Validate setup with a small 0.5B model on 1 GPU
2. **Stage 2 (Production Training)**: Full-scale training with 3B model on 8 GPUs

## Prerequisites

- **Operating System**: Linux (Ubuntu 20.04+ recommended) or Windows with WSL2
- **CUDA**: Version 11.6+ (12.4 recommended)
- **conda**: Anaconda or Miniconda installed
- **Git**: For cloning repository
- **Disk Space**:
  - Local (Stage 1): ~10GB
  - Server (Stage 2): ~60GB

## Hardware Requirements

### Stage 1: Local Toy Testing
- **GPUs**: 1 GPU with 8GB+ VRAM (e.g., RTX 3070, RTX 4060 Ti)
- **RAM**: 16GB system memory
- **Expected Duration**: 10-30 minutes

### Stage 2: Production Training
- **GPUs**: 8 GPUs with 24-48GB VRAM each (e.g., A6000, L40, A100)
- **RAM**: 128GB+ system memory
- **Expected Duration**: 8-16 hours

---

## STAGE 1: Local Toy Testing

### Step 1: Environment Setup

```bash
# Navigate to LLaMA-Factory directory
cd ARPO/LLaMA-Factory

# Create conda environment
conda create -n sft python=3.10 -y
conda activate sft

# Install PyTorch with CUDA support
pip3 install torch==2.4.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install LLaMA-Factory dependencies
pip install -r requirements.txt

# Optional: Install flash-attention for faster training
pip install flash-attn --no-build-isolation
```

### Step 2: Verify Installation

```bash
# Test GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Test transformers installation
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# Test deepspeed installation
python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"
```

### Step 3: Download ARPO-SFT-54K Dataset

```bash
# Login to HuggingFace (you'll need an account)
huggingface-cli login

# Download the full dataset
huggingface-cli download dongguanting/ARPO-SFT-54K --local-dir ./temp_dataset

# Move to correct location
mv ./temp_dataset/*.json ./data/final_sft_edition9.json

# Or if it's a different format, find and move the data file
find ./temp_dataset -name "*.json" -o -name "*.jsonl" | head -1
```

### Step 4: Create Toy Dataset Subset

```bash
# Navigate to training directory
cd arpo_train_sft

# Run the toy dataset extraction script
python create_toy_dataset.py \
    --num_samples 500 \
    --input_file ../data/final_sft_edition9.json \
    --output_file ../data/final_sft_edition9_toy.jsonl

# Verify toy dataset created
ls -lh ../data/final_sft_edition9_toy.jsonl
```

### Step 5: Download Qwen2.5-0.5B Model

```bash
# Option 1: Download with HuggingFace CLI (recommended)
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./models/Qwen2.5-0.5B-Instruct

# Option 2: Auto-download during training (requires internet)
# Just use "Qwen/Qwen2.5-0.5B-Instruct" in config file
```

### Step 6: Configure Toy Training

Edit `arpo_train_sft/yaml/qwen_toy.yaml`:

```bash
# Open with your preferred editor
nano yaml/qwen_toy.yaml  # or vim, code, etc.
```

Update the following line:
```yaml
model_name_or_path: ./models/Qwen2.5-0.5B-Instruct  # Update to your actual path
```

If you used Option 2 above (auto-download):
```yaml
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
```

### Step 7: Launch Toy Training

```bash
# Make script executable
chmod +x sft_train_toy.sh

# Launch training
bash sft_train_toy.sh
```

**Monitor Training**:
```bash
# In another terminal, watch GPU usage
watch -n 1 nvidia-smi

# Or monitor training log
tail -f ./checkpoints/qwen0.5b_toy/training.log
```

### Step 8: Validate Toy Training

**Check training completed successfully**:
```bash
# List checkpoints
ls -lh ./checkpoints/qwen0.5b_toy/

# Check final checkpoint
ls -lh ./checkpoints/qwen0.5b_toy/checkpoint-*/
```

**Quick inference test**:
```bash
# Test the trained model
cd ../  # Go back to LLaMA-Factory root

python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = './arpo_train_sft/checkpoints/qwen0.5b_toy'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='auto')

prompt = 'Hello, how are you?'
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"
```

### Stage 1 Troubleshooting

**Issue: CUDA Out of Memory**
```bash
# Solution 1: Reduce batch size
# Edit yaml/qwen_toy.yaml, change:
per_device_train_batch_size: 1  # Reduce from 2 to 1

# Solution 2: Reduce sequence length
cutoff_len: 2048  # Reduce from 4096
```

**Issue: Dataset not found**
```bash
# Verify dataset exists
ls -lh ../data/final_sft_edition9_toy.jsonl

# Verify dataset_info.json is correct
cat dataset_info/dataset_info.json | grep -A 6 "arpo_sft_toy"
```

**Issue: Model download fails**
```bash
# Use mirror (for users in regions with limited access)
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./models/Qwen2.5-0.5B-Instruct
```

---

## STAGE 2: Production Training on Server

### Step 1: Replicate Environment on Server

```bash
# SSH into your server
ssh user@your-server

# Navigate to project
cd ARPO/LLaMA-Factory

# Create identical conda environment
conda create -n sft python=3.10 -y
conda activate sft

# Install dependencies (same as Stage 1)
pip3 install torch==2.4.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Step 2: Verify 8 GPUs Available

```bash
# Check all 8 GPUs visible
nvidia-smi

# Verify CUDA sees all GPUs
python -c "import torch; print(f'Available GPUs: {torch.cuda.device_count()}')"
```

### Step 3: Download Full Dataset

**Option A: Download directly on server**
```bash
cd arpo_train_sft

# Login to HuggingFace
huggingface-cli login

# Download dataset
huggingface-cli download dongguanting/ARPO-SFT-54K --local-dir ../temp_dataset

# Move to correct location
mv ../temp_dataset/*.json ../data/final_sft_edition9.json
```

**Option B: Transfer from local machine**
```bash
# On local machine
scp ./data/final_sft_edition9.json user@your-server:~/ARPO/LLaMA-Factory/data/
```

### Step 4: Download Qwen2.5-3B Model

```bash
# Download to server (requires ~6GB)
huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
    --local-dir /data/models/Qwen2.5-3B-Instruct

# Or use different path based on your server setup
```

### Step 5: Configure Production Training

Edit `arpo_train_sft/yaml/qwen_3b.yaml`:

```bash
cd arpo_train_sft
nano yaml/qwen_3b.yaml
```

Update these lines:
```yaml
model_name_or_path: /data/models/Qwen2.5-3B-Instruct  # Your actual model path
output_dir: /data/checkpoints/qwen2.5-3b-sft          # Your checkpoint directory
```

Edit `arpo_train_sft/sft_train_3b.sh`:

```bash
nano sft_train_3b.sh
```

Update:
```bash
OUTPUT_DIR="/data/checkpoints/qwen2.5-3b-sft"  # Match yaml config
```

### Step 6: Dry Run Test (Recommended)

Before running full training, test with 1000 samples:

```bash
# Edit qwen_3b.yaml temporarily
nano yaml/qwen_3b.yaml

# Change max_samples
max_samples: 1000  # Temporary - change back to 1000000 after test
```

```bash
# Run test
chmod +x sft_train_3b.sh
bash sft_train_3b.sh
```

**Verify**:
- All 8 GPUs show activity in `nvidia-smi`
- No OOM errors
- Training progresses smoothly
- Checkpoint saves successfully

**If successful, stop training (Ctrl+C) and revert**:
```bash
nano yaml/qwen_3b.yaml
# Change back: max_samples: 1000000
```

### Step 7: Launch Full Production Training

```bash
# Ensure you're in the right directory
cd arpo_train_sft

# Launch training (this will run for 8-16 hours)
bash sft_train_3b.sh
```

**Alternative: Run in background with nohup**:
```bash
nohup bash sft_train_3b.sh > training_output.log 2>&1 &

# Get process ID
echo $!

# Monitor progress
tail -f training_output.log
```

**Alternative: Run in tmux/screen session** (recommended):
```bash
# Create tmux session
tmux new -s sft_training

# Inside tmux, launch training
bash sft_train_3b.sh

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t sft_training
```

### Step 8: Monitor Production Training

**Monitor GPU usage**:
```bash
# In a separate terminal/tmux pane
watch -n 1 nvidia-smi
```

**Monitor training progress**:
```bash
# Watch training log
tail -f /data/checkpoints/qwen2.5-3b-sft/training.log

# Or if using nohup
tail -f training_output.log
```

**Monitor checkpoints**:
```bash
# Watch checkpoint directory size
watch -n 60 'du -sh /data/checkpoints/qwen2.5-3b-sft/*'
```

**Expected milestones** (54K samples, batch size 16, 3 epochs):
- Total steps: ~10,125 (54,000 ÷ 16 × 3)
- Checkpoint 1 (step 2000): ~1.5-2.5 hours
- Checkpoint 2 (step 4000): ~3-5 hours
- Checkpoint 3 (step 6000): ~4.5-7.5 hours
- Checkpoint 4 (step 8000): ~6-10 hours
- Final checkpoint: ~8-16 hours

### Step 9: Verify Completion

```bash
# Check final checkpoint exists
ls -lh /data/checkpoints/qwen2.5-3b-sft/

# Verify all model files present
ls -lh /data/checkpoints/qwen2.5-3b-sft/*.safetensors
ls -lh /data/checkpoints/qwen2.5-3b-sft/config.json
ls -lh /data/checkpoints/qwen2.5-3b-sft/tokenizer*

# Check training completed all epochs
grep -i "epoch 3" /data/checkpoints/qwen2.5-3b-sft/training.log
```

### Step 10: Test Trained Model

Quick inference test:

```bash
cd ../  # Go to LLaMA-Factory root

python << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model_path = "/data/checkpoints/qwen2.5-3b-sft"
print(f"Loading model from: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map='auto',
    torch_dtype='auto'
)

# Test generation
test_prompts = [
    "Explain reinforcement learning in simple terms.",
    "What is the capital of France?",
    "Write Python code to calculate factorial."
]

for prompt in test_prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")

print("\n✓ Model inference test completed!")
EOF
```

### Stage 2 Troubleshooting

**Issue: OOM errors during training**
```bash
# Solution 1: Reduce batch size
# Edit yaml/qwen_3b.yaml:
per_device_train_batch_size: 1
gradient_accumulation_steps: 4  # Keep effective batch size = 32

# Solution 2: Reduce sequence length
cutoff_len: 12000  # Reduce from 15000
```

**Issue: Slow training speed**
```bash
# Verify all GPUs utilized
nvidia-smi

# Check if flash-attention installed
python -c "import flash_attn; print('Flash Attention installed')"

# Enable gradient checkpointing (may help memory, but slows training)
# Not recommended unless necessary
```

**Issue: Training interrupted**
```bash
# Resume from last checkpoint
# Edit yaml/qwen_3b.yaml, add:
resume_from_checkpoint: /data/checkpoints/qwen2.5-3b-sft/checkpoint-XXXX

# Or use --resume_from_checkpoint flag
```

---

## Configuration File Reference

### Toy Config (`qwen_toy.yaml`)
```yaml
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
dataset: arpo_sft_toy              # 500 samples
cutoff_len: 4096                   # Shorter sequences
num_train_epochs: 1.0              # Quick validation
per_device_train_batch_size: 2     # Larger batch for small model
save_steps: 50                     # Frequent checkpoints
deepspeed: ds_z2_config.json       # ZeRO-2 sufficient
```

### Production Config (`qwen_3b.yaml`)
```yaml
model_name_or_path: Qwen/Qwen2.5-3B-Instruct
dataset: arpo_sft_54k              # 54K samples
cutoff_len: 15000                  # Full sequences
num_train_epochs: 3.0              # Complete training
per_device_train_batch_size: 1     # Conservative for GPU memory
save_steps: 2000                   # Less frequent checkpoints
deepspeed: ds_z3_offload_config.json  # ZeRO-3 + CPU offload
```

## Dataset Format

The ARPO-SFT dataset uses **ShareGPT format**:

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "User question or instruction"
    },
    {
      "from": "gpt",
      "value": "Model response with potential tool calls like <python>code</python> or <search>query</search>"
    },
    {
      "from": "human",
      "value": "<result>tool execution result</result>"
    },
    {
      "from": "gpt",
      "value": "Final answer based on tool result"
    }
  ],
  "system": "Optional system prompt"
}
```

## Next Steps After SFT

Once SFT training completes successfully:

1. **Record checkpoint path** for RL training
2. **Run inference tests** to validate model quality
3. **Proceed to ARPO/AEPO RL training**:
   - Use SFT checkpoint as `actor_model_path`
   - Follow ARPO/AEPO training guides
   - Located in `ARPO/` or `AEPO/` directories

## Additional Resources

- **LLaMA-Factory Documentation**: https://github.com/hiyouga/LLaMA-Factory
- **Qwen Model Cards**: https://huggingface.co/Qwen
- **ARPO Paper**: https://arxiv.org/abs/2507.19849
- **DeepSpeed Documentation**: https://www.deepspeed.ai/

## Support

For issues or questions:
1. Check `CLAUDE.md` in repository root
2. Review LLaMA-Factory GitHub issues
3. Check training logs for error messages
4. Verify hardware meets minimum requirements
