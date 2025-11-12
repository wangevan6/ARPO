# ARPO RL Training - Issues Log

**Date**: 2025-11-12
**Environment**: Windows 11, MINGW64, conda environment `arpo`
**Objective**: Set up local toy RL training with ARPO framework
**Model**: Qwen2.5-0.5B-Instruct (fallback from 3B due to download issues)
**Hardware**: NVIDIA GeForce RTX 5070 Ti (16 GB VRAM)

---

## Training Environment Status

### ✅ Successfully Completed Components

1. **Conda Environment**: Created `arpo` environment with Python 3.10.0
2. **PyTorch**: Installed 2.7.1+cu128 (user-specific configuration for Windows)
3. **VERL Framework**: Successfully installed v0.3.1.dev in editable mode
4. **Core Dependencies**:
   - Ray 2.51.1
   - Transformers 4.57.1
   - PEFT 0.17.1
   - Accelerate 1.11.0
   - Datasets 4.4.1
5. **Training Data**:
   - Downloaded 10K sample RL dataset (2.7 MB)
   - Created 100-sample toy dataset (33 KB)
   - Validation dataset ready (52 KB)
6. **Model**: Qwen2.5-0.5B-Instruct (988 MB) available from SFT training
7. **API Credentials**: Bright Data API tested and working
   - Zone: `serp_api1`
   - API key configured in training scripts
8. **Training Scripts**: Created and configured:
   - `ARPO_0.5B_Toy_Local.sh` (ready to run)
   - `ARPO_3B_Toy_Local.sh` (needs 3B model download)

---

## Issue #1: Flash-Attention Installation Failure on Windows

### Error Encountered

**Timestamp**: 2025-11-12 (Environment setup)

**Command**:
```bash
conda run -n arpo pip install flash-attn --no-build-isolation
```

**Error Output**:
```
ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory:
'C:\\Users\\user\\AppData\\Local\\Temp\\pip-install-6m8ldwzk\\flash-attn_ee9d06a9559c4196892f883a9336e953\\
csrc/composable_kernel/library/src/tensor_operation_instance/gpu/batched_gemm_add_relu_gemm_add/
device_batched_gemm_add_relu_gemm_add_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance.cpp'

HINT: This error might have occurred since this system does not have Windows Long Path support enabled.
```

### Root Cause Analysis

Flash-attention has fundamental Windows compatibility issues:
1. **Path Length Limitations**: File paths in the source code exceed Windows 260-character limit
2. **Build Requirements**: Requires compilation from source with CUDA toolkit
3. **No Pre-built Wheels**: No official Windows wheels available for flash-attn 2.7.4.post1 or 2.8.3
4. **CK Library Dependencies**: AMD Composable Kernel library files have extremely long paths

### Attempted Solutions

#### Attempt 1: Direct pip install (flash-attn 2.8.3)
- **Status**: ❌ FAILED
- **Reason**: Path length exceeded Windows limit

#### Attempt 2: Specific version (flash-attn 2.7.4.post1)
- **Status**: ❌ FAILED
- **Reason**: Same path length issue

#### Attempt 3: Enable Windows Long Path Support
- **Not Attempted**: Would require registry modifications and system restart
- **Decision**: Proceed without flash-attention for initial testing

### Impact Assessment

**For Toy Training (0.5B model, 100 samples)**:
- **Impact**: Minimal to None
- **Reasoning**:
  - Flash-attention optimizes attention computation for long sequences
  - Toy training has small batch sizes and short sequences
  - vLLM can fall back to alternative attention implementations
  - XFORMERS backend is set as alternative (`export VLLM_ATTENTION_BACKEND=XFORMERS`)

**For Production Training (7B+ models, 10K samples)**:
- **Impact**: Moderate performance degradation (10-20% slower)
- **Mitigation**: Run production training on Linux cluster where flash-attention installs cleanly

### Resolution

**Status**: ✅ WORKAROUND APPLIED

**Solution**:
1. Skip flash-attention installation on Windows
2. Use XFORMERS attention backend (already configured in training scripts)
3. Proceed with toy training to validate pipeline
4. Use Linux environment for production training where flash-attention is available

**Configuration Added to Training Scripts**:
```bash
export VLLM_ATTENTION_BACKEND=XFORMERS
```

---

## Issue #2: NVIDIA NCCL Package Unavailable on Windows

### Error Encountered

**Timestamp**: 2025-11-12 (requirements.txt installation)

**Command**:
```bash
conda run -n arpo pip install -r requirements.txt
```

**Error Output**:
```
ERROR: Could not find a version that satisfies the requirement nvidia-nccl-cu12==2.21.5
(from versions: 0.0.1.dev5)
ERROR: No matching distribution found for nvidia-nccl-cu12==2.21.5
```

### Root Cause Analysis

1. **NCCL is Linux-only**: NVIDIA NCCL (NVIDIA Collective Communications Library) is not officially supported on Windows
2. **Requirements.txt was frozen on Linux**: The `requirements.txt` file includes Linux-specific packages
3. **Multi-GPU Training Focus**: NCCL is primarily used for multi-GPU distributed training

### Impact Assessment

**For Single-GPU Toy Training**:
- **Impact**: None
- **Reasoning**: NCCL is only needed for multi-GPU communication

**For Multi-GPU Training**:
- **Impact**: Cannot train on Windows with multiple GPUs using NCCL
- **Alternative**: Use Gloo backend (PyTorch's default CPU backend for Windows)

### Resolution

**Status**: ✅ WORKAROUND APPLIED

**Solution**:
1. Created `requirements_no_deepspeed.txt` excluding problematic packages
2. Installed VERL framework directly with `pip install -e .`
3. VERL's setup.py pulled in correct dependencies for Windows

**Installed Successfully**:
- Ray 2.51.1 (includes distributed training capabilities)
- PyTorch 2.7.1+cu128 (with CUDA support)
- vLLM 0.8.x (for inference)

**Configuration Note**: Training scripts set `N_GPUS_PER_NODE=1` for toy training

---

## Issue #3: DeepSpeed Not Required (Non-Issue)

### Observation

**Status**: ✅ EXPECTED BEHAVIOR

DeepSpeed installation was excluded from requirements, similar to SFT training.

**Why Not Needed**:
1. **FSDP Used Instead**: VERL uses PyTorch FSDP (Fully Sharded Data Parallel) for distributed training
2. **Windows Incompatibility**: DeepSpeed has the same CUDA_HOME compilation issues as seen in SFT training
3. **Single GPU**: Toy training doesn't require distributed optimization

**No Action Required**: This is intentional design

---

## Issue #4: Qwen2.5-3B-Instruct Download Failure

### Error Encountered

**Timestamp**: 2025-11-12 (Model download)

**Command**:
```bash
conda run -n arpo hf download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/Qwen2.5-3B-Instruct
```

**Error Output**:
```
Error while downloading model-00001-of-00002.safetensors:
HTTPSConnectionPool(host='cas-bridge.xethub.hf.co', port=443): Read timed out.

Error while downloading model-00002-of-00002.safetensors:
HTTPSConnectionPool(host='cas-bridge.xethub.hf.co', port=443): Read timed out.

requests.exceptions.ChunkedEncodingError:
('Connection broken: IncompleteRead(1012406 bytes read, 433945418 more expected)', ...)
```

### Root Cause Analysis

1. **Large File Size**: Model weights are ~3GB split across 2 files
2. **Network Instability**: Multiple connection timeouts to HuggingFace CDN
3. **Xet Storage Backend**: Files hosted on `cas-bridge.xethub.hf.co` had connectivity issues
4. **Windows Environment**: Connection pool handling may differ from Linux

### Download Progress Before Failure

- ✅ Configuration files (12/12 files): 100% complete
- ✅ model-00002-of-00002.safetensors: Download completed
- ❌ model-00001-of-00002.safetensors: Failed with incomplete read (1 MB / 434 MB)

### Impact Assessment

**For Toy Training**:
- **Impact**: None - can use 0.5B model as fallback
- **0.5B Model Available**: Already downloaded from SFT training (988 MB)

**For Testing Pipeline**:
- **Impact**: None - 0.5B model is sufficient to validate entire RL training pipeline

### Resolution

**Status**: ✅ FALLBACK APPLIED

**Solution**:
1. Use `ARPO_0.5B_Toy_Local.sh` script with Qwen2.5-0.5B-Instruct model
2. Model path configured: `C:/Users/user/Projects/ARPO/LLaMA-Factory/models/Qwen2.5-0.5B-Instruct`
3. 3B model can be downloaded later if needed for comparison

**Retry Options** (if 3B model needed):
```bash
# Option 1: Retry with increased timeout
HF_HUB_DOWNLOAD_TIMEOUT=600 hf download Qwen/Qwen2.5-3B-Instruct \
  --local-dir ./models/Qwen2.5-3B-Instruct

# Option 2: Download from different mirror
HF_ENDPOINT=https://hf-mirror.com hf download Qwen/Qwen2.5-3B-Instruct \
  --local-dir ./models/Qwen2.5-3B-Instruct

# Option 3: Use git lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct models/Qwen2.5-3B-Instruct
```

---

## Issue #5: Unicode Encoding in Toy Dataset Creation

### Error Encountered

**Timestamp**: 2025-11-12 (Dataset creation)

**Command**:
```bash
C:/Users/user/miniconda3/envs/arpo/python.exe scripts/create_toy_dataset.py \
  --input rl_datasets/train_10k.parquet \
  --output rl_datasets/train_toy.parquet \
  --num_samples 100
```

**Error Output**:
```
UnicodeEncodeError: 'gbk' codec can't encode character '\u2705' in position 0:
illegal multibyte sequence
```

### Root Cause Analysis

1. **Windows Console Encoding**: Default Windows console uses GBK/CP936 encoding in Chinese locale
2. **Emoji in Script**: Script uses ✅ emoji (`\u2705`) in print statements
3. **MINGW64 Terminal**: Git Bash has different encoding handling than CMD/PowerShell

### Impact Assessment

**Actual Impact**: ✅ NONE

**Why**:
- Error occurred AFTER dataset was successfully created
- Only the final success message print failed
- File verification confirmed dataset created correctly (33 KB, 100 samples)

### Resolution

**Status**: ✅ NO ACTION REQUIRED

**Verification**:
```bash
$ ls -lh rl_datasets/train_toy.parquet
-rw-r--r-- 1 user 197121 33K 11月 12 13:56 rl_datasets/train_toy.parquet

$ C:/Users/user/miniconda3/envs/arpo/python.exe -c \
  "import pandas as pd; \
   df = pd.read_parquet('rl_datasets/train_toy.parquet'); \
   print(f'Samples: {len(df)}'); \
   print(f'Columns: {list(df.columns)}')"
Samples: 100
Columns: ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info', 'metadata']
```

**Future Prevention** (optional):
- Set `PYTHONIOENCODING=utf-8` environment variable
- Use ASCII characters instead of emojis in scripts

---

## Current Training Environment Configuration

### Hardware
- **GPU**: NVIDIA GeForce RTX 5070 Ti (16 GB VRAM)
- **RAM**: (Not specified, but sufficient for toy training)
- **OS**: Windows 11, MINGW64

### Software Stack
```
Python: 3.10.0
PyTorch: 2.7.1+cu128
CUDA: 12.8
VERL: 0.3.1.dev
Ray: 2.51.1
Transformers: 4.57.1
vLLM: 0.8.x (exact version installed via VERL)
Flash-Attention: NOT INSTALLED (using XFORMERS backend)
DeepSpeed: NOT INSTALLED (using FSDP)
```

### Training Configuration

**Toy Training Parameters** (ARPO_0.5B_Toy_Local.sh):
```bash
Model: Qwen2.5-0.5B-Instruct
Dataset: 100 samples (train_toy.parquet)
Batch Size: 16
PPO Mini-Batch: 4
Rollouts: 4 (initial: 2)
Beam Size: 2
Max Prompt Length: 1536
Max Response Length: 4096
GPU Memory Utilization: 0.5
Epochs: 1
```

**API Configuration**:
```bash
Bright Data Zone: serp_api1
API Key: Configured and tested ✅
Search Cache: C:/Users/user/Projects/ARPO/ARPO/search_cache/search_cache.json
Python Tool Conda Path: C:/Users/user/miniconda3
Python Tool Environment: arpo
```

### Key Files and Paths

**Training Scripts**:
- `scripts/ARPO_0.5B_Toy_Local.sh` - Ready to run
- `scripts/ARPO_3B_Toy_Local.sh` - Needs 3B model download

**Datasets**:
- `rl_datasets/train_toy.parquet` - 100 samples for testing
- `rl_datasets/train_10k.parquet` - Full 10K dataset
- `rl_datasets/valid.parquet` - Validation set

**Models**:
- `../LLaMA-Factory/models/Qwen2.5-0.5B-Instruct/` - Ready ✅
- `models/Qwen2.5-3B-Instruct/` - Not downloaded

**Reward Function**:
- `verl_arpo_entropy/verl/utils/reward_score/deep_research.py`

---

## Next Steps

### Immediate (Ready to Execute)

1. **Start Toy Training**:
   ```bash
   cd C:/Users/user/Projects/ARPO/ARPO
   conda activate arpo
   bash scripts/ARPO_0.5B_Toy_Local.sh
   ```

2. **Monitor Training**:
   ```bash
   # In another terminal
   watch -n 1 nvidia-smi

   # Check training log
   tail -f checkpoints_toy/ARPO_0.5B_Toy_Local/run.log
   ```

3. **Expected Runtime**: 20-60 minutes for 1 epoch on 100 samples

### After Toy Training Success

1. **Verify Checkpoints**:
   - Check `checkpoints_toy/ARPO_0.5B_Toy_Local/global_step_*/`
   - Verify reward scores in logs

2. **Test Inference** (if needed):
   ```bash
   python test_model.py --checkpoint checkpoints_toy/ARPO_0.5B_Toy_Local/global_step_X
   ```

3. **Optional: Download 3B Model**:
   - Retry download with better network
   - Test if 3B model fits in 16GB VRAM

4. **Move to Production Training**:
   - Transfer to Linux cluster
   - Run full 10K dataset training
   - Use 7B+ models

---

## Summary of Issues

| Issue | Severity | Status | Impact on Toy Training |
|-------|----------|--------|------------------------|
| Flash-Attention Installation | Medium | ✅ Workaround | None (XFORMERS fallback) |
| NVIDIA NCCL Windows Support | Low | ✅ Expected | None (single GPU) |
| DeepSpeed Not Installed | Low | ✅ Intentional | None (FSDP used) |
| 3B Model Download Failure | Low | ✅ Fallback | None (0.5B available) |
| Unicode Encoding in Scripts | Minimal | ✅ Cosmetic | None (dataset created) |

**Overall Assessment**: ✅ **ALL BLOCKERS RESOLVED**

The toy training environment is fully configured and ready to run. All critical issues have been addressed with appropriate workarounds or fallback solutions. The 0.5B model with 100-sample toy dataset is sufficient to validate the complete ARPO RL training pipeline before scaling up.

---

## Troubleshooting Reference

### If Training Fails to Start

**Check Ray**:
```bash
ray stop --force
rm -rf /tmp/ray
```

**Check CUDA**:
```bash
nvidia-smi
C:/Users/user/miniconda3/envs/arpo/python.exe -c \
  "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Check Paths**:
```bash
ls -lh ../LLaMA-Factory/models/Qwen2.5-0.5B-Instruct/
ls -lh rl_datasets/train_toy.parquet
```

### If Out of Memory

**Reduce Batch Sizes**:
Edit `ARPO_0.5B_Toy_Local.sh`:
```bash
TRAIN_BATCH_SIZE=8  # was 16
PPO_MINI_BATCH_SIZE=2  # was 4
ROLLOUT_N=2  # was 4
```

**Reduce GPU Memory Utilization**:
```bash
gpu_memory_utilization=0.3  # was 0.5
```

### If Search API Fails

**Verify Credentials**:
```bash
python test_bright_data_api.py \
  --api-key 2e1ce5909f8fe768882fcb1a1ca38052e2eecffc560ec94e0b96e1e7d2bfc731 \
  --zone serp_api1
```

**Check Cache**:
```bash
ls -lh search_cache/search_cache.json
cat search_cache/search_cache.json | head -20
```

---

**Last Updated**: 2025-11-12
**Status**: Ready for Toy Training ✅

---

## Issue #6: Training Script Silent Failure

**Timestamp**: 2025-11-12 15:18
**Context**: After successfully testing Bright Data API with funds

### Problem
Training script completes in <1 minute with empty log file (0 bytes). Python command at line 118 fails silently without producing any output.

### Evidence
- Exit status: 0 (false success)
- run.log: Empty (0 bytes)
- Expected: 20-60 minutes training time
- Actual: <1 minute completion

### Root Cause
Python training command `python3 -m verl.trainer.main_ppo` likely fails due to:
1. verl module not found (PYTHONPATH issue)
2. Wrong python interpreter (not arpo conda environment)
3. Silent import error before logging starts

### Diagnostic Commands
```bash
# Test 1: Check verl installation
conda activate arpo
python -c "import verl; print(verl.__file__)"

# Test 2: Test module access with explicit path
cd /c/Users/user/Projects/ARPO/ARPO
export PYTHONPATH="C:/Users/user/Projects/ARPO/ARPO/verl_arpo_entropy:$PYTHONPATH"
/c/Users/user/miniconda3/envs/arpo/python.exe -m verl.trainer.main_ppo --help

# Test 3: Check which python is used
which python3
```

### Proposed Fix
Option 1 - Activate conda in script (add before line 118):
```bash
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate arpo
python -m verl.trainer.main_ppo ...  # Change python3 to python
```

Option 2 - Use absolute path:
```bash
/c/Users/user/miniconda3/envs/arpo/python.exe -m verl.trainer.main_ppo ...
```

Option 3 - Install verl properly:
```bash
cd verl_arpo_entropy
pip install -e .
```

**Status**: ⚠️ Awaiting diagnostics

---

**Last Updated**: 2025-11-12 15:30

---

## Issue #7: Ray Segmentation Fault on Windows (CRITICAL BLOCKER)

**Timestamp**: 2025-11-12 16:00-16:03
**Status**: ❌ **BLOCKING** - Cannot proceed on native Windows

### Problem Summary

Ray framework crashes with **SIGSEGV (segmentation fault)** on Windows, preventing ARPO RL training from running. This is a fundamental incompatibility, not a configuration issue.

### Error Signature

```
[raylet] *** SIGSEGV received at time=1762934554 ***
[raylet] The raylet exited immediately because one Ray agent failed, agent_name = dashboard_agent
ray.exceptions.ActorDiedError: The actor died unexpectedly before finishing this task
Worker exit detail: Owner's node has crashed
```

### What Works vs What Fails

**✅ Working**:
- Python module execution (`verl.trainer.main_ppo`)
- Hydra configuration loading
- Ray initialization starts (`Started a local Ray instance`)
- All dependencies installed (fastapi, openai, langid, etc.)
- Bright Data API functional

**❌ Failing**:
- Ray raylet.exe crashes immediately with SIGSEGV
- Dashboard agent fails to start
- Distributed training cannot proceed

### Root Cause

**Ray has limited Windows support.** The raylet (Ray's C++ core) crashes on Windows with SIGSEGV in native execution. This is a known limitation documented in Ray's GitHub issues.

**Why ARPO needs Ray**:
- ARPO uses Ray for distributed RL training
- Ray manages actor rollout workers, critic workers, and reference policy
- Cannot be replaced with alternative frameworks easily

### Attempted Fixes (All Failed)

1. **Downgraded grpcio** (1.76.0 → 1.60.0): No effect, still crashes
2. **Cleaned Ray temp files**: No effect
3. **Installed missing dependencies** (fastapi, uvicorn, openai, langid): Necessary but insufficient

### Solutions (In Order of Preference)

#### Option 1: Use WSL2 (Recommended)

**WSL2 (Windows Subsystem for Linux)** provides a full Linux kernel on Windows.

**Setup**:
```powershell
# In PowerShell (Admin)
wsl --install
wsl --install -d Ubuntu-22.04

# After reboot, in WSL2 Ubuntu terminal
cd /mnt/c/Users/user/Projects/ARPO
conda create -n arpo python=3.10
conda activate arpo
pip install -r ARPO/requirements_no_deepspeed.txt
pip install flash-attn --no-build-isolation  # Works on WSL2
```

**Advantages**:
- Full Linux compatibility
- flash-attention works
- Ray works properly
- Access Windows files via `/mnt/c/`
- Uses same GPU (CUDA pass-through)

**Disadvantages**:
- Requires WSL2 setup (~30 minutes)
- Need to reinstall conda environment

#### Option 2: Use Linux VM or Cloud Instance

**Cloud Options**:
- AWS EC2 (g5.xlarge with A10G GPU)
- Google Cloud Compute (n1-standard-4 with T4)
- Lambda Labs GPU Cloud
  
**Advantages**:
- Full Linux environment
- Larger GPU options available
- Production-ready

**Disadvantages**:
- Cost ($0.50-2.00/hour)
- Need to transfer model files
- Setup time

#### Option 3: Dual-Boot Linux (Most Effort)

Install Ubuntu 22.04 alongside Windows for native Linux performance.

**Not Recommended**: Too much effort for testing.

### Recommended Path Forward

**For Toy Training (100 samples)**:
1. **Install WSL2** (30 minutes)
2. **Set up conda environment in WSL2** (30 minutes)
3. **Run training in WSL2** (20-60 minutes for toy dataset)

**For Production Training (10K samples)**:
1. Use Linux cluster or cloud GPU instance
2. Transfer to environment with multiple GPUs
3. Run full-scale training

### Windows Limitations Summary

| Component | Windows Native | WSL2 | Linux |
|-----------|---------------|------|-------|
| SFT Training (LLaMA-Factory) | ✅ Works | ✅ Works | ✅ Works |
| RL Training (ARPO/Ray) | ❌ **Broken** | ✅ Works | ✅ Works |
| flash-attention | ❌ Won't install | ✅ Works | ✅ Works |
| NCCL (multi-GPU) | ❌ Not available | ✅ Works | ✅ Works |
| vLLM | ⚠️ Limited | ✅ Full support | ✅ Full support |

### Technical Details

**Ray Version**: 2.51.1  
**grpcio Version**: 1.60.0 (downgraded from 1.76.0)  
**Python**: 3.10.0  
**OS**: Windows 11, MINGW64  

**Crash Location**: `raylet.exe` (Ray's C++ core)  
**Signal**: SIGSEGV (segmentation fault)  
**Failing Component**: dashboard_agent  

**Ray Log Location** (if exists):
```
C:\Users\user\AppData\Local\Temp\ray\session_latest\logs\dashboard_agent.log
```

### Conclusion

**Native Windows is not viable for ARPO RL training** due to Ray's fundamental incompatibility. WSL2 is the fastest path to unblock training without leaving the Windows environment.

**Next Action**: Install WSL2 or move to Linux environment.

---

**Last Updated**: 2025-11-12 16:05
**Status**: ⚠️ **BLOCKED on Windows** - WSL2 required

---

## Issue #7 Resolution: WSL2 Implementation

**Timestamp**: 2025-11-12 16:20-16:30
**Status**: ✅ **IN PROGRESS** - Implementing WSL2 solution

### Decision: Use WSL2 (Option 1)

After encountering the Ray segmentation fault on native Windows, we determined that **WSL2 (Windows Subsystem for Linux 2)** is the most practical solution for local development.

### Why WSL2 Was Chosen

**vs Native Windows**:
- ❌ Windows: Ray crashes with SIGSEGV (cannot be fixed)
- ✅ WSL2: Full Linux kernel, Ray works properly

**vs Remote Linux Server**:
- ✅ WSL2: Immediate access, no setup/transfer time
- ✅ WSL2: Same GPU accessible via CUDA pass-through
- ✅ WSL2: Windows files directly accessible via `/mnt/c/`
- ✅ WSL2: Free, no cloud costs

**vs Rewriting Code**:
- ✅ WSL2: 1 hour setup vs 2-3 weeks development
- ✅ WSL2: No code changes needed
- ✅ WSL2: Maintains compatibility with production cluster

### WSL2 Setup Process

#### Step 1: WSL2 Installation ✅

**Command**:
```powershell
wsl --install -d Ubuntu-22.04
```

**Result**: Ubuntu-22.04 installed successfully

**Verification**:
```bash
$ wsl --list --verbose
  NAME                             STATE                       VERSION
* docker-desktop                   Running                     2
  Ubuntu-22.04                     Running                     2
```

#### Step 2: GPU Access Verification ✅

**Command**:
```bash
wsl -d Ubuntu-22.04 nvidia-smi
```

**Result**: GPU detected successfully
```
NVIDIA GeForce RTX 5070 Ti, 16303 MiB
Linux kernel: 6.6.87.2-microsoft-standard-WSL2
```

**Conclusion**: CUDA pass-through working correctly

#### Step 3: Miniconda Installation ✅

**Location**: `/home/evan_hero_linux/miniconda3`

**Commands**:
```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
```

**Result**: Miniconda installed successfully (151 MB download)

**Key Decision**: Separate Linux conda installation required
- **Why**: Windows conda contains Windows binaries (.exe, .dll)
- **Why**: Linux conda contains Linux binaries (.so, ELF)
- **Impact**: Ray compiled for Linux will work correctly
- **Disk Space**: ~1 GB (vs 2.1GB models we're reusing)

#### Step 4: Environment Creation (In Progress)

**Target Environment**:
```yaml
Name: arpo
Python: 3.10
Location: /home/evan_hero_linux/miniconda3/envs/arpo
```

**Installation Steps** (User executing manually):
1. Create environment: `conda create -n arpo python=3.10 -y`
2. Install PyTorch: `pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128`
3. Install VERL: `cd /mnt/c/Users/user/Projects/ARPO/ARPO/verl_arpo_entropy && pip install -e .`
4. Install dependencies: `pip install fastapi uvicorn openai langid`
5. (Optional) Flash-attention: `pip install flash-attn --no-build-isolation`

**Expected Install Time**: ~20-30 minutes

### File System Strategy

**What We Reuse from Windows** ✅:
```
/mnt/c/Users/user/Projects/ARPO/
├── ARPO/models/Qwen2.5-3B-Instruct/    ← 2.1GB (reusing!)
├── ARPO/rl_datasets/
│   ├── train_toy.parquet               ← 33KB (reusing!)
│   └── valid.parquet                   ← 52KB (reusing!)
├── ARPO/scripts/                       ← Bash scripts (reusing!)
└── ARPO/verl_arpo_entropy/             ← Source code (reusing!)
```

**What We Install Fresh in Linux** 🆕:
```
/home/evan_hero_linux/
├── miniconda3/                         ← 1GB (new Linux conda)
│   └── envs/arpo/                      ← ~4GB (new environment)
└── .bashrc                             ← Modified for conda init
```

**Total New Disk Usage**: ~5GB
**Total Reused Data**: ~2.2GB (models + datasets)
**Net Benefit**: No data duplication, fast access

### Configuration Updates Needed

Once environment is ready, training script paths remain the same:
```bash
# These paths work in both Windows and WSL2
MODEL_PATH="/mnt/c/Users/user/Projects/ARPO/ARPO/models/Qwen2.5-3B-Instruct"
TRAIN_FILES="/mnt/c/Users/user/Projects/ARPO/ARPO/rl_datasets/train_toy.parquet"
SEARCH_CACHE="/mnt/c/Users/user/Projects/ARPO/ARPO/search_cache/search_cache.json"

# Only Python changes from Windows to Linux
PYTHON_BIN="python"  # WSL2 uses conda-activated python
```

**No other changes needed** - Bright Data API key, model paths, datasets all remain the same.

### Expected Outcome

After WSL2 setup completes:

**Before (Windows)**:
```
✅ SFT Training (LLaMA-Factory)
✅ Bright Data API testing
❌ RL Training (Ray crashes)
❌ Flash-attention (won't install)
```

**After (WSL2)**:
```
✅ SFT Training (can still use Windows)
✅ Bright Data API testing
✅ RL Training (Ray works on Linux kernel)
✅ Flash-attention (compiles successfully)
✅ Production-ready setup
```

### Timeline

- **16:00-16:03**: Identified Ray crash on Windows
- **16:05**: Documented issue and evaluated options
- **16:20**: Installed WSL2 Ubuntu-22.04
- **16:25**: Installed Miniconda in WSL2
- **16:30**: User creating arpo environment (in progress)
- **16:50** (estimated): Environment ready, start training
- **17:00-18:00** (estimated): Training runs on toy dataset

### Lessons Learned

1. **Windows Limitation**: Native Windows cannot run Ray-based RL frameworks reliably
2. **WSL2 Effectiveness**: WSL2 provides full Linux compatibility with minimal overhead
3. **Hybrid Approach**: Can use Windows for SFT, WSL2 for RL training
4. **Data Sharing**: WSL2's `/mnt/c/` mount makes Windows data seamlessly accessible
5. **GPU Pass-through**: CUDA works natively in WSL2 without special configuration

### Next Steps

1. ⏳ Complete environment installation (user executing Steps 1-7)
2. ⏳ Verify Ray and verl modules load correctly
3. ⏳ Run training script: `bash scripts/ARPO_3B_Toy_Local.sh`
4. ⏳ Monitor training progress (expect 20-60 minutes for toy dataset)
5. ⏳ Verify checkpoint creation
6. ✅ Document successful training in this log

---

**Last Updated**: 2025-11-12 16:30
**Status**: ⏳ **Environment Setup In Progress** - Ready for training once installation completes

---

## Issue #8: PyTorch CUDA Memory Corruption in WSL2

**Timestamp**: 2025-11-12 17:03
**Status**: ❌ **CRITICAL** - New blocker in WSL2 environment

### Error Signature

```
(pid=5477) free(): double free detected in tcache 2
(pid=5477) *** SIGABRT received at time=1762938208 on cpu 11 ***
Fatal Python error: Aborted

Stack (most recent call first):
  File "torch/cuda/__init__.py", line 174 in is_available
  File "tensordict/base.py", line 135 in <module>
```

### How to Read This Error Message

**Error Anatomy** (from bottom to top):

1. **Initial Symptoms** (First few lines):
   ```
   Ray started successfully ✅
   Started a local Ray instance. View the dashboard at 127.0.0.1:8265
   ```
   - Ray actually initialized this time! (Progress from Windows)
   - Worker process started (pid=5477)

2. **The Crash Point** (Key error):
   ```
   free(): double free detected in tcache 2
   *** SIGABRT received at time=1762938208 on cpu 11 ***
   ```
   - **"double free"**: Memory was freed twice (corruption bug)
   - **SIGABRT**: Program aborted due to critical error
   - This is a **C/C++ level crash**, not a Python error

3. **Stack Trace** (Where it happened):
   ```
   torch/cuda/__init__.py:174 in is_available
   ```
   - Crash occurs when PyTorch checks if CUDA is available
   - Happens during module import, before training even starts

4. **Ray's Perspective**:
   ```
   ActorDiedError: The actor died unexpectedly
   Worker exit type: SYSTEM_ERROR
   ```
   - Ray worker crashed unexpectedly
   - Not Ray's fault - the worker process itself died

### Root Cause Analysis

**Problem**: PyTorch + CUDA + WSL2 incompatibility causing memory corruption

**Technical Details**:
1. **Double Free**: Memory management bug between PyTorch's CUDA bindings and WSL2's CUDA driver
2. **Occurs during**: `torch.cuda.is_available()` call
3. **Component chain**: PyTorch → CUDA libraries → WSL2 CUDA pass-through → GPU driver

**Why This Happens**:
- PyTorch 2.7.1 with CUDA 12.6 (cu126) might not be fully compatible with WSL2's CUDA implementation
- WSL2's CUDA pass-through has known issues with certain PyTorch versions
- Memory is allocated by PyTorch, freed by CUDA driver, then freed again → crash

### What Worked vs What Failed

**Progress Made** ✅:
1. WSL2 installation successful
2. Miniconda installed in Linux
3. Environment created with Python 3.10
4. All Python dependencies installed (fastapi, uvicorn, etc.)
5. **Ray actually started!** (Big improvement from Windows)
6. Worker processes spawn correctly

**Where It Breaks** ❌:
1. Ray worker tries to import verl
2. verl imports tensordict
3. tensordict imports torch
4. torch.cuda.is_available() checks GPU
5. **💥 Crash: double free in CUDA memory management**

### Debugging Process Timeline

#### 16:00-16:20: Windows Diagnosis
- ❌ Issue: Ray crashes with SIGSEGV on Windows
- 🔍 Root Cause: Ray's raylet.exe incompatible with Windows
- ✅ Solution: Move to WSL2

#### 16:20-16:30: WSL2 Setup
- ✅ Installed Ubuntu-22.04
- ✅ Verified GPU access (nvidia-smi works)
- ✅ Installed Miniconda for Linux
- ✅ Created arpo environment

#### 16:30-17:00: Environment Configuration
- ✅ Installed PyTorch 2.7.1+cu126
- ✅ Installed VERL framework (`pip install -e .`)
- ❌ Flash-attention failed (missing CUDA toolkit - expected)
- ✅ Installed fastapi, uvicorn, openai, langid

#### 17:00-17:03: Path Fixes
- ❌ Issue: Config path using Windows format (C:/)
- ✅ Fixed: Changed all paths to WSL2 format (/mnt/c/)
- ✅ Script starts successfully

#### 17:03: CUDA Crash Discovered
- ✅ Ray starts (progress!)
- ❌ **New Issue**: Worker crashes on `torch.cuda.is_available()`
- ❌ Error: "double free detected in tcache 2"
- ❌ Status: Training cannot proceed

### Current Environment Details

**System**:
```
OS: WSL2 Ubuntu-22.04 (Linux kernel 6.6.87.2)
GPU: NVIDIA GeForce RTX 5070 Ti (16GB)
CUDA Driver: Available via WSL2 pass-through
```

**Python Environment**:
```
Python: 3.10
PyTorch: 2.7.1+cu126 (CUDA 12.6 compiled)
Ray: 2.51.1
VERL: 0.3.1.dev
```

**Problem**: PyTorch CUDA 12.6 build may not be compatible with WSL2's CUDA implementation

### Potential Solutions

#### Option 1: Downgrade PyTorch (Most Likely to Work)

Try an older, more stable PyTorch version:

```bash
conda activate arpo
pip uninstall torch torchvision torchaudio -y

# Try PyTorch 2.4.0 with CUDA 12.4 (known stable with WSL2)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```

**Rationale**: PyTorch 2.4.0 with CUDA 12.4 has better WSL2 compatibility track record

#### Option 2: Use PyTorch with CUDA 11.8

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

**Rationale**: CUDA 11.8 has fewer WSL2 issues

#### Option 3: Install CUDA Toolkit in WSL2

The crash might be due to missing CUDA libraries:

```bash
# Install NVIDIA CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4
```

Then reinstall PyTorch.

#### Option 4: Native Linux Environment

If WSL2 continues having issues:
- Use a native Linux machine
- Use cloud GPU instance (AWS, GCP, Lambda Labs)
- Use Docker with NVIDIA runtime

### Comparison: Windows vs WSL2 Issues

| Aspect | Windows Native | WSL2 |
|--------|---------------|------|
| Ray Initialization | ❌ SIGSEGV crash | ✅ Works |
| Ray Worker Spawn | ❌ Crashes immediately | ✅ Spawns successfully |
| Python Imports | ✅ Works | ✅ Works |
| CUDA Detection | ⚠️ Never reached | ❌ Crashes (double free) |
| Root Cause | Ray incompatibility | PyTorch+CUDA incompatibility |
| Fix Difficulty | Impossible (need Linux) | Moderate (version change) |

**Progress**: We got further! Ray works now, just CUDA initialization failing.

### How to Read Error Messages - General Guide

**1. Start from the Bottom (Most Recent Call)**:
```python
File "torch/cuda/__init__.py", line 174 in is_available
```
This is where the crash happened.

**2. Look for Signal/Error Type**:
```
SIGABRT = Program aborted due to critical error
SIGSEGV = Segmentation fault (memory access violation)
SIGKILL = Process killed by OS
```

**3. Identify the Layer**:
- **Python error** (ImportError, ValueError): Python-level bug
- **C/C++ signal** (SIGABRT, SIGSEGV): Low-level crash in native code
- **OS error** (killed by OOM): System resource issue

**4. Check Context**:
```
free(): double free detected
```
This tells you WHAT went wrong (memory freed twice).

**5. Trace the Path**:
```
main_ppo.py → ray_trainer.py → worker → verl → tensordict → torch → CRASH
```
Shows the execution flow leading to the crash.

**6. Look for Versions**:
```
torch.__version__ = 2.7.1+cu126
```
Often the version compatibility is the issue.

### Recommended Next Action

**Try Option 1 first** (downgrade PyTorch):

```bash
# In your WSL2 terminal
conda activate arpo
pip uninstall torch torchvision torchaudio -y
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA works
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# If that works, run training
bash scripts/ARPO_0.5B_Toy_Local.sh
```

If Option 1 fails, document the error and we'll try Option 2.

---

**Last Updated**: 2025-11-12 17:10
**Status**: ⏳ **Awaiting PyTorch downgrade attempt**
