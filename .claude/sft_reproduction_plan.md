  ⎿  Two-Stage SFT Training Setup Plan

     STAGE 1: Local Toy Testing (Single GPU, 0.5B Model)

     Goal: Validate environment and workflow before full-scale training

     STAGE 2: Server Production Training (8 GPUs, 3B Model)

     Goal: Full SFT training reproduction

     ---
     STAGE 1: LOCAL TOY TESTING

     Phase 1: Environment Setup (Local Machine)

     1.1 Create Conda Environment
     - Create Python 3.10 conda environment named sft
     - Install dependencies from LLaMA-Factory/requirements.txt
     - Install PyTorch with CUDA support
     - Install flash-attention (optional for toy testing)

     1.2 Verify Installation
     - Test single GPU visibility (CUDA_VISIBLE_DEVICES=0)
     - Verify transformers, deepspeed, accelerate installed

     Phase 2: Data Preparation (Local)

     2.1 Download Full ARPO-SFT-54K Dataset
     - Login to HuggingFace CLI
     - Download dongguanting/ARPO-SFT-54K
     - Save to LLaMA-Factory/data/final_sft_edition9.json

     2.2 Create Toy Dataset Subset
     - Extract first 100-500 samples from full dataset
     - Save as LLaMA-Factory/data/final_sft_edition9_toy.jsonl
     - Maintain same ShareGPT format

     2.3 Configure Dataset Registry
     - Update arpo_train_sft/dataset_info/dataset_info.json
     - Add TWO entries:
       - arpo_sft_54k: Full dataset
       - arpo_sft_toy: Toy dataset (100-500 samples)

     Phase 3: Model Preparation (Local)

     3.1 Download Qwen2.5-0.5B-Instruct
     - Download from HuggingFace: Qwen/Qwen2.5-0.5B-Instruct
     - Save to local directory (e.g., ./models/Qwen2.5-0.5B-Instruct)
     - Much smaller: ~1GB vs ~6GB for 3B

     Phase 4: Training Configuration (Local Toy)

     4.1 Create Toy Training YAML
     - Copy arpo_train_sft/yaml/qwen.yaml → arpo_train_sft/yaml/qwen_toy.yaml
     - Modifications:
       - model_name_or_path: ./models/Qwen2.5-0.5B-Instruct
       - dataset: arpo_sft_toy
       - output_dir: ./checkpoints/qwen0.5b_toy
       - cutoff_len: 4096 (reduce from 15000 for faster testing)
       - max_samples: 500 (limit to toy dataset size)
       - num_train_epochs: 1.0 (single epoch for quick validation)
       - save_steps: 50 (save more frequently for small dataset)
       - logging_steps: 5 (log frequently to monitor)
       - per_device_train_batch_size: 2 (can use larger batch for 0.5B)
       - gradient_accumulation_steps: 1
       - deepspeed: ../examples/deepspeed/ds_z2_config.json (ZeRO-2 sufficient for 0.5B)

     4.2 Create Toy Training Script
     - Copy arpo_train_sft/sft_train.sh → arpo_train_sft/sft_train_toy.sh
     - Modifications:
       - export CUDA_VISIBLE_DEVICES=0 (single GPU)
       - PROC_PER_NODE=1 (1 GPU)
       - OUTPUT_DIR="./checkpoints/qwen0.5b_toy"
       - TRAIN_ARGS="yaml/qwen_toy.yaml"
       - Remove or comment out multi-node configuration

     Phase 5: Execute Local Toy Training

     5.1 Pre-flight Checks
     - Verify GPU available: nvidia-smi
     - Check disk space: ~10GB needed
     - Verify all paths exist
     - Test data loads: Quick Python check

     5.2 Launch Toy Training
     - cd arpo_train_sft/
     - bash sft_train_toy.sh
     - Expected duration: 10-30 minutes on 500 samples

     5.3 Validate Toy Training
     - Training completes without OOM errors
     - Checkpoints save successfully
     - Loss decreases over training
     - Final model can be loaded for inference

     5.4 Troubleshooting Checklist
     - If OOM: Reduce batch size to 1, reduce cutoff_len to 2048
     - If slow: Disable DeepSpeed for single GPU (finetuning_type: full without deepspeed)
     - If data errors: Check dataset format and dataset_info.json
     - If model errors: Verify model path and trust_remote_code setting

     ---
     STAGE 2: SERVER PRODUCTION TRAINING

     Phase 6: Server Environment Setup

     6.1 Replicate Environment on Server
     - Copy exact conda environment specification
     - Install same package versions as local
     - Verify 8 GPUs visible and accessible
     - Install flash-attention for performance

     6.2 Transfer or Download Data
     - Option A: Download dataset directly on server
     - Option B: Transfer from local machine
     - Ensure full dataset available: final_sft_edition9.json

     Phase 7: Server Model Preparation

     7.1 Download Qwen2.5-3B-Instruct
     - Download on server: Qwen/Qwen2.5-3B-Instruct
     - Save to server path (e.g., /data/models/Qwen2.5-3B-Instruct)
     - ~6GB download

     Phase 8: Production Training Configuration

     8.1 Create Production Training YAML
     - Copy validated toy config → arpo_train_sft/yaml/qwen_3b.yaml
     - Modifications from toy config:
       - model_name_or_path: /data/models/Qwen2.5-3B-Instruct
       - dataset: arpo_sft_54k (full dataset)
       - output_dir: /data/checkpoints/qwen2.5-3b-sft
       - cutoff_len: 15000 (restore to full length)
       - max_samples: 1000000 (use all data)
       - num_train_epochs: 3.0 (full 3 epochs)
       - save_steps: 2000 (less frequent for large training)
       - logging_steps: 10
       - per_device_train_batch_size: 1 (conservative for 3B)
       - gradient_accumulation_steps: 2
       - deepspeed: ../examples/deepspeed/ds_z3_offload_config.json (ZeRO-3 for efficiency)

     8.2 Create Production Training Script
     - Copy toy script → arpo_train_sft/sft_train_3b.sh
     - Modifications:
       - export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 (all 8 GPUs)
       - PROC_PER_NODE=8 (8 GPUs)
       - OUTPUT_DIR="/data/checkpoints/qwen2.5-3b-sft"
       - TRAIN_ARGS="yaml/qwen_3b.yaml"

     Phase 9: Execute Production Training

     9.1 Dry Run Test
     - First test with max_samples: 1000 (1K samples)
     - Verify distributed training works across 8 GPUs
     - Check memory usage per GPU
     - Ensure no OOM errors

     9.2 Launch Full Training
     - Remove sample limit or set to 1000000
     - Launch: bash sft_train_3b.sh
     - Expected duration: 8-16 hours for 54K samples × 3 epochs

     9.3 Monitor Production Training
     - GPU utilization: watch -n 1 nvidia-smi
     - Training logs: tail -f /data/checkpoints/qwen2.5-3b-sft/training.log
     - Checkpoint disk usage
     - Loss curves

     Phase 10: Validation & Handoff to RL

     10.1 Verify Final Checkpoint
     - Training completed all epochs
     - Final checkpoint saved
     - Model files intact and loadable

     10.2 Quick Inference Test
     - Load model with LLaMA-Factory
     - Test on sample prompts
     - Verify output quality

     10.3 Prepare for RL Stage
     - Note final checkpoint path for ARPO/AEPO training
     - Document any deviations from standard config
     - Preserve training logs for reference

     ---
     Key Differences Between Stages

     | Aspect        | Stage 1 (Local Toy) | Stage 2 (Server Production) |
     |---------------|---------------------|-----------------------------|
     | Model         | Qwen2.5-0.5B        | Qwen2.5-3B                  |
     | GPUs          | 1 GPU               | 8 GPUs                      |
     | Dataset       | 100-500 samples     | 54K samples                 |
     | Epochs        | 1 epoch             | 3 epochs                    |
     | Duration      | 10-30 min           | 8-16 hours                  |
     | Batch Size    | 2 per device        | 1 per device                |
     | Cutoff Length | 4096                | 15000                       |
     | DeepSpeed     | ZeRO-2              | ZeRO-3 + offload            |
     | Purpose       | Validate workflow   | Full reproduction           |

     Additional Modifications Needed

     1. Dataset Creation Script
     - Create Python script to extract toy subset:
       - Read final_sft_edition9.json
       - Take first 100-500 entries
       - Save as final_sft_edition9_toy.jsonl
       - Preserve ShareGPT format

     2. Configuration File Management
     - Keep separate configs for toy vs production
     - Document differences clearly
     - Use descriptive naming (qwen_toy.yaml vs qwen_3b.yaml)

     3. Output Directory Organization
     checkpoints/
     ├── qwen0.5b_toy/          # Local toy training
     │   └── checkpoint-50/
     └── qwen2.5-3b-sft/        # Production training
         ├── checkpoint-2000/
         ├── checkpoint-4000/
         └── ...

     4. Disk Space Planning
     - Local: ~10GB (0.5B model + toy data + checkpoints)
     - Server: ~60GB (3B model + full data + checkpoints)

     5. Error Handling Strategy
     - Stage 1 failures: Fix locally before moving to server
     - Stage 2 failures: Reference working toy config
     - Keep toy config as "known good" baseline

     Success Criteria

     Stage 1 Success:
     - ✓ Training completes without errors
     - ✓ Loss decreases consistently
     - ✓ Checkpoints save and load correctly
     - ✓ GPU memory usage stable
     - ✓ All configurations validated

     Stage 2 Success:
     - ✓ All 8 GPUs utilized evenly
     - ✓ Training completes 3 epochs on full dataset
     - ✓ Final checkpoint ready for RL training
     - ✓ Training logs show stable convergence
     - ✓ Model quality validated via inference test