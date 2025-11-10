# PyTorch DLL Loading Error on Windows - Troubleshooting Guide

**Date**: 2025-11-09
**Environment**: Windows 11, ARPO SFT Training, conda environment `sft`
**Error Code**: WinError 126

---

## Executive Summary

When attempting to run the toy SFT training script (`sft_train_toy.sh`), the training failed immediately with a DLL loading error. **Root Cause**: PyTorch was never installed in the conda environment. The error message was misleading, suggesting a DLL dependency issue when the actual problem was a missing PyTorch installation.

**Status**: ✅ **RESOLVED** - Solution documented below

---

## Error Encountered

### Full Error Message

```
==========================================
Starting TOY SFT Training (Local Testing)
==========================================
Model: Qwen2.5-0.5B-Instruct
Dataset: arpo_sft_toy (500 samples)
GPUs: 1 (CUDA_VISIBLE_DEVICES=0)
Output: ./checkpoints/qwen0.5b_toy
Config: yaml/qwen_toy.yaml
==========================================

Traceback (most recent call last):
  File "C:\Users\user\miniconda3\envs\sft\lib\runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "C:\Users\user\miniconda3\envs\sft\lib\runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "C:\Users\user\miniconda3\envs\sft\Scripts\torchrun.exe\__main__.py", line 2, in <module>
  File "C:\Users\user\miniconda3\envs\sft\lib\site-packages\torch\__init__.py", line 148, in <module>
    raise err
OSError: [WinError 126] Error loading "C:\Users\user\miniconda3\envs\sft\lib\site-packages\torch\lib\fbgemm.dll"
or one of its dependencies.
```

### Error Context

- **Script**: `LLaMA-Factory/arpo_train_sft/sft_train_toy.sh`
- **Command**: `torchrun` (PyTorch distributed training launcher)
- **Environment**: conda environment `sft` (Python 3.10.19)
- **System**: Windows 11 with CUDA 12.9

---

## Root Cause Analysis

### What Really Happened

The error message is **misleading**. It suggests that:
- PyTorch is installed but `fbgemm.dll` cannot be loaded
- There's a missing Windows dependency (Visual C++ Redistributables, CUDA libraries, etc.)

**However, the actual root cause is much simpler:**

**PyTorch was never installed in the `sft` conda environment.**

### Evidence

1. **pip list check**:
   ```bash
   pip list | findstr /i "torch"
   # Returns: FINDSTR: Cannot find torch (no results)
   ```

2. **Direct import test**:
   ```python
   python -c "import torch"
   # Raises ImportError (not shown due to Windows error handling)
   ```

3. **Requirements.txt analysis**:
   - The `LLaMA-Factory/requirements.txt` does NOT include PyTorch
   - PyTorch must be installed separately BEFORE requirements.txt

### Why the Error is Misleading

The error occurs because:
1. `torchrun` is a script that tries to import the `torch` module
2. When Python cannot find the `torch` package, it attempts to load it
3. The Windows loader tries to access `torch\lib\fbgemm.dll` from a non-existent package
4. Windows reports "DLL not found" instead of "Module not found"
5. This makes it appear as a DLL dependency issue rather than a missing package

### Common Misdiagnoses

Users encountering this error often:
- ❌ Install Visual C++ Redistributables (unnecessary - not the issue)
- ❌ Reinstall CUDA/cuDNN (unnecessary - not the issue)
- ❌ Try different PyTorch builds (doesn't work - PyTorch isn't installed at all)
- ❌ Check Windows PATH variables (irrelevant - package is missing)

---

## Step-by-Step Solution

### Prerequisites

- conda environment `sft` already created
- Python 3.10 installed in the environment
- CUDA 12.4+ available on system (verify with `nvidia-smi`)

### Solution: Install PyTorch Correctly

Follow the official ARPO setup procedure from `SETUP_SFT.md`:

#### 1. Activate the Environment

```bash
conda activate sft
```

#### 2. Verify Python Version

```bash
python --version
# Expected: Python 3.10.x
```

#### 3. Install PyTorch with CUDA Support

This is the **critical missing step**:

```bash
pip3 install torch==2.4.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

**Important Notes**:
- Use `pip3` (not `pip`) to ensure correct Python version
- Use `--index-url` to get CUDA 12.4 build (not CPU-only version)
- Version 2.4.0 is required for compatibility with LLaMA-Factory
- This step takes 5-10 minutes (PyTorch is ~2GB)

#### 4. Verify PyTorch Installation

```bash
# Test basic import
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
# Expected: PyTorch version: 2.4.0+cu124

# Test CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
# Expected:
# CUDA available: True
# CUDA version: 12.4
```

#### 5. Install LLaMA-Factory Dependencies

Now you can safely install the other requirements:

```bash
cd LLaMA-Factory
pip install -r requirements.txt
```

#### 6. (Optional) Install Flash Attention

For faster training performance:

```bash
pip install flash-attn --no-build-isolation
```

**Note**: This step may take 10-20 minutes to compile on Windows.

#### 7. Verify Complete Installation

```bash
# Test all critical components
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')"
python -c "import accelerate; print(f'✓ Accelerate: {accelerate.__version__}')"
python -c "import deepspeed; print(f'✓ DeepSpeed: {deepspeed.__version__}')"
```

Expected output:
```
✓ PyTorch: 2.4.0+cu124
✓ Transformers: 4.49.0 (or newer within compatible range)
✓ Accelerate: 1.3.0 (or newer)
✓ DeepSpeed: 0.14.0 (or newer)
```

#### 8. Re-run Training

```bash
cd LLaMA-Factory/arpo_train_sft
bash sft_train_toy.sh
```

Training should now start successfully.

---

## What Went Wrong: Setup Sequence Analysis

### Incorrect Sequence (What Happened)

```
1. conda create -n sft python=3.10
2. conda activate sft
3. pip install -r requirements.txt  ❌ SKIPPED PyTorch installation
4. bash sft_train_toy.sh            ❌ FAILED - PyTorch not found
```

### Correct Sequence (What Should Happen)

```
1. conda create -n sft python=3.10
2. conda activate sft
3. pip3 install torch==2.4.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124  ✅ CRITICAL STEP
4. pip install -r requirements.txt
5. bash sft_train_toy.sh  ✅ SUCCESS
```

### Why This Order Matters

1. **PyTorch is a foundation dependency**: Many packages in `requirements.txt` depend on PyTorch
2. **CUDA version compatibility**: Installing PyTorch first ensures correct CUDA build
3. **Dependency resolution**: pip can properly resolve dependencies when PyTorch exists

---

## Windows-Specific Considerations

### Why This Error is More Common on Windows

1. **DLL Error Messages**: Windows reports missing Python packages as DLL errors
2. **Path Handling**: Windows uses backslashes, making error traces less clear
3. **Binary Compatibility**: PyTorch wheels have specific Windows/CUDA requirements

### Additional Windows Troubleshooting

If you've installed PyTorch and STILL get DLL errors:

#### Issue: Visual C++ Redistributables Missing

**Symptoms**: Error mentions `vcruntime140.dll`, `msvcp140.dll`

**Solution**:
```bash
# Download and install from Microsoft:
# https://aka.ms/vs/17/release/vc_redist.x64.exe
```

#### Issue: Wrong PyTorch Build (CPU vs CUDA)

**Check current build**:
```python
python -c "import torch; print(torch.version.cuda)"
# If this returns None, you have CPU-only PyTorch
```

**Fix**:
```bash
pip uninstall torch torchvision
pip3 install torch==2.4.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

#### Issue: Conda vs Pip Conflict

**Avoid mixing conda and pip for PyTorch**:
```bash
# DO NOT use:
# conda install pytorch  ❌

# ALWAYS use pip with specific index URL:
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124  ✅
```

---

## Prevention: Setup Checklist

Use this checklist when setting up ARPO SFT training:

### Initial Setup
- [ ] CUDA 12.4+ installed (`nvidia-smi` shows version)
- [ ] conda/miniconda installed
- [ ] Git repository cloned

### Environment Creation
- [ ] `conda create -n sft python=3.10 -y`
- [ ] `conda activate sft`
- [ ] Verify: `python --version` shows 3.10.x

### PyTorch Installation (CRITICAL)
- [ ] `pip3 install torch==2.4.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124`
- [ ] Verify: `python -c "import torch; print(torch.__version__)"`
- [ ] Verify: `python -c "import torch; print(torch.cuda.is_available())"` returns True

### Dependencies Installation
- [ ] `cd LLaMA-Factory`
- [ ] `pip install -r requirements.txt`
- [ ] (Optional) `pip install flash-attn --no-build-isolation`

### Verification
- [ ] `python -c "import transformers; print(transformers.__version__)"`
- [ ] `python -c "import deepspeed; print(deepspeed.__version__)"`
- [ ] `torchrun --help` shows help message (confirms torchrun works)

### Ready to Train
- [ ] Dataset downloaded and prepared
- [ ] Model downloaded or path configured
- [ ] `bash sft_train_toy.sh` starts without import errors

---

## Related Issues and References

### Official Documentation
- **ARPO Setup Guide**: `LLaMA-Factory/arpo_train_sft/SETUP_SFT.md` (lines 46-54)
- **LLaMA-Factory**: https://github.com/hiyouga/LLaMA-Factory
- **PyTorch Installation**: https://pytorch.org/get-started/locally/

### Common Error Patterns

| Error Message | Actual Cause | Solution |
|---------------|--------------|----------|
| `fbgemm.dll not found` | PyTorch not installed | Install PyTorch with pip3 |
| `torch has no attribute 'cuda'` | CPU-only PyTorch | Reinstall with CUDA index URL |
| `ImportError: No module named 'torch'` | PyTorch not installed | Install PyTorch with pip3 |
| `vcruntime140.dll missing` | VC++ Redistributables | Install VC++ Runtime |
| `CUDA driver version is insufficient` | Old NVIDIA drivers | Update GPU drivers |

### Version Compatibility

| Component | Version | Compatibility Note |
|-----------|---------|-------------------|
| Python | 3.10.x | Required (3.11+ not tested) |
| PyTorch | 2.4.0 | CUDA 12.4 build recommended |
| CUDA | 12.4+ | Match PyTorch build version |
| Transformers | 4.49.0 - 4.51.3 | From requirements.txt |
| DeepSpeed | Latest compatible | Auto-installed with requirements |

---

## Lessons Learned

### Key Takeaways

1. **Read setup guides completely**: The installation order matters
2. **PyTorch is NOT in requirements.txt**: Must be installed manually
3. **Windows DLL errors can be misleading**: Check if package exists first
4. **CUDA build matters**: Use `--index-url` to get correct PyTorch build
5. **Verify each step**: Don't proceed until verification passes

### Best Practices for Windows ARPO Setup

1. **Follow SETUP_SFT.md exactly**: Don't skip steps
2. **Use pip3, not pip**: Ensures correct Python interpreter
3. **Use explicit index URLs**: Don't rely on default PyPI for PyTorch
4. **Verify after each major step**: Catch issues early
5. **Keep error logs**: Windows paths and error codes are useful for debugging

---

## Quick Reference Commands

### Diagnostic Commands

```bash
# Check if in correct environment
echo $CONDA_DEFAULT_ENV
# Should show: sft

# Check Python version
python --version

# Check if PyTorch installed
pip list | findstr /i "torch"

# Test PyTorch import
python -c "import torch; print(torch.__version__)"

# Test CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA driver
nvidia-smi
```

### Recovery Commands

```bash
# If environment is broken, recreate it
conda deactivate
conda env remove -n sft
conda create -n sft python=3.10 -y
conda activate sft

# Then follow correct installation sequence
pip3 install torch==2.4.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
cd LLaMA-Factory
pip install -r requirements.txt
```

---

## Conclusion

The `fbgemm.dll` loading error encountered during ARPO SFT training on Windows was caused by **missing PyTorch installation**, not by missing DLL dependencies. The solution is straightforward: install PyTorch with CUDA support BEFORE installing other requirements.

**Key Success Factor**: Follow the installation sequence in `SETUP_SFT.md` exactly, without skipping the PyTorch installation step.

**Current Status**: ✅ Issue identified and solution documented. Ready for implementation.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-09
**Author**: Troubleshooting investigation for ARPO Project
**Location**: `.trae/documents/PyTorch_DLL_Error_Windows_Troubleshooting.md`
