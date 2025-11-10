#!/usr/bin/env python
import torch
import transformers
import accelerate
import peft
import trl

print('='*60)
print('ARPO SFT Environment Verification')
print('='*60)
print('')
print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
print('')
print(f'Transformers: {transformers.__version__}')
print(f'Accelerate: {accelerate.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'TRL: {trl.__version__}')
print('')
print('='*60)
print('All critical packages installed successfully!')
print('='*60)
