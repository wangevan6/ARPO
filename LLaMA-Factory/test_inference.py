#!/usr/bin/env python
# Test inference on trained toy model

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print('Loading model...')
model_path = 'arpo_train_sft/checkpoints/qwen0.5b_toy'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

print(f'Model loaded: {model_path}')
print(f'Device: {model.device}')
print(f'Dtype: {model.dtype}')
print()

# Test prompts
test_prompts = [
    "What is 2+2?",
    "Hello, how are you?",
    "Explain Python in one sentence."
]

for prompt in test_prompts:
    print(f'Prompt: {prompt}')
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f'Response: {response}')
    print('-' * 80)
    print()

print('Inference test completed successfully!')
