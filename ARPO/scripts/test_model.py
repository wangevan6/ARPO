#!/usr/bin/env python3
"""
Test RL-Trained Model

This script tests a trained ARPO/AEPO model to verify it generates
tool-augmented responses correctly.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_model(model_path, model_name="Model"):
    """
    Test a trained model with sample prompts.

    Args:
        model_path: Path to HuggingFace checkpoint
        model_name: Name for display
    """
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}\n")

    # Load model
    print(f"Loading model from: {model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        print(f"✅ Model loaded successfully!")
        print(f"   Model type: {model.config.model_type}")
        print(f"   Vocab size: {len(tokenizer)}")
        print(f"   Device: {model.device}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Test prompts
    test_prompts = [
        {
            "prompt": "What is the derivative of x^3 + 2x^2 - 5x + 3?",
            "expected_tools": ["python"],
            "description": "Math calculation (should use Python)"
        },
        {
            "prompt": "Calculate 12345 * 67890",
            "expected_tools": ["python"],
            "description": "Simple arithmetic (should use Python)"
        },
        {
            "prompt": "What is the capital of France? Search online if you're not sure.",
            "expected_tools": ["search"],
            "description": "Knowledge question (may use Search)"
        },
        {
            "prompt": "Solve for x: 2x + 5 = 17",
            "expected_tools": ["python"],
            "description": "Algebra problem (should use Python)"
        }
    ]

    print(f"\n{'='*60}")
    print(f"Running Test Cases")
    print(f"{'='*60}\n")

    for i, test_case in enumerate(test_prompts, 1):
        prompt = test_case["prompt"]
        expected_tools = test_case["expected_tools"]
        description = test_case["description"]

        print(f"\n{'─'*60}")
        print(f"Test {i}/{len(test_prompts)}: {description}")
        print(f"{'─'*60}")
        print(f"\n📝 Prompt:")
        print(f"   {prompt}\n")

        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Generate response
        print(f"🤖 Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
                top_p=1.0
            )

        # Decode response
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=False
        )

        print(f"\n💬 Response:")
        print(f"{'─'*60}")
        print(response)
        print(f"{'─'*60}")

        # Check for tool usage
        has_think = "<think>" in response
        has_python = "<python>" in response
        has_search = "<search>" in response
        has_answer = "<answer>" in response

        print(f"\n📊 Analysis:")
        print(f"   Reasoning: {'✅' if has_think else '❌'} <think> tag")
        print(f"   Python:    {'✅' if has_python else '❌'} <python> tag")
        print(f"   Search:    {'✅' if has_search else '❌'} <search> tag")
        print(f"   Answer:    {'✅' if has_answer else '❌'} <answer> tag")

        # Check expected tools
        print(f"\n🎯 Expected Tools: {', '.join(expected_tools)}")
        for tool in expected_tools:
            if tool == "python" and has_python:
                print(f"   ✅ Python tool used as expected")
            elif tool == "search" and has_search:
                print(f"   ✅ Search tool used as expected")
            elif tool == "python" and not has_python:
                print(f"   ⚠️  Python tool not used (expected)")
            elif tool == "search" and not has_search:
                print(f"   ⚠️  Search tool not used (expected)")

    print(f"\n{'='*60}")
    print(f"✅ Testing Complete!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test RL-trained model"
    )
    parser.add_argument(
        '--model_path',
        required=True,
        help='Path to HuggingFace checkpoint'
    )
    parser.add_argument(
        '--model_name',
        default='Model',
        help='Name for display (default: Model)'
    )

    args = parser.parse_args()

    test_model(
        model_path=args.model_path,
        model_name=args.model_name
    )


if __name__ == "__main__":
    main()
