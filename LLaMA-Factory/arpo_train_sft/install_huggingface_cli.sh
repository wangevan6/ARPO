#!/bin/bash

# Script to install huggingface-cli and verify installation

echo "Installing huggingface_hub package..."
pip install -U huggingface_hub

echo ""
echo "Verifying installation..."
which huggingface-cli

echo ""
echo "Testing huggingface-cli..."
huggingface-cli --version

echo ""
echo "✓ Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Login to HuggingFace: huggingface-cli login"
echo "  2. Download dataset: huggingface-cli download dongguanting/ARPO-SFT-54K"
echo "  3. Download model: huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct"
