#!/bin/bash
# VM Setup Script for Data Synthesizer
# Run this on a fresh GCP/AWS free tier VM

set -e

echo "=========================================="
echo "Data Synthesizer VM Setup"
echo "=========================================="

# Update system
echo "Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Python and essentials
echo "Installing Python and dependencies..."
sudo apt-get install -y python3 python3-pip python3-venv git screen

# Clone repository (if not already present)
if [ ! -d "data-synthesizer" ]; then
    echo "Cloning repository..."
    git clone https://github.com/username/data-synthesizer.git
    cd data-synthesizer
else
    echo "Repository already exists"
    cd data-synthesizer
    git pull
fi

# Create virtual environment
echo "Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs output

# Copy environment template
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "=========================================="
    echo "IMPORTANT: Edit .env file with your API keys"
    echo "Run: nano .env"
    echo "=========================================="
fi

# Make scripts executable
chmod +x scripts/*.py
chmod +x scripts/*.sh

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys:"
echo "   nano .env"
echo ""
echo "2. Edit config.yaml for your use case:"
echo "   nano config.yaml"
echo ""
echo "3. Run with screen (persists after disconnect):"
echo "   screen -S synth"
echo "   source venv/bin/activate"
echo "   python scripts/run.py"
echo "   # Press Ctrl+A, D to detach"
echo ""
echo "4. Reattach to screen:"
echo "   screen -r synth"
echo ""
