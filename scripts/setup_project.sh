#!/bin/bash

# Project Setup Script for Fraud Detection System

set -e

echo "=========================================="
echo "Setting up Fraud Detection System"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Install development dependencies
echo -e "${YELLOW}Installing development dependencies...${NC}"
pip install black flake8 mypy pre-commit pytest pytest-cov

# Install package in development mode
echo -e "${YELLOW}Installing package in development mode...${NC}"
pip install -e .

# Setup pre-commit hooks
echo -e "${YELLOW}Setting up pre-commit hooks...${NC}"
pre-commit install

# Create .env file from example
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cp .env.example .env
    echo -e "${GREEN}Please update .env file with your API keys${NC}"
fi

# Create necessary directories
echo -e "${YELLOW}Creating project directories...${NC}"
mkdir -p data/raw data/processed data/policies
mkdir -p models logs
mkdir -p notebooks
mkdir -p tests/unit tests/integration tests/load

# Initialize MLflow
echo -e "${YELLOW}Initializing MLflow...${NC}"
mkdir -p mlruns mlartifacts

# Download sample data
echo -e "${YELLOW}Downloading datasets...${NC}"
python scripts/download_data.py

echo -e "${GREEN}=========================================="
echo "Setup completed successfully!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Update .env file with your API keys"
echo "3. Run the API: uvicorn src.api.main:app --reload"
echo "4. Visit http://localhost:8000/docs for API documentation"
echo ""
