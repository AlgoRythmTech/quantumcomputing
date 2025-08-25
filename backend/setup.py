"""
Setup and Installation Script for Rythm AI 1.2 Europa
Complete setup for the production-ready 8B parameter model
"""

import os
import sys
import subprocess
import platform
import torch
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_python_version():
    """Check Python version compatibility"""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        logger.error(f"Python {required_version[0]}.{required_version[1]} or higher is required")
        logger.error(f"Current version: Python {current_version[0]}.{current_version[1]}")
        sys.exit(1)
    
    logger.info(f"Python version {current_version[0]}.{current_version[1]} - OK")


def check_cuda():
    """Check CUDA availability and version"""
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        gpu_count = torch.cuda.device_count()
        
        logger.info(f"CUDA {cuda_version} available")
        logger.info(f"Found {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        return True
    else:
        logger.warning("CUDA not available - training will be slow on CPU")
        return False


def install_dependencies():
    """Install required dependencies"""
    logger.info("Installing dependencies...")
    
    # Upgrade pip
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install PyTorch with CUDA support if available
    if torch.cuda.is_available():
        # Install PyTorch with CUDA 11.8 or 12.1
        logger.info("Installing PyTorch with CUDA support...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ])
    else:
        # Install CPU-only PyTorch
        logger.info("Installing PyTorch (CPU-only)...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio"
        ])
    
    # Install other requirements
    if Path("requirements.txt").exists():
        logger.info("Installing requirements from requirements.txt...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
    else:
        logger.warning("requirements.txt not found, installing essential packages...")
        essential_packages = [
            "transformers",
            "accelerate",
            "sentencepiece",
            "tiktoken",
            "datasets",
            "peft",
            "bitsandbytes",
            "wandb",
            "fastapi",
            "uvicorn",
            "tqdm",
            "numpy",
            "pandas"
        ]
        for package in essential_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except:
                logger.warning(f"Failed to install {package}")


def setup_directories():
    """Create necessary directories"""
    logger.info("Setting up directory structure...")
    
    directories = [
        "checkpoints",
        "data",
        "logs",
        "tokenizer",
        "uploads",
        "financial_datasets",
        "cache",
        "outputs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"  Created {directory}/")


def download_financial_data():
    """Download sample financial datasets"""
    logger.info("Preparing financial datasets...")
    
    # Create sample financial data
    sample_data = {
        "tax_codes": [
            "Section 80C allows deductions up to Rs 1,50,000",
            "Section 80D provides health insurance deductions",
            "Section 24 allows home loan interest deduction",
            "Capital gains tax rates vary by holding period",
            "GST rates: 0%, 5%, 12%, 18%, 28%"
        ],
        "investment_terms": [
            "P/E ratio measures stock valuation",
            "Diversification reduces portfolio risk",
            "Compound interest formula: A = P(1+r)^n",
            "SIP averages out market volatility",
            "Asset allocation depends on risk tolerance"
        ],
        "accounting_principles": [
            "Assets = Liabilities + Equity",
            "Revenue recognition principle",
            "Matching principle for expenses",
            "Depreciation methods: straight-line, declining balance",
            "Cash flow: operating, investing, financing"
        ]
    }
    
    # Save sample data
    data_path = Path("financial_datasets/sample_data.json")
    with open(data_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    logger.info(f"  Sample data saved to {data_path}")


def test_model_creation():
    """Test if model can be created successfully"""
    logger.info("Testing model creation...")
    
    try:
        from rythm_model_architecture import create_rythm_europa_8b
        
        # Create model
        model, config = create_rythm_europa_8b()
        
        # Get parameter count
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model created successfully with {param_count:,} parameters")
        
        # Test forward pass
        dummy_input = torch.randint(0, 1000, (1, 32))
        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = dummy_input.cuda()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        logger.info("  Forward pass successful")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        return False


def test_tokenizer():
    """Test tokenizer functionality"""
    logger.info("Testing tokenizer...")
    
    try:
        from tokenizer_system import create_tokenizer
        
        # Create tokenizer
        tokenizer = create_tokenizer()
        
        # Test encoding/decoding
        test_text = "Calculate tax deduction for FY 2024-25"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        logger.info(f"  Tokenizer test successful")
        logger.info(f"    Original: {test_text}")
        logger.info(f"    Decoded:  {decoded}")
        
        return True
        
    except Exception as e:
        logger.error(f"Tokenizer test failed: {e}")
        return False


def create_config_file():
    """Create configuration file"""
    logger.info("Creating configuration file...")
    
    config = {
        "model": {
            "name": "rythm-europa-8b",
            "parameters": "8B",
            "hidden_size": 5120,
            "num_layers": 48,
            "num_heads": 40,
            "vocab_size": 128000,
            "max_length": 32768
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 5e-5,
            "num_epochs": 3,
            "use_lora": True,
            "use_mixed_precision": True
        },
        "paths": {
            "checkpoints": "./checkpoints",
            "data": "./data",
            "logs": "./logs",
            "tokenizer": "./tokenizer"
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1
        }
    }
    
    config_path = Path("config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"  Configuration saved to {config_path}")


def print_system_info():
    """Print system information"""
    logger.info("\n" + "="*60)
    logger.info("SYSTEM INFORMATION")
    logger.info("="*60)
    
    # Platform info
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info(f"Python: {sys.version}")
    
    # PyTorch info
    logger.info(f"PyTorch: {torch.__version__}")
    
    # CUDA info
    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"cuDNN: {torch.backends.cudnn.version()}")
    
    # Memory info
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"RAM: {memory.total / 1024**3:.1f} GB")
        logger.info(f"Available: {memory.available / 1024**3:.1f} GB")
    except:
        pass
    
    logger.info("="*60 + "\n")


def main():
    """Main setup function"""
    print("="*80)
    print("RYTHM AI 1.2 EUROPA - SETUP")
    print("Production-Ready 8B Parameter Financial Expert Model")
    print("="*80)
    print()
    
    # Check Python version
    check_python_version()
    
    # Check CUDA
    has_cuda = check_cuda()
    
    # Install dependencies
    response = input("\nInstall/update dependencies? (y/n): ")
    if response.lower() == 'y':
        install_dependencies()
    
    # Setup directories
    setup_directories()
    
    # Download sample data
    download_financial_data()
    
    # Create config file
    create_config_file()
    
    # Test model creation
    test_model = test_model_creation()
    
    # Test tokenizer
    test_token = test_tokenizer()
    
    # Print system info
    print_system_info()
    
    # Print summary
    print("\n" + "="*80)
    print("SETUP COMPLETE")
    print("="*80)
    
    if test_model and test_token:
        print("✓ All systems operational")
        print("\nTo start training:")
        print("  python train_rythm_model.py")
        print("\nTo start the API server:")
        print("  python photonai_backend.py")
        print("\nTo use advanced training features:")
        print("  python advanced_training_system.py")
    else:
        print("⚠ Some tests failed - please check the logs above")
    
    if not has_cuda:
        print("\n⚠ WARNING: No GPU detected - training will be very slow")
        print("  For optimal performance, use a machine with NVIDIA GPUs")
    
    print("\nDocumentation:")
    print("  - Model architecture: rythm_model_architecture.py")
    print("  - Tokenizer: tokenizer_system.py")
    print("  - Training: train_rythm_model.py")
    print("  - Advanced training: advanced_training_system.py")
    print("  - API backend: photonai_backend.py")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
