#!/usr/bin/env python3
"""
Test script for Digital Twin Generator
"""

import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config import *
from utils.face_utils import FaceProcessor
from utils.model_loader import get_model_loader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required packages can be imported"""
    logger.info("Testing imports...")
    
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
        
        import torchvision
        logger.info(f"✓ TorchVision {torchvision.__version__}")
        
        import diffusers
        logger.info(f"✓ Diffusers {diffusers.__version__}")
        
        import transformers
        logger.info(f"✓ Transformers {transformers.__version__}")
        
        import cv2
        logger.info(f"✓ OpenCV {cv2.__version__}")
        
        from PIL import Image
        logger.info("✓ Pillow")
        
        import numpy as np
        logger.info("✓ NumPy")
        
        import flask
        logger.info("✓ Flask")
        
        import insightface
        logger.info("✓ InsightFace")
        
        logger.info("✓ All imports successful!")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_gpu():
    """Test GPU availability"""
    logger.info("Testing GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            logger.info(f"✓ GPU available: {device_name}")
            logger.info(f"✓ GPU count: {device_count}")
            logger.info(f"✓ GPU memory: {memory:.1f} GB")
            return True
        else:
            logger.warning("⚠ No GPU available, will use CPU")
            return False
            
    except Exception as e:
        logger.error(f"✗ GPU test failed: {e}")
        return False

def test_face_processing():
    """Test face processing utilities"""
    logger.info("Testing face processing...")
    
    try:
        processor = FaceProcessor()
        
        if processor.app is not None:
            logger.info("✓ Face processing initialized successfully")
            return True
        else:
            logger.warning("⚠ Face processing not available")
            return False
            
    except Exception as e:
        logger.error(f"✗ Face processing test failed: {e}")
        return False

def test_model_loading():
    """Test model loading (without downloading)"""
    logger.info("Testing model loading...")
    
    try:
        loader = get_model_loader()
        logger.info("✓ Model loader initialized")
        return True
        
    except Exception as e:
        logger.error(f"✗ Model loading test failed: {e}")
        return False

def test_directories():
    """Test directory structure"""
    logger.info("Testing directory structure...")
    
    directories = [
        MODELS_DIR,
        SELFIES_DIR,
        OUTPUT_DIR,
        WEB_DIR,
        IP_ADAPTER_DIR,
        SDXL_DIR,
        FACE_ENHANCE_DIR
    ]
    
    for directory in directories:
        if directory.exists():
            logger.info(f"✓ Directory exists: {directory}")
        else:
            logger.warning(f"⚠ Directory missing: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Created directory: {directory}")
    
    return True

def test_config():
    """Test configuration"""
    logger.info("Testing configuration...")
    
    try:
        # Test basic config values
        assert GENERATION_CONFIG["width"] > 0
        assert GENERATION_CONFIG["height"] > 0
        assert PROCESSING_CONFIG["min_selfies"] > 0
        
        logger.info("✓ Configuration valid")
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting Digital Twin Generator tests...")
    
    tests = [
        ("Imports", test_imports),
        ("GPU", test_gpu),
        ("Face Processing", test_face_processing),
        ("Model Loading", test_model_loading),
        ("Directories", test_directories),
        ("Configuration", test_config)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} test passed")
            else:
                logger.warning(f"⚠ {test_name} test failed")
        except Exception as e:
            logger.error(f"✗ {test_name} test error: {e}")
    
    logger.info(f"\n--- Test Results ---")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed! Digital Twin Generator is ready to use.")
        return 0
    else:
        logger.warning(f"⚠ {total - passed} tests failed. Some features may not work.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 