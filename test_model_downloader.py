#!/usr/bin/env python3
"""
Test script for model downloader functionality
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from download_models import ModelDownloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model_downloader():
    """Test the model downloader functionality"""
    logger.info("Testing Model Downloader")
    logger.info("=" * 40)
    
    try:
        # Initialize downloader
        downloader = ModelDownloader()
        
        # Test directory creation
        logger.info("Testing directory creation...")
        success = downloader.create_models_directory()
        if success:
            logger.info("✅ Directory creation successful")
        else:
            logger.error("❌ Directory creation failed")
            return False
        
        # Test prerequisite checks
        logger.info("Testing prerequisite checks...")
        git_ok = downloader.check_git_installed()
        wget_ok = downloader.check_wget_installed()
        curl_ok = downloader.check_curl_installed()
        
        logger.info(f"Git: {'✅' if git_ok else '❌'}")
        logger.info(f"Wget: {'✅' if wget_ok else '❌'}")
        logger.info(f"Curl: {'✅' if curl_ok else '❌'}")
        
        if not git_ok:
            logger.error("Git is required for HuggingFace model downloads")
            return False
        
        # Test model configurations
        logger.info("Testing model configurations...")
        for model_key, model_config in downloader.models.items():
            logger.info(f"✅ {model_config['name']}: {model_config['type']} -> {model_config['path']}")
        
        # Test verification
        logger.info("Testing model verification...")
        verification_results = downloader.verify_models()
        
        for model_key, is_verified in verification_results.items():
            status = "✅ Verified" if is_verified else "❌ Missing"
            logger.info(f"{downloader.models[model_key]['name']}: {status}")
        
        logger.info("✅ Model downloader test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model downloader test failed: {e}")
        return False

def show_cloud_deployment_info():
    """Show cloud deployment information"""
    logger.info("\n☁️ Cloud GPU Deployment Information")
    logger.info("=" * 40)
    
    info = [
        "📦 Models to Download:",
        "   • Stable Diffusion XL Base (~6.9GB)",
        "   • IPAdapter FaceID (~1.2GB)",
        "   • GFPGAN Face Enhancement (~340MB)",
        "   • InsightFace Models (~500MB, auto-downloaded)",
        "",
        "💾 Total Storage Required: ~9GB",
        "",
        "🚀 Deployment Steps:",
        "   1. Upload scripts to cloud GPU instance",
        "   2. Run: ./download_models.sh",
        "   3. Models downloaded to /models directory",
        "   4. Use generate_twin.py for avatar generation",
        "",
        "💡 Optimization Tips:",
        "   • Use high-bandwidth connection",
        "   • Ensure 50GB+ available storage",
        "   • Consider HuggingFace authentication",
        "   • Models cached for future use"
    ]
    
    for item in info:
        logger.info(item)

if __name__ == "__main__":
    logger.info("🎯 Model Downloader Test")
    logger.info("=" * 30)
    
    # Run test
    success = test_model_downloader()
    
    if success:
        show_cloud_deployment_info()
        logger.info("\n🎉 All tests passed!")
        logger.info("🚀 Ready for cloud GPU deployment!")
    else:
        logger.error("\n❌ Tests failed!")
        sys.exit(1) 