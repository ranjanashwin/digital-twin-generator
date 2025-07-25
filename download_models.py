#!/usr/bin/env python3
"""
Model Download Script for Digital Twin Generator
Downloads all required models for cloud GPU deployment
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import logging
import time
import requests
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelDownloader:
    """Downloads and manages model files for the digital twin generator"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.current_dir = Path.cwd()
        
        # Model configurations
        self.models = {
            "sdxl": {
                "name": "Stable Diffusion XL Base",
                "type": "huggingface",
                "repo": "stabilityai/stable-diffusion-xl-base-1.0",
                "path": self.models_dir / "sdxl",
                "description": "Base SDXL model for image generation"
            },
            "ip_adapter_faceid": {
                "name": "IPAdapter FaceID",
                "type": "huggingface",
                "repo": "h94/IP-Adapter",
                "path": self.models_dir / "ip_adapter_faceid",
                "description": "IPAdapter for face identity preservation"
            },
            "gfpgan": {
                "name": "GFPGAN Face Enhancement",
                "type": "direct",
                "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                "path": self.models_dir / "gfpgan",
                "filename": "GFPGANv1.4.pth",
                "description": "GFPGAN for face restoration and enhancement"
            },
            "insightface": {
                "name": "InsightFace Models",
                "type": "insightface",
                "path": self.models_dir / "insightface",
                "description": "InsightFace for face detection and analysis"
            }
        }
    
    def create_models_directory(self):
        """Create models directory structure"""
        logger.info("Creating models directory structure...")
        
        try:
            # Create main models directory
            self.models_dir.mkdir(exist_ok=True)
            
            # Create subdirectories for each model
            for model_key, model_config in self.models.items():
                model_config["path"].mkdir(exist_ok=True)
                logger.info(f"✓ Created directory: {model_config['path']}")
            
            logger.info("✅ Models directory structure created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create models directory: {e}")
            return False
    
    def check_git_installed(self):
        """Check if git is installed"""
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def check_wget_installed(self):
        """Check if wget is installed"""
        try:
            subprocess.run(["wget", "--version"], check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def check_curl_installed(self):
        """Check if curl is installed"""
        try:
            subprocess.run(["curl", "--version"], check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def download_with_progress(self, url: str, filepath: Path, description: str):
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as file, tqdm(
                desc=description,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    pbar.update(size)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download {url}: {e}")
            return False
    
    def download_huggingface_model(self, repo: str, local_path: Path, model_name: str):
        """Download HuggingFace model using git-lfs"""
        try:
            logger.info(f"📥 Downloading {model_name} from {repo}...")
            
            # Check if directory already exists
            if local_path.exists() and any(local_path.iterdir()):
                logger.info(f"✓ {model_name} already exists at {local_path}")
                return True
            
            # Clone repository
            cmd = ["git", "clone", f"https://huggingface.co/{repo}", str(local_path)]
            logger.info(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ {model_name} downloaded successfully")
                return True
            else:
                logger.error(f"❌ Failed to download {model_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to download {model_name}: {e}")
            return False
    
    def download_direct_file(self, url: str, filepath: Path, model_name: str):
        """Download file directly using wget or curl"""
        try:
            logger.info(f"📥 Downloading {model_name} from {url}...")
            
            # Check if file already exists
            if filepath.exists():
                logger.info(f"✓ {model_name} already exists at {filepath}")
                return True
            
            # Try wget first, then curl
            if self.check_wget_installed():
                cmd = ["wget", "-O", str(filepath), url]
                logger.info(f"Using wget: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
            elif self.check_curl_installed():
                cmd = ["curl", "-L", "-o", str(filepath), url]
                logger.info(f"Using curl: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
            else:
                # Fallback to Python requests
                logger.info("Using Python requests for download...")
                return self.download_with_progress(url, filepath, model_name)
            
            if result.returncode == 0:
                logger.info(f"✅ {model_name} downloaded successfully")
                return True
            else:
                logger.error(f"❌ Failed to download {model_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to download {model_name}: {e}")
            return False
    
    def setup_insightface_models(self, model_path: Path):
        """Setup InsightFace models (they auto-download on first use)"""
        try:
            logger.info("🔧 Setting up InsightFace models...")
            
            # Create directory
            model_path.mkdir(exist_ok=True)
            
            # InsightFace models are downloaded automatically on first use
            # We just need to ensure the directory exists
            logger.info("✅ InsightFace models will be downloaded automatically on first use")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup InsightFace models: {e}")
            return False
    
    def download_all_models(self):
        """Download all required models"""
        logger.info("🚀 Starting model download process...")
        logger.info("💡 Note: This script is optimized for cloud GPU deployment")
        logger.info("💡 Models will be downloaded to the /models directory")
        
        # Check prerequisites
        if not self.check_git_installed():
            logger.error("❌ Git is not installed. Please install git first.")
            return False
        
        # Create directory structure
        if not self.create_models_directory():
            return False
        
        # Download each model
        success_count = 0
        total_models = len(self.models)
        
        for model_key, model_config in self.models.items():
            logger.info(f"\n📦 Processing {model_config['name']}...")
            
            success = False
            
            if model_config["type"] == "huggingface":
                success = self.download_huggingface_model(
                    model_config["repo"],
                    model_config["path"],
                    model_config["name"]
                )
            
            elif model_config["type"] == "direct":
                filepath = model_config["path"] / model_config["filename"]
                success = self.download_direct_file(
                    model_config["url"],
                    filepath,
                    model_config["name"]
                )
            
            elif model_config["type"] == "insightface":
                success = self.setup_insightface_models(model_config["path"])
            
            if success:
                success_count += 1
                logger.info(f"✅ {model_config['name']} is ready!")
            else:
                logger.error(f"❌ Failed to download {model_config['name']}")
        
        # Summary
        logger.info(f"\n📊 Download Summary:")
        logger.info(f"   • Total models: {total_models}")
        logger.info(f"   • Successful: {success_count}")
        logger.info(f"   • Failed: {total_models - success_count}")
        
        if success_count == total_models:
            logger.info("🎉 All models downloaded successfully!")
            logger.info("🚀 Ready for cloud GPU deployment!")
            return True
        else:
            logger.warning("⚠️ Some models failed to download. Check logs above.")
            return False
    
    def verify_models(self):
        """Verify that all models are properly downloaded"""
        logger.info("🔍 Verifying downloaded models...")
        
        verification_results = {}
        
        for model_key, model_config in self.models.items():
            path = model_config["path"]
            
            if path.exists() and any(path.iterdir()):
                verification_results[model_key] = True
                logger.info(f"✅ {model_config['name']}: Verified")
            else:
                verification_results[model_key] = False
                logger.error(f"❌ {model_config['name']}: Missing or empty")
        
        return verification_results
    
    def print_usage_instructions(self):
        """Print usage instructions for cloud GPU deployment"""
        logger.info("\n📋 Cloud GPU Deployment Instructions:")
        logger.info("=" * 50)
        logger.info("1. Upload this script to your cloud GPU instance")
        logger.info("2. Run: python download_models.py")
        logger.info("3. Models will be downloaded to /models directory")
        logger.info("4. Use generate_twin.py with the downloaded models")
        logger.info("")
        logger.info("💡 Tips for cloud deployment:")
        logger.info("   • Ensure sufficient storage space (50GB+ recommended)")
        logger.info("   • Use high-bandwidth connection for faster downloads")
        logger.info("   • Consider using HuggingFace CLI for authentication")
        logger.info("   • Models are cached for future use")

def main():
    """Main function"""
    logger.info("🎯 Digital Twin Generator - Model Downloader")
    logger.info("=" * 60)
    logger.info("💡 Optimized for cloud GPU deployment")
    logger.info("💡 Models will be downloaded to /models directory")
    logger.info("")
    
    # Check if running on cloud GPU
    is_cloud = os.getenv('CUDA_VISIBLE_DEVICES') is not None or os.getenv('GPU') is not None
    if is_cloud:
        logger.info("☁️ Detected cloud GPU environment")
    else:
        logger.info("💻 Running on local machine")
    
    # Initialize downloader
    downloader = ModelDownloader()
    
    try:
        # Download all models
        success = downloader.download_all_models()
        
        if success:
            # Verify models
            verification_results = downloader.verify_models()
            
            # Print final instructions
            downloader.print_usage_instructions()
            
            logger.info("\n🎉 Model download process completed!")
            logger.info("🚀 Ready to generate digital twins!")
            
            return 0
        else:
            logger.error("❌ Model download failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Download interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 