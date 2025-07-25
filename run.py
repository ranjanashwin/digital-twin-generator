#!/usr/bin/env python3
"""
Digital Twin Generator - Run Script
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if all requirements are met"""
    logger.info("Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ required")
        return False
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        logger.error("requirements.txt not found. Please run from the project root.")
        return False
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    logger.info("Installing dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        logger.info("✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def download_models():
    """Download required models"""
    logger.info("Checking models...")
    
    try:
        result = subprocess.run([
            sys.executable, "download_models.py", "--check-only"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✓ Models already downloaded")
            return True
        else:
            logger.info("Downloading models...")
            subprocess.run([
                sys.executable, "download_models.py"
            ], check=True)
            logger.info("✓ Models downloaded")
            return True
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download models: {e}")
        return False

def run_tests():
    """Run installation tests"""
    logger.info("Running tests...")
    
    try:
        result = subprocess.run([
            sys.executable, "test_installation.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✓ All tests passed")
            return True
        else:
            logger.warning("Some tests failed, but continuing...")
            return True
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run tests: {e}")
        return False

def start_web_app():
    """Start the web application"""
    logger.info("Starting web application...")
    
    try:
        subprocess.run([
            sys.executable, "app.py"
        ], check=True)
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start application: {e}")

def main():
    """Main function"""
    logger.info("🤖 Digital Twin Generator")
    logger.info("=" * 40)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Install dependencies if needed
    try:
        import torch
        import flask
        logger.info("✓ Dependencies already installed")
    except ImportError:
        if not install_dependencies():
            sys.exit(1)
    
    # Download models
    if not download_models():
        logger.error("Failed to download models")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        logger.warning("Tests failed, but continuing...")
    
    # Start application
    logger.info("Starting Digital Twin Generator...")
    logger.info("Open http://localhost:5000 in your browser")
    logger.info("Press Ctrl+C to stop")
    
    start_web_app()

if __name__ == "__main__":
    main() 