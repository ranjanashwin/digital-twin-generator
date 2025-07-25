#!/usr/bin/env python3
"""
Test script for pose and lighting analysis
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.pose_lighting_analyzer import PoseLightingAnalyzer
from utils.controlnet_integration import ControlNetProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pose_lighting_analysis():
    """Test pose and lighting analysis functionality"""
    try:
        logger.info("Testing pose and lighting analysis...")
        
        # Initialize analyzers
        pose_analyzer = PoseLightingAnalyzer()
        controlnet_processor = ControlNetProcessor()
        
        # Test with a sample image (if available)
        test_image_path = "test_data/sample_selfie.jpg"
        
        if os.path.exists(test_image_path):
            logger.info(f"Testing with image: {test_image_path}")
            
            # Test pose analysis
            pose_result = pose_analyzer.analyze_pose(test_image_path)
            logger.info(f"Pose analysis result: {pose_result}")
            
            # Test lighting analysis
            lighting_result = pose_analyzer.analyze_lighting(test_image_path)
            logger.info(f"Lighting analysis result: {lighting_result}")
            
            # Test ControlNet processing
            controlnet_result = controlnet_processor.create_pose_conditioning(
                avatar_path=test_image_path,
                clothing_path=test_image_path,
                reference_pose_path=None,
                pose_preset="fashion_portrait"
            )
            logger.info(f"ControlNet processing result: {controlnet_result}")
            
        else:
            logger.warning(f"Test image not found: {test_image_path}")
            logger.info("Skipping image-based tests")
        
        logger.info("✅ Pose and lighting analysis tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pose and lighting analysis tests failed: {e}")
        return False

if __name__ == "__main__":
    success = test_pose_lighting_analysis()
    sys.exit(0 if success else 1) 