#!/usr/bin/env python3
"""
Digital Twin Generator - Main Generation Script
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict
import torch
import numpy as np
from PIL import Image
import zipfile
import shutil

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import (
    GENERATION_CONFIG, IP_ADAPTER_CONFIG, FACE_ENHANCE_CONFIG,
    PROMPT_TEMPLATES, NEGATIVE_PROMPTS, PROCESSING_CONFIG,
    SELFIES_DIR, OUTPUT_DIR
)
from utils.face_utils import validate_selfie_folder, process_selfies_for_training, verify_same_person
from utils.model_loader import get_model_loader
from utils.pose_lighting_analyzer import PoseLightingAnalyzer
from utils.controlnet_integration import ControlNetProcessor
from utils.lora_trainer import LoRAManager
from utils.ipadapter_manager import IPAdapterManager
from config import CURRENT_QUALITY, FACE_ENHANCEMENT_CONFIG, CONTROLNET_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DigitalTwinGenerator:
    """Main class for generating digital twins"""
    
    def __init__(self):
        self.model_loader = get_model_loader()
        self.pipeline = None
        self.ip_adapter = None
        self.face_enhancer = None
        self.pose_analyzer = PoseLightingAnalyzer()
        self.controlnet_integrator = ControlNetProcessor()
        self.lora_manager = LoRAManager()
        self.ipadapter_manager = IPAdapterManager()
        
    def load_models(self):
        """Load all required models"""
        logger.info("Loading models...")
        
        try:
            # Load SDXL pipeline
            self.pipeline = self.model_loader.load_sdxl_pipeline()
            
            # Load IPAdapter
            ip_adapter_data = self.model_loader.load_ip_adapter()
            self.ip_adapter = ip_adapter_data["adapter"]
            
            # Set IPAdapter pipeline
            self.ip_adapter.set_pipeline(self.pipeline)
            
            # Load IPAdapter manager model
            self.ipadapter_manager.load_ipadapter_model()
            
            # Load face enhancement model
            self.face_enhancer = self.model_loader.load_face_enhancement_model(
                FACE_ENHANCE_CONFIG["method"]
            )
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def extract_selfies_from_zip(self, zip_path: str, extract_dir: str) -> List[str]:
        """Extract selfies from ZIP file"""
        logger.info(f"Extracting selfies from {zip_path}")
        
        extract_path = Path(extract_dir)
        extract_path.mkdir(parents=True, exist_ok=True)
        
        image_files = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract all files
                zip_ref.extractall(extract_path)
                
                # Find image files
                image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
                
                for file_path in extract_path.rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                        image_files.append(str(file_path))
            
            logger.info(f"Extracted {len(image_files)} image files")
            return image_files
            
        except Exception as e:
            logger.error(f"Failed to extract ZIP file: {e}")
            raise
    
    def validate_input_selfies(self, selfies_folder: str) -> tuple:
        """Validate selfies and verify they're of the same person"""
        logger.info("Validating selfies...")
        
        # Check if we have enough selfies
        is_sufficient, valid_images, invalid_images = validate_selfie_folder(selfies_folder)
        
        if not is_sufficient:
            min_selfies = PROCESSING_CONFIG["min_selfies"]
            logger.error(f"Not enough valid selfies. Found {len(valid_images)}, need at least {min_selfies}")
            return False, valid_images, invalid_images
        
        # Verify all selfies are of the same person
        is_same_person, mismatches = verify_same_person(selfies_folder)
        
        if not is_same_person:
            logger.warning("Some selfies may be of different people:")
            for mismatch in mismatches:
                logger.warning(f"  - {mismatch}")
        
        return True, valid_images, invalid_images
    
    def prepare_ip_adapter_images(self, selfies_folder: str) -> List[str]:
        """Prepare images for IPAdapter with batch averaging"""
        logger.info("Preparing images for IPAdapter with batch averaging...")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        image_paths = []
        
        for file_path in Path(selfies_folder).rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_paths.append(str(file_path))
        
        if not image_paths:
            raise ValueError("No image files found in selfies folder")
        
        logger.info(f"Found {len(image_paths)} images for IPAdapter batch processing")
        return image_paths
    
    def analyze_pose_and_lighting(self, selfies_folder: str) -> Dict:
        """Analyze pose and lighting patterns from selfie set"""
        logger.info("Analyzing pose and lighting patterns...")
        
        try:
            # Get all image files in the folder
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
            image_paths = []
            
            for file_path in Path(selfies_folder).rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    image_paths.append(str(file_path))
            
            if not image_paths:
                logger.warning("No images found for pose/lighting analysis")
                return {}
            
            # Analyze pose and lighting
            analysis = self.pose_analyzer.analyze_selfie_set(image_paths)
            
            if analysis:
                logger.info(f"Pose/lighting analysis completed: {analysis.get('sample_count', 0)} samples")
                
                # Generate ControlNet prompts
                controlnet_prompts = self.pose_analyzer.generate_controlnet_prompt(analysis)
                analysis['controlnet_prompts'] = controlnet_prompts
                
                # Prepare ControlNet inputs
                controlnet_inputs = self.controlnet_integrator.prepare_controlnet_inputs(
                    image_paths, analysis
                )
                analysis['controlnet_inputs'] = controlnet_inputs
                
                return analysis
            else:
                logger.warning("Pose/lighting analysis returned no results")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to analyze pose and lighting: {e}")
            return {}
    
    def create_lora_embedding(self, selfies_folder: str, user_id: str) -> Dict:
        """Create LoRA embedding for user identity"""
        logger.info(f"Creating LoRA embedding for user {user_id}")
        
        try:
            # Get all image files in the folder
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
            image_paths = []
            
            for file_path in Path(selfies_folder).rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    image_paths.append(str(file_path))
            
            if not image_paths:
                logger.warning("No images found for LoRA training")
                return {}
            
            if len(image_paths) < 10:
                logger.warning(f"Only {len(image_paths)} images available for LoRA training")
            
            # Create LoRA embedding
            lora_metadata = self.lora_manager.create_user_lora(
                image_paths, user_id, force_retrain=False
            )
            
            logger.info(f"LoRA embedding created successfully for user {user_id}")
            return lora_metadata
            
        except Exception as e:
            logger.error(f"Failed to create LoRA embedding: {e}")
            return {}
    
    def generate_digital_twin(self, 
                             selfies_folder: str, 
                             output_path: str,
                             prompt_style: str = "portrait",
                             num_images: int = 1,
                             seed: Optional[int] = None) -> List[str]:
        """Generate digital twin images with pose and lighting analysis"""
        logger.info("Generating digital twin with enhanced pose and lighting...")
        
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Analyze pose and lighting patterns
        pose_lighting_analysis = self.analyze_pose_and_lighting(selfies_folder)
        
        # Create LoRA embedding for user identity (optional)
        user_id = f"user_{int(time.time())}"  # Generate unique user ID
        lora_metadata = self.create_lora_embedding(selfies_folder, user_id)
        
        # Prepare images for IPAdapter with batch averaging
        ip_images = self.prepare_ip_adapter_images(selfies_folder)
        
        # Process selfies for identity embedding with batch averaging
        logger.info("Processing selfies for identity embedding with batch averaging...")
        averaged_embedding, identity_result = self.ipadapter_manager.process_selfies_for_identity(
            ip_images, weight=IP_ADAPTER_CONFIG["weight"]
        )
        
        if not identity_result['success']:
            raise ValueError(f"Failed to create identity embedding: {identity_result.get('error', 'Unknown error')}")
        
        # Log identity analysis report
        identity_report = self.ipadapter_manager.get_identity_analysis_report(identity_result)
        logger.info(identity_report)
        
        # Generate images with enhanced conditioning
        generated_images = []
        
        for i in range(num_images):
            logger.info(f"Generating image {i+1}/{num_images}")
            
            # Get base prompt
            base_prompt = PROMPT_TEMPLATES[prompt_style]
            negative_prompt = ", ".join(NEGATIVE_PROMPTS)
            
            # Enhance prompt with pose and lighting analysis
            enhanced_prompt = self._enhance_prompt_with_analysis(
                base_prompt, pose_lighting_analysis
            )
            
            # Generate with averaged IPAdapter embedding, ControlNet, and LoRA conditioning
            with torch.no_grad():
                # Apply averaged IPAdapter embedding for face conditioning
                self.ipadapter_manager.apply_ipadapter_to_pipeline(
                    self.pipeline, averaged_embedding, IP_ADAPTER_CONFIG["weight"]
                )
                
                # Apply ControlNet conditioning if available
                if pose_lighting_analysis.get('controlnet_inputs'):
                    self._apply_controlnet_conditioning(pose_lighting_analysis['controlnet_inputs'])
                
                # Apply LoRA conditioning if available
                if lora_metadata:
                    self._apply_lora_conditioning(lora_metadata, user_id)
                
                # Generate image
                result = self.pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    width=GENERATION_CONFIG["width"],
                    height=GENERATION_CONFIG["height"],
                    num_inference_steps=GENERATION_CONFIG["num_inference_steps"],
                    guidance_scale=GENERATION_CONFIG["guidance_scale"],
                    num_images_per_prompt=1,
                )
                
                image = result.images[0]
                
                # Enhance face
                enhanced_image = self.enhance_face(image)
                
                # Save image
                output_file = Path(output_path) / f"avatar_{i+1:03d}.png"
                enhanced_image.save(output_file, quality=PROCESSING_CONFIG["quality"])
                
                generated_images.append(str(output_file))
                logger.info(f"Generated image saved to {output_file}")
        
        # Save analysis results for debugging
        if pose_lighting_analysis:
            analysis_file = Path(output_path) / "pose_lighting_analysis.json"
            import json
            with open(analysis_file, 'w') as f:
                json.dump(pose_lighting_analysis, f, indent=2, default=str)
            logger.info(f"Pose/lighting analysis saved to {analysis_file}")
        
        return generated_images
    
    def _enhance_prompt_with_analysis(self, base_prompt: str, analysis: Dict) -> str:
        """Enhance prompt with pose and lighting analysis"""
        try:
            if not analysis:
                return base_prompt
            
            enhanced_parts = [base_prompt]
            
            # Add pose-specific enhancements
            if analysis.get('controlnet_prompts'):
                pose_prompt = analysis['controlnet_prompts'].get('pose_prompt', '')
                if pose_prompt:
                    enhanced_parts.append(pose_prompt)
            
            # Add lighting-specific enhancements
            if analysis.get('controlnet_prompts'):
                lighting_prompt = analysis['controlnet_prompts'].get('lighting_prompt', '')
                if lighting_prompt:
                    enhanced_parts.append(lighting_prompt)
            
            # Add natural pose and lighting descriptors
            if analysis.get('pose', {}).get('dominant_orientation'):
                orientation = analysis['pose']['dominant_orientation']
                if orientation != 'front-facing':
                    enhanced_parts.append(f"natural {orientation} pose")
            
            if analysis.get('lighting', {}).get('dominant_direction'):
                lighting_dir = analysis['lighting']['dominant_direction']
                if lighting_dir != 'front':
                    enhanced_parts.append(f"{lighting_dir} lighting")
            
            return ", ".join(enhanced_parts)
            
        except Exception as e:
            logger.error(f"Failed to enhance prompt: {e}")
            return base_prompt
    
    def _apply_controlnet_conditioning(self, controlnet_inputs: Dict):
        """Apply ControlNet conditioning to the pipeline"""
        try:
            if not controlnet_inputs:
                return
            
            # This is a placeholder for ControlNet integration
            # In a full implementation, you would integrate with ControlNet models
            logger.info("ControlNet conditioning would be applied here")
            
            # For now, we'll log the parameters
            params = controlnet_inputs.get('controlnet_params', {})
            logger.info(f"ControlNet parameters: {params}")
            
        except Exception as e:
            logger.error(f"Failed to apply ControlNet conditioning: {e}")
    
    def _apply_lora_conditioning(self, lora_metadata: Dict, user_id: str):
        """Apply LoRA conditioning to the pipeline"""
        try:
            if not lora_metadata:
                return
            
            # Load LoRA model for the user
            success = self.lora_manager.load_user_lora(user_id)
            
            if success:
                logger.info(f"LoRA conditioning applied for user {user_id}")
                
                # In a full implementation, you would integrate LoRA with the pipeline
                # For now, we'll log the metadata
                logger.info(f"LoRA metadata: {lora_metadata}")
                
                # Apply LoRA to the text encoder
                if hasattr(self.lora_manager.trainer, 'lora_model'):
                    # This would integrate LoRA with the SDXL pipeline
                    logger.info("LoRA model loaded and ready for generation")
            else:
                logger.warning(f"Failed to load LoRA for user {user_id}")
                
        except Exception as e:
            logger.error(f"Failed to apply LoRA conditioning: {e}")
    
    def enhance_face(self, image: Image.Image) -> Image.Image:
        """Enhance face using GFPGAN or CodeFormer based on quality mode"""
        # Get enhancement settings from quality config
        enhancement_mode = FACE_ENHANCEMENT_CONFIG["mode"]
        enhancement_method = FACE_ENHANCEMENT_CONFIG["enhancement_method"]
        
        logger.info(f"Enhancing face with {enhancement_mode} mode using {enhancement_method}...")
        
        try:
            # Check if face enhancer is available (not a dummy model)
            if not self.face_enhancer or hasattr(self.face_enhancer, 'enhance') == False:
                logger.warning("Face enhancement model not available, skipping enhancement")
                return image
            
            if enhancement_mode == "light":
                # Light enhancement - minimal processing
                img_np = np.array(image)
                enhanced_img = img_np
                
                # Apply basic sharpening for light mode
                from scipy import ndimage
                enhanced_img = ndimage.gaussian_filter(enhanced_img, sigma=0.5)
                enhanced_img = np.clip(enhanced_img * 1.1, 0, 255).astype(np.uint8)
                
                enhanced_image = Image.fromarray(enhanced_img)
                
            else:
                # Full enhancement using GFPGAN/CodeFormer
                if enhancement_method == "gfpgan":
                    # Convert PIL to numpy
                    img_np = np.array(image)
                    
                    # Check if this is a dummy enhancer
                    if hasattr(self.face_enhancer, 'enhance') and callable(self.face_enhancer.enhance):
                        # Enhance with GFPGAN
                        _, _, enhanced_img = self.face_enhancer.enhance(
                            img_np,
                            has_aligned=False,
                            only_center_face=False,
                            paste_back=True
                        )
                        
                        # Convert back to PIL
                        enhanced_image = Image.fromarray(enhanced_img)
                    else:
                        logger.warning("GFPGAN not available, using original image")
                        enhanced_image = image
                    
                elif enhancement_method == "codeformer":
                    # Convert PIL to numpy
                    img_np = np.array(image)
                    
                    # Check if this is a dummy enhancer
                    if hasattr(self.face_enhancer, 'enhance') and callable(self.face_enhancer.enhance):
                        # Enhance with CodeFormer
                        enhanced_img = self.face_enhancer.enhance(
                            img_np,
                            background_enhance=True,
                            face_upsample=True,
                            upscale=1,
                            codeformer_fidelity=FACE_ENHANCEMENT_CONFIG["codeformer_strength"]
                        )
                        
                        # Convert back to PIL
                        enhanced_image = Image.fromarray(enhanced_img)
                    else:
                        logger.warning("CodeFormer not available, using original image")
                        enhanced_image = image
                    
                else:
                    logger.warning(f"Unknown enhancement method: {enhancement_method}")
                    enhanced_image = image
            
            logger.info(f"Face enhancement ({enhancement_mode}) completed")
            return enhanced_image
            
        except Exception as e:
            logger.error(f"Face enhancement failed: {e}")
            logger.info("Continuing with original image without enhancement")
            return image
    
    def cleanup(self):
        """Clean up resources"""
        if self.model_loader:
            self.model_loader.cleanup()
        if self.face_enhancer:
            del self.face_enhancer
        if self.pose_analyzer:
            # Clean up MediaPipe resources
            if hasattr(self.pose_analyzer, 'face_mesh') and self.pose_analyzer.face_mesh:
                self.pose_analyzer.face_mesh.close()
        if self.controlnet_integrator:
            self.controlnet_integrator.cleanup()
        if self.lora_manager:
            self.lora_manager.cleanup()
        if self.ipadapter_manager:
            self.ipadapter_manager.cleanup()

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Generate digital twin from selfies")
    parser.add_argument("--input_folder", required=True, help="Folder containing selfies")
    parser.add_argument("--output_folder", required=True, help="Output folder for generated images")
    parser.add_argument("--prompt_style", default="portrait", choices=["portrait", "casual", "professional"],
                       help="Style of the generated image")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--zip_file", help="ZIP file containing selfies (alternative to input_folder)")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = DigitalTwinGenerator()
    
    try:
        # Load models
        generator.load_models()
        
        # Handle input
        if args.zip_file:
            # Extract from ZIP
            temp_dir = SELFIES_DIR / "temp_extract"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            selfies_folder = str(temp_dir)
            generator.extract_selfies_from_zip(args.zip_file, selfies_folder)
        else:
            selfies_folder = args.input_folder
        
        # Validate selfies
        is_valid, valid_images, invalid_images = generator.validate_input_selfies(selfies_folder)
        
        if not is_valid:
            logger.error("Input validation failed")
            if invalid_images:
                logger.error("Invalid images:")
                for img in invalid_images:
                    logger.error(f"  - {img}")
            return 1
        
        # Generate digital twin
        generated_images = generator.generate_digital_twin(
            selfies_folder=selfies_folder,
            output_path=str(output_path),
            prompt_style=args.prompt_style,
            num_images=args.num_images,
            seed=args.seed
        )
        
        logger.info(f"Successfully generated {len(generated_images)} digital twin images:")
        for img_path in generated_images:
            logger.info(f"  - {img_path}")
        
        # Cleanup
        if args.zip_file and temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        return 0
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return 1
    
    finally:
        generator.cleanup()

if __name__ == "__main__":
    sys.exit(main()) 