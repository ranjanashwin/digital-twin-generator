"""
Model loading utilities for Digital Twin Generator
"""

import os
import torch
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from transformers import CLIPVisionModelWithProjection
import logging
from typing import Optional, Dict, Any
import requests
from tqdm import tqdm

from config import MODEL_CONFIGS, GPU_CONFIG, MODELS_DIR

logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles loading and caching of all required models"""
    
    def __init__(self):
        self.device = torch.device(GPU_CONFIG["device"])
        self.models = {}
        self._setup_directories()
    
    def _setup_directories(self):
        """Create model directories"""
        for model_type in ["ip_adapter", "sdxl", "face_enhance"]:
            model_dir = MODELS_DIR / model_type
            model_dir.mkdir(parents=True, exist_ok=True)
    
    def load_sdxl_pipeline(self, use_refiner: bool = True) -> StableDiffusionXLPipeline:
        """Load Stable Diffusion XL pipeline"""
        if "sdxl_pipeline" in self.models:
            return self.models["sdxl_pipeline"]
        
        logger.info("Loading Stable Diffusion XL pipeline...")
        
        try:
            # Load base model
            model_id = MODEL_CONFIGS["sdxl"]["base"]
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if GPU_CONFIG["precision"] == "fp16" else torch.float32,
                use_safetensors=True,
                variant="fp16" if GPU_CONFIG["precision"] == "fp16" else None,
            )
            
            # Load refiner if requested
            if use_refiner:
                try:
                    refiner_id = MODEL_CONFIGS["sdxl"]["refiner"]
                    refiner = StableDiffusionXLPipeline.from_pretrained(
                        refiner_id,
                        torch_dtype=torch.float16 if GPU_CONFIG["precision"] == "fp16" else torch.float32,
                        use_safetensors=True,
                        variant="fp16" if GPU_CONFIG["precision"] == "fp16" else None,
                    )
                    
                    # Apply same optimizations to refiner
                    if GPU_CONFIG["attention_slicing"]:
                        refiner.enable_attention_slicing()
                    
                    # Move refiner to device
                    refiner = refiner.to(self.device)
                    
                    pipeline.refiner = refiner
                    logger.info("SDXL refiner loaded successfully")
                    
                except Exception as e:
                    logger.warning(f"Failed to load refiner, continuing without it: {e}")
                    # Continue without refiner if it fails to load
            
            # Optimize for memory
            if GPU_CONFIG["attention_slicing"]:
                pipeline.enable_attention_slicing()
            
            # Note: enable_gradient_checkpointing() is not available in current diffusers version
            # if GPU_CONFIG["gradient_checkpointing"]:
            #     pipeline.enable_gradient_checkpointing()
            
            # Move to device
            pipeline = pipeline.to(self.device)
            
            # Use DPM++ scheduler for better quality
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            
            self.models["sdxl_pipeline"] = pipeline
            logger.info("SDXL pipeline loaded successfully")
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to load SDXL pipeline: {e}")
            raise
    
    def load_ip_adapter(self, variant: str = "faceid_plus_v2") -> Dict[str, Any]:
        """Load IPAdapter for face identity preservation"""
        if f"ip_adapter_{variant}" in self.models:
            return self.models[f"ip_adapter_{variant}"]
        
        logger.info(f"Loading IPAdapter {variant}...")
        
        try:
            from utils.ip_adapter_wrapper import create_ip_adapter
            
            # Create IPAdapter wrapper
            ip_adapter = create_ip_adapter(
                pipeline=None,  # Will be set later
                device=self.device,
                implementation="simple"
            )
            
            self.models[f"ip_adapter_{variant}"] = {
                "adapter": ip_adapter,
                "model_id": MODEL_CONFIGS["ip_adapter"][variant]
            }
            
            logger.info(f"IPAdapter {variant} loaded successfully")
            return self.models[f"ip_adapter_{variant}"]
            
        except Exception as e:
            logger.error(f"Failed to load IPAdapter {variant}: {e}")
            raise
    
    def load_face_enhancement_model(self, method: str = "gfpgan") -> Any:
        """Load face enhancement model (GFPGAN or CodeFormer)"""
        if f"face_enhance_{method}" in self.models:
            return self.models[f"face_enhance_{method}"]
        
        logger.info(f"Loading face enhancement model: {method}")
        
        try:
            if method == "gfpgan":
                from gfpgan import GFPGANer
                
                model_path = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
                model = GFPGANer(
                    model_path=model_path,
                    upscale=2,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None
                )
                
            elif method == "codeformer":
                from basicsr.utils.download_util import load_file_from_url
                from facexlib.utils.face_restoration_helper import FaceRestoreHelper
                from codeformer import CodeFormer
                
                model_path = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/CodeFormer.pth"
                model = CodeFormer(
                    dim_embd=512,
                    codebook_size=1024,
                    n_head=8,
                    n_layers=9,
                    ch_mult=[1, 2, 2, 4, 4, 8],
                    resolution=256
                )
                
                # Load weights
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['params_ema'])
                model.eval()
                model = model.to(self.device)
                
            else:
                raise ValueError(f"Unknown face enhancement method: {method}")
            
            self.models[f"face_enhance_{method}"] = model
            logger.info(f"Face enhancement model {method} loaded successfully")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load face enhancement model {method}: {e}")
            raise
    
    def load_clip_vision_model(self) -> CLIPVisionModelWithProjection:
        """Load CLIP vision model for IPAdapter"""
        if "clip_vision" in self.models:
            return self.models["clip_vision"]
        
        logger.info("Loading CLIP vision model...")
        
        try:
            model = CLIPVisionModelWithProjection.from_pretrained(
                "h94/IP-Adapter",
                subfolder="models/image_encoder",
                torch_dtype=torch.float16 if GPU_CONFIG["precision"] == "fp16" else torch.float32,
            )
            
            model = model.to(self.device)
            self.models["clip_vision"] = model
            
            logger.info("CLIP vision model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load CLIP vision model: {e}")
            raise
    
    def download_model_if_needed(self, model_id: str, local_path: Path) -> bool:
        """Download model if not already present"""
        if local_path.exists():
            logger.info(f"Model already exists at {local_path}")
            return True
        
        logger.info(f"Downloading model {model_id} to {local_path}")
        
        try:
            from huggingface_hub import snapshot_download
            
            snapshot_download(
                repo_id=model_id,
                local_dir=local_path,
                local_dir_use_symlinks=False
            )
            
            logger.info(f"Model {model_id} downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            return False
    
    def preload_all_models(self):
        """Preload all models for faster inference"""
        logger.info("Preloading all models...")
        
        try:
            # Load SDXL pipeline
            self.load_sdxl_pipeline()
            
            # Load IPAdapter
            self.load_ip_adapter()
            
            # Load face enhancement
            self.load_face_enhancement_model()
            
            # Load CLIP vision
            self.load_clip_vision_model()
            
            logger.info("All models preloaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to preload models: {e}")
            raise
    
    def cleanup(self):
        """Clean up loaded models to free memory"""
        for model_name, model in self.models.items():
            if hasattr(model, 'to'):
                model.to('cpu')
            del model
        
        self.models.clear()
        torch.cuda.empty_cache()
        logger.info("Models cleaned up and memory freed")

def get_model_loader() -> ModelLoader:
    """Get singleton model loader instance"""
    if not hasattr(get_model_loader, '_instance'):
        get_model_loader._instance = ModelLoader()
    return get_model_loader._instance 