"""
Configuration settings for Digital Twin Generator
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
SELFIES_DIR = BASE_DIR / "selfies"
OUTPUT_DIR = BASE_DIR / "output"
WEB_DIR = BASE_DIR / "web"

# Model paths
IP_ADAPTER_DIR = MODELS_DIR / "ip_adapter_faceid"
SDXL_DIR = MODELS_DIR / "sdxl"
FACE_ENHANCE_DIR = MODELS_DIR / "gfpgan"
INSIGHTFACE_DIR = MODELS_DIR / "insightface"

# Model URLs and configurations
MODEL_CONFIGS = {
    "ip_adapter": {
        "faceid": "h94/IP-Adapter",
        "faceid_plus": "h94/IP-Adapter",
        "faceid_plus_v2": "h94/IP-Adapter",
    },
    "sdxl": {
        "base": "stabilityai/stable-diffusion-xl-base-1.0",
        "refiner": "stabilityai/stable-diffusion-xl-refiner-1.0",
    },
    "face_enhance": {
        "gfpgan": "TencentARC/GFPGAN",
        "codeformer": "microsoft/CodeFormer",
    }
}

# Generation settings
GENERATION_CONFIG = {
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "num_images_per_prompt": 1,
    "seed": None,  # Random seed
}

# IPAdapter settings
IP_ADAPTER_CONFIG = {
    "weight": 0.8,
    "noise": 0.1,
    "start": 0.0,
    "end": 1.0,
}

# Face enhancement settings
FACE_ENHANCE_CONFIG = {
    "method": "gfpgan",  # "gfpgan" or "codeformer"
    "weight": 0.5,
    "background_enhance": True,
    "face_upsample": True,
}

# Web interface settings
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": False,
    "max_file_size": 100 * 1024 * 1024,  # 100MB
    "allowed_extensions": {'.zip', '.jpg', '.jpeg', '.png', '.webp'},
}

# Processing settings
PROCESSING_CONFIG = {
    "min_selfies": 15,
    "max_selfies": 50,
    "face_detection_confidence": 0.8,
    "face_size_threshold": 100,
    "output_format": "png",
    "quality": 95,
}

# Prompt templates
PROMPT_TEMPLATES = {
    "portrait": "a high-quality photorealistic portrait of the person, centered, realistic lighting, studio quality, DSLR shot, soft shadows, sharp features, professional photography",
    "casual": "a natural, candid photo of the person, good lighting, sharp focus, realistic skin texture",
    "professional": "a professional headshot of the person, studio lighting, clean background, business attire, high resolution",
}

# Negative prompts
NEGATIVE_PROMPTS = [
    "blurry", "low quality", "distorted", "deformed", "bad anatomy", "disfigured",
    "poorly drawn face", "mutation", "mutated", "extra limb", "ugly", "poorly drawn hands",
    "missing limb", "floating limbs", "disconnected limbs", "malformed hands", "out of focus",
    "long neck", "long body", "morbid", "mutilated", "extra limbs", "cloned face",
    "disfigured", "gross proportions", "malformed limbs", "missing arms", "missing legs",
    "extra arms", "extra legs", "fused fingers", "too many fingers", "long neck"
]

# Create directories if they don't exist
def create_directories():
    """Create necessary directories"""
    directories = [
        MODELS_DIR, IP_ADAPTER_DIR, SDXL_DIR, FACE_ENHANCE_DIR, INSIGHTFACE_DIR,
        SELFIES_DIR, OUTPUT_DIR, WEB_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# GPU settings
GPU_CONFIG = {
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu",
    "memory_fraction": 0.9,
    "precision": "fp16",  # "fp16" or "fp32"
    "attention_slicing": True,
    # Note: gradient_checkpointing is not available in current diffusers version
    # "gradient_checkpointing": True,
}

# Logging settings
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "digital_twin.log",
}

# Initialize directories
create_directories() 

# Generation Quality Modes
QUALITY_MODES = {
    "fast": {
        "name": "Fast",
        "description": "Lower detail, faster generation",
        "width": 768,
        "height": 768,
        "num_inference_steps": 20,
        "guidance_scale": 7.0,
        "lora_rank": 8,
        "lora_alpha": 16,
        "training_epochs": 25,
        "face_enhancement": "light",
        "controlnet_strength": 0.6,
        "ip_adapter_weight": 0.7,
        "estimated_time": "2-3 minutes"
    },
    "high_fidelity": {
        "name": "High Fidelity",
        "description": "Maximum detail, slower generation",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 50,
        "guidance_scale": 8.5,
        "lora_rank": 16,
        "lora_alpha": 32,
        "training_epochs": 50,
        "face_enhancement": "full",
        "controlnet_strength": 0.8,
        "ip_adapter_weight": 0.8,
        "estimated_time": "5-8 minutes"
    }
}

# Default quality mode
DEFAULT_QUALITY_MODE = "high_fidelity"

# Quality mode selection (can be overridden via environment variable)
QUALITY_MODE = os.getenv("AVATAR_QUALITY_MODE", DEFAULT_QUALITY_MODE)

# Validate quality mode
if QUALITY_MODE not in QUALITY_MODES:
    logger.warning(f"Invalid quality mode '{QUALITY_MODE}', using default '{DEFAULT_QUALITY_MODE}'")
    QUALITY_MODE = DEFAULT_QUALITY_MODE

# Get current quality settings
CURRENT_QUALITY = QUALITY_MODES[QUALITY_MODE]

# Update generation config with quality settings
GENERATION_CONFIG.update({
    "width": CURRENT_QUALITY["width"],
    "height": CURRENT_QUALITY["height"],
    "num_inference_steps": CURRENT_QUALITY["num_inference_steps"],
    "guidance_scale": CURRENT_QUALITY["guidance_scale"]
})

# Update IPAdapter config with quality settings
IP_ADAPTER_CONFIG.update({
    "weight": CURRENT_QUALITY["ip_adapter_weight"]
})

# LoRA quality settings
LORA_CONFIG = {
    "rank": CURRENT_QUALITY["lora_rank"],
    "alpha": CURRENT_QUALITY["lora_alpha"],
    "training_epochs": CURRENT_QUALITY["training_epochs"],
    "dropout": 0.1,
    "learning_rate": 1e-4,
    "batch_size": 2,
    "gradient_accumulation_steps": 4
}

# Face enhancement quality settings
FACE_ENHANCEMENT_CONFIG = {
    "mode": CURRENT_QUALITY["face_enhancement"],
    "gfpgan_strength": 0.8 if CURRENT_QUALITY["face_enhancement"] == "full" else 0.5,
    "codeformer_strength": 0.7 if CURRENT_QUALITY["face_enhancement"] == "full" else 0.4,
    "enhancement_method": "gfpgan" if CURRENT_QUALITY["face_enhancement"] == "full" else "light"
}

# ControlNet quality settings
CONTROLNET_CONFIG = {
    "pose_strength": CURRENT_QUALITY["controlnet_strength"],
    "depth_strength": CURRENT_QUALITY["controlnet_strength"] * 0.8,
    "enable_pose": True,
    "enable_depth": True
} 