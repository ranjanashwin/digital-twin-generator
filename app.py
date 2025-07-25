#!/usr/bin/env python3
"""
Digital Twin Generator - Fashion Content Creator Workflow
"""

import os
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import zipfile
import shutil
import uuid
from datetime import datetime
import json

from flask import Flask, request, jsonify, render_template, send_file, abort, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import traceback

from config import WEB_CONFIG, SELFIES_DIR, OUTPUT_DIR, QUALITY_MODES, CURRENT_QUALITY, QUALITY_MODE
from generate_twin import DigitalTwinGenerator
from utils.image_validator import ImageValidator
from utils.resource_manager import ResourceManager
from utils.viton_integration import VitonHDProcessor
from utils.controlnet_integration import ControlNetProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            static_folder='web/static',
            template_folder='web/templates')
# Flask configuration for large file uploads
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {
    'zip': 'application/zip',
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'webp': 'image/webp',
    'gif': 'image/gif',
    'bmp': 'image/bmp',
    'tiff': 'image/tiff',
    'tga': 'image/x-tga'
}

# Create upload directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('selfies', exist_ok=True)
os.makedirs('output', exist_ok=True)
os.makedirs('models', exist_ok=True)

CORS(app)

# Global variables for job tracking
jobs = {}
job_lock = threading.Lock()

# Global variable for session tracking
sessions = {}
session_lock = threading.Lock()

# Initialize validation and resource management
image_validator = ImageValidator()
resource_manager = ResourceManager()

# Initialize specialized processors
viton_processor = VitonHDProcessor()
controlnet_processor = ControlNetProcessor()

# Ensure directories exist
SELFIES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fashion workflow configuration
FASHION_WORKFLOW_STEPS = {
    "step1": {
        "name": "Upload Selfies",
        "description": "Upload 15+ selfies to create your digital twin",
        "status": "pending"
    },
    "step2": {
        "name": "Upload Clothing & Scene",
        "description": "Add clothing and describe the scene",
        "status": "pending"
    },
    "step3": {
        "name": "Generate Fashion Photo",
        "description": "Create your fashion content",
        "status": "pending"
    }
}

# Avatar styles for fashion content
FASHION_AVATAR_STYLES = {
    "fashion_portrait": {
        "name": "Fashion Portrait",
        "description": "High-fashion editorial style",
        "prompt_template": "a high-fashion portrait of the person, editorial lighting, professional photography, fashion magazine style, sharp details"
    },
    "street_style": {
        "name": "Street Style",
        "description": "Casual street fashion look",
        "prompt_template": "a street style photo of the person, natural lighting, urban background, candid fashion photography"
    },
    "studio_fashion": {
        "name": "Studio Fashion",
        "description": "Professional studio fashion shoot",
        "prompt_template": "a professional studio fashion photo of the person, studio lighting, clean background, high-end fashion photography"
    },
    "editorial": {
        "name": "Editorial",
        "description": "Magazine editorial style",
        "prompt_template": "an editorial fashion photo of the person, dramatic lighting, artistic composition, fashion magazine cover style"
    }
}

# Quality modes for fashion content
FASHION_QUALITY_MODES = {
    "standard": {
        "name": "Standard",
        "description": "Good quality, faster generation",
        "resolution": (768, 768),
        "steps": 20,
        "guidance_scale": 7.0,
        "estimated_time": "3-5 minutes"
    },
    "high_fidelity": {
        "name": "High Fidelity",
        "description": "Excellent quality, balanced speed",
        "resolution": (1024, 1024),
        "steps": 30,
        "guidance_scale": 8.0,
        "estimated_time": "5-8 minutes"
    },
    "ultra_fidelity": {
        "name": "Ultra Fidelity",
        "description": "Maximum quality, slower generation",
        "resolution": (1024, 1024),
        "steps": 50,
        "guidance_scale": 8.5,
        "estimated_time": "8-12 minutes"
    }
}

def allowed_file(filename, allowed_extensions=None):
    """Check if file extension is allowed"""
    if allowed_extensions is None:
        allowed_extensions = app.config['ALLOWED_EXTENSIONS']
    
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def secure_filename_with_timestamp(filename):
    """Create a secure filename with timestamp"""
    import time
    import uuid
    
    # Get file extension
    if '.' in filename:
        name, ext = filename.rsplit('.', 1)
        ext = f'.{ext.lower()}'
    else:
        name = filename
        ext = ''
    
    # Create unique filename
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    safe_name = f"{timestamp}_{unique_id}{ext}"
    
    return safe_name

def save_uploaded_file(file, directory='uploads'):
    """Save uploaded file with proper error handling"""
    try:
        if file and file.filename:
            # Validate file type
            if not allowed_file(file.filename):
                raise ValueError(f"File type not allowed: {file.filename}")
            
            # Create secure filename
            filename = secure_filename_with_timestamp(file.filename)
            filepath = os.path.join(directory, filename)
            
            # Ensure directory exists
            os.makedirs(directory, exist_ok=True)
            
            # Save file with chunked writing for large files
            file.save(filepath)
            
            # Verify file was saved
            if not os.path.exists(filepath):
                raise ValueError("File was not saved properly")
            
            return filepath
        else:
            raise ValueError("No file provided")
            
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise

def create_session_id():
    """Create unique session ID with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}"

def create_job_id():
    """Create unique job ID"""
    return str(uuid.uuid4())

def update_job_status(job_id: str, status: str, progress: int = 0, message: str = "", result: Dict[str, Any] = None):
    """Update job status"""
    with job_lock:
        if job_id in jobs:
            jobs[job_id].update({
                'status': status,
                'progress': progress,
                'message': message,
                'updated_at': time.time()
            })
            if result:
                jobs[job_id]['result'] = result

def cleanup_job_files(job_id: str):
    """Clean up temporary files for a job"""
    try:
        job_dir = SELFIES_DIR / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir)
        
        output_dir = OUTPUT_DIR / job_id
        if output_dir.exists():
            shutil.rmtree(output_dir)
    except Exception as e:
        logger.error(f"Failed to cleanup job files for {job_id}: {e}")

def validate_zip_file(zip_path: str) -> Dict[str, Any]:
    """Validate ZIP file contents"""
    try:
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        image_files = []
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            for file_name in file_list:
                if not file_name.endswith('/'):  # Skip directories
                    file_ext = Path(file_name).suffix.lower()
                    if file_ext in image_extensions:
                        image_files.append(file_name)
        
        if len(image_files) < 15:
            return {
                'valid': False,
                'error': f'Insufficient images. Found {len(image_files)} images, minimum 15 required.',
                'count': len(image_files)
            }
        
        return {
            'valid': True,
            'count': len(image_files),
            'files': image_files
        }
        
    except zipfile.BadZipFile:
        return {
            'valid': False,
            'error': 'Invalid ZIP file format.'
        }
    except Exception as e:
        return {
            'valid': False,
            'error': f'Error processing ZIP file: {str(e)}'
        }

def extract_zip_to_session(zip_path: str, session_dir: Path) -> List[str]:
    """Extract ZIP file to session directory and return image paths"""
    image_paths = []
    extract_dir = session_dir / "selfies"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        for file_path in extract_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_paths.append(str(file_path))
    
    return image_paths

def save_clothing_image(file, session_dir: Path) -> str:
    """Save clothing image to session directory"""
    if file and file.filename:
        filename = secure_filename(file.filename)
        clothing_path = session_dir / "clothing" / filename
        clothing_path.parent.mkdir(parents=True, exist_ok=True)
        file.save(str(clothing_path))
        return str(clothing_path)
    return None

def save_prompt_text(prompt: str, session_dir: Path):
    """Save prompt text to session directory"""
    prompt_file = session_dir / "prompt.txt"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(prompt)

def generate_avatar_worker(session_id: str, job_id: str, session_dir: Path, 
                          prompt: str, avatar_style: str, quality_mode: str):
    """Worker function to generate digital twin avatar"""
    try:
        # Update resource manager
        resource_manager.update_job_access(job_id)
        
        update_job_status(job_id, "loading_models", 10, "Loading AI models...")
        
        # Initialize generator
        generator = DigitalTwinGenerator()
        
        # Load models
        generator.load_models()
        
        update_job_status(job_id, "validating_images", 20, "Validating uploaded images...")
        
        # Validate selfies
        selfies_dir = session_dir / "selfies"
        
        # Check if selfies directory exists
        if not selfies_dir.exists():
            raise ValueError(f"Selfies directory not found: {selfies_dir}")
        
        # Get all image files
        image_paths = list(selfies_dir.glob("*"))
        image_paths = [str(p) for p in image_paths if p.is_file()]
        
        logger.info(f"Found {len(image_paths)} files in selfies directory")
        
        # Filter for valid image files
        valid_image_paths = [p for p in image_paths if allowed_file(Path(p).name, {'png', 'jpg', 'jpeg', 'webp'})]
        
        logger.info(f"Found {len(valid_image_paths)} valid image files")
        
        if len(valid_image_paths) < 15:
            raise ValueError(f"Insufficient images. Found {len(valid_image_paths)} valid images, minimum 15 required.")
        
        # Validate images with comprehensive checks
        validation_results = image_validator.validate_image_set(valid_image_paths)
        if not validation_results['summary']['meets_requirements']:
            raise ValueError(f"Image validation failed: {validation_results['summary']}")
        
        update_job_status(job_id, "analyzing_pose_lighting", 30, "Analyzing pose and lighting patterns...")
        
        # Analyze pose and lighting
        pose_analysis = generator.analyze_pose_and_lighting(str(selfies_dir))
        
        update_job_status(job_id, "creating_lora_embedding", 40, "Creating personalized identity embedding...")
        
        # Create LoRA embedding
        lora_metadata = generator.create_lora_embedding(str(selfies_dir), session_id)
        
        update_job_status(job_id, "generating_avatar", 50, "Generating your digital twin...")
        
        # Prepare output directory
        output_dir = session_dir / "avatar_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate avatar with enhanced prompt
        style_config = FASHION_AVATAR_STYLES.get(avatar_style, FASHION_AVATAR_STYLES["fashion_portrait"])
        base_prompt = style_config["prompt_template"]
        
        # Combine custom prompt with style template
        if prompt and prompt.strip():
            enhanced_prompt = f"{base_prompt}, {prompt.strip()}"
        else:
            enhanced_prompt = base_prompt
        
        # Apply pose and lighting analysis
        enhanced_prompt = generator._enhance_prompt_with_analysis(enhanced_prompt, pose_analysis)
        
        # Generate avatar
        generated_images = generator.generate_digital_twin(
            selfies_folder=str(selfies_dir),
            output_path=str(output_dir),
            prompt_style=avatar_style,
            num_images=1,
            seed=None
        )
        
        update_job_status(job_id, "enhancing_face", 80, "Enhancing facial details...")
        
        # Enhance the generated image
        if generated_images:
            enhanced_image_path = generator.enhance_face(generated_images[0])
            final_image_path = enhanced_image_path
        else:
            raise ValueError("Failed to generate avatar image")
        
        update_job_status(job_id, "completed", 100, "Avatar generation completed!", {
            'session_id': session_id,
            'avatar_path': str(final_image_path),
            'prompt': enhanced_prompt,
            'style': avatar_style,
            'quality_mode': quality_mode,
            'image_count': len(valid_image_paths),
            'validation_summary': validation_results['summary'],
            'workflow_step': 'step1_completed'
        })
        
        # Cleanup
        generator.cleanup()
        
    except Exception as e:
        logger.error(f"Avatar generation failed for job {job_id}: {e}")
        logger.error(traceback.format_exc())
        update_job_status(job_id, "failed", 0, f"Avatar generation failed: {str(e)}")
        
        # Cleanup on failure
        try:
            generator.cleanup()
        except:
            pass

def generate_fashion_photo_worker(session_id: str, job_id: str, session_dir: Path,
                                 clothing_path: str, scene_prompt: str, quality_mode: str):
    """Worker function to generate fashion photo with clothing and scene"""
    try:
        # Update resource manager
        resource_manager.update_job_access(job_id)
        
        update_job_status(job_id, "loading_fashion_models", 10, "Loading fashion AI models...")
        
        # Initialize processors
        viton_processor.load_models()
        controlnet_processor.load_models()
        
        update_job_status(job_id, "processing_clothing", 20, "Processing clothing image...")
        
        # Process clothing with VITON-HD
        clothing_processed = viton_processor.process_clothing(clothing_path)
        
        update_job_status(job_id, "generating_pose", 30, "Generating optimal pose...")
        
        # Get avatar from previous step
        avatar_dir = session_dir / "avatar_output"
        avatar_files = list(avatar_dir.glob("*.png"))
        if not avatar_files:
            raise ValueError("No avatar found. Please complete step 1 first.")
        
        avatar_path = str(avatar_files[0])
        
        update_job_status(job_id, "applying_controlnet", 40, "Applying ControlNet for pose alignment...")
        
        # Apply ControlNet for pose and depth
        pose_conditioning = controlnet_processor.create_pose_conditioning(avatar_path, clothing_processed)
        
        update_job_status(job_id, "generating_fashion_photo", 50, "Generating fashion photo...")
        
        # Prepare output directory
        output_dir = session_dir / "fashion_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate fashion photo with all components
        fashion_image_path = controlnet_processor.generate_fashion_photo(
            avatar_path=avatar_path,
            clothing_path=clothing_processed,
            scene_prompt=scene_prompt,
            pose_conditioning=pose_conditioning,
            output_path=str(output_dir),
            quality_mode=quality_mode
        )
        
        update_job_status(job_id, "enhancing_result", 80, "Enhancing final result...")
        
        # Enhance the final image
        enhanced_fashion_path = controlnet_processor.enhance_image(fashion_image_path)
        
        update_job_status(job_id, "completed", 100, "Fashion photo generation completed!", {
            'session_id': session_id,
            'fashion_photo_path': str(enhanced_fashion_path),
            'clothing_path': clothing_path,
            'scene_prompt': scene_prompt,
            'quality_mode': quality_mode,
            'workflow_step': 'step3_completed'
        })
        
        # Cleanup
        viton_processor.cleanup()
        controlnet_processor.cleanup()
        
    except Exception as e:
        logger.error(f"Fashion photo generation failed for job {job_id}: {e}")
        logger.error(traceback.format_exc())
        update_job_status(job_id, "failed", 0, f"Fashion photo generation failed: {str(e)}")
        
        # Cleanup on failure
        try:
            viton_processor.cleanup()
            controlnet_processor.cleanup()
        except:
            pass

def generate_fashion_photo_job(job_id: str, session_id: str, job_data: Dict[str, Any]):
    """Generate fashion photo with advanced pose control and multiple clothing items"""
    try:
        session = sessions[session_id]
        data = job_data['data']
        
        # Update job status
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['progress'] = 10
        jobs[job_id]['message'] = 'Loading models and processing clothing...'
        
        # Initialize processors
        controlnet_processor = ControlNetProcessor()
        viton_processor = VitonHDProcessor()
        
        # Load models
        controlnet_processor.load_models()
        viton_processor.load_models()
        
        jobs[job_id]['progress'] = 30
        jobs[job_id]['message'] = 'Creating pose conditioning...'
        
        # Create pose conditioning with advanced control
        pose_conditioning = controlnet_processor.create_pose_conditioning(
            avatar_path=data['avatar_path'],
            clothing_path=data['clothing_items'][0]['image_path'],  # Use first clothing for pose
            reference_pose_path=data.get('reference_pose_path'),
            pose_preset=data.get('pose_preset')
        )
        
        jobs[job_id]['progress'] = 50
        jobs[job_id]['message'] = 'Processing multiple clothing items...'
        
        # Process multiple clothing items
        processed_clothing_paths = viton_processor.process_multiple_clothing(data['clothing_items'])
        
        jobs[job_id]['progress'] = 70
        jobs[job_id]['message'] = 'Generating virtual try-on...'
        
        # Generate virtual try-on with multiple clothing items
        tryon_result = viton_processor.generate_virtual_tryon_multiple(
            person_image=data['avatar_path'],
            clothing_items=data['clothing_items'],
            pose_data=pose_conditioning.get('pose_data')
        )
        
        jobs[job_id]['progress'] = 85
        jobs[job_id]['message'] = 'Generating final fashion photo...'
        
        # Generate final fashion photo
        output_dir = session['output_dir']
        fashion_photo_path = controlnet_processor.generate_fashion_photo(
            avatar_path=data['avatar_path'],
            clothing_path=tryon_result,  # Use try-on result as clothing
            scene_prompt=data['scene_prompt'],
            pose_conditioning=pose_conditioning,
            output_path=output_dir,
            quality_mode=data['quality_mode']
        )
        
        # Save session data
        session['fashion_photo_path'] = fashion_photo_path
        session['clothing_items'] = data['clothing_items']
        session['scene_prompt'] = data['scene_prompt']
        session['pose_conditioning'] = pose_conditioning
        session['reference_pose_path'] = data.get('reference_pose_path')
        session['pose_preset'] = data.get('pose_preset')
        
        # Update job status
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['message'] = 'Fashion photo generated successfully!'
        jobs[job_id]['result'] = {
            'fashion_photo_path': fashion_photo_path,
            'clothing_items': data['clothing_items'],
            'scene_prompt': data['scene_prompt'],
            'compatibility_result': data['compatibility_result'],
            'suggestions': data['suggestions']
        }
        
        logger.info(f"Fashion photo generation completed: {job_id}")
        
    except Exception as e:
        logger.error(f"Error in generate_fashion_photo_job: {e}")
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['message'] = f'Error: {str(e)}'

@app.route('/')
def index():
    """Main page - Fashion workflow"""
    return render_template('fashion_workflow.html')

@app.route('/demo')
def demo():
    """Demo page"""
    return render_template('demo.html')

@app.route('/test')
def test():
    """Test page"""
    return render_template('test.html')

@app.route('/upload-selfies', methods=['POST'])
def upload_selfies():
    """Upload selfies ZIP file for avatar generation"""
    try:
        logger.info("Starting selfies upload process...")
        
        # Check if file was uploaded
        if 'selfies_zip' not in request.files:
            logger.error("No file uploaded")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['selfies_zip']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        logger.info(f"Processing file: {file.filename}")
        
        # Validate file type
        if not allowed_file(file.filename, {'zip'}):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Please upload a ZIP file containing selfies'}), 400
        
        # Get form data
        avatar_style = request.form.get('avatar_style', 'fashion_portrait')
        custom_prompt = request.form.get('custom_prompt', '').strip()
        quality_mode = request.form.get('quality_mode', 'high_fidelity')
        
        logger.info(f"Form data - style: {avatar_style}, quality: {quality_mode}")
        
        # Create session
        session_id = create_session_id()
        session_dir = OUTPUT_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created session directory: {session_dir}")
        
        # Save uploaded file
        try:
            logger.info("Saving uploaded file...")
            uploaded_file_path = save_uploaded_file(file, str(session_dir))
            logger.info(f"File saved to: {uploaded_file_path}")
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            return jsonify({'error': f'File upload failed: {str(e)}'}), 500
        
        # Validate ZIP contents
        try:
            logger.info("Validating ZIP contents...")
            import zipfile
            with zipfile.ZipFile(uploaded_file_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                image_files = [f for f in file_list if allowed_file(f, {'png', 'jpg', 'jpeg', 'webp'})]
                
                logger.info(f"Found {len(image_files)} image files in ZIP")
                
                if len(image_files) < 15:
                    logger.error(f"Insufficient images: {len(image_files)}")
                    return jsonify({'error': f'ZIP file must contain at least 15 selfie images. Found: {len(image_files)}'}), 400
                
                logger.info(f"ZIP contains {len(image_files)} valid image files")
                
                # Extract ZIP file to session directory
                extract_dir = session_dir / "selfies"
                extract_dir.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Extracting ZIP to: {extract_dir}")
                zip_ref.extractall(extract_dir)
                
                # Verify extraction
                extracted_images = list(extract_dir.glob("*"))
                extracted_images = [f for f in extracted_images if f.is_file() and allowed_file(f.name, {'png', 'jpg', 'jpeg', 'webp'})]
                
                logger.info(f"Extracted {len(extracted_images)} image files")
                
                if len(extracted_images) < 15:
                    logger.error(f"Failed to extract enough images: {len(extracted_images)}")
                    return jsonify({'error': f'Failed to extract enough images. Found: {len(extracted_images)}'}), 400
                
        except zipfile.BadZipFile:
            logger.error("Invalid ZIP file format")
            return jsonify({'error': 'Invalid ZIP file format'}), 400
        except Exception as e:
            logger.error(f"ZIP validation/extraction failed: {e}")
            return jsonify({'error': f'ZIP processing failed: {str(e)}'}), 500
        
        # Create job for avatar generation
        job_id = f"avatar_{session_id}_{int(time.time())}"
        
        logger.info(f"Creating job: {job_id}")
        
        job_data = {
            'type': 'avatar',
            'status': 'pending',
            'progress': 0,
            'message': 'Initializing avatar generation...',
            'data': {
                'session_id': session_id,
                'uploaded_file_path': uploaded_file_path,
                'avatar_style': avatar_style,
                'custom_prompt': custom_prompt,
                'quality_mode': quality_mode,
                'session_dir': str(session_dir)
            },
            'created_at': time.time()
        }
        
        jobs[job_id] = job_data
        
        # Start avatar generation in background
        logger.info("Starting avatar generation worker...")
        thread = threading.Thread(
            target=generate_avatar_worker,
            args=[session_id, job_id, session_dir, custom_prompt, avatar_style, quality_mode]
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started avatar generation job: {job_id}")
        
        response_data = {
            'success': True,
            'session_id': session_id,
            'job_id': job_id,
            'message': 'Selfies uploaded successfully. Avatar generation started.'
        }
        
        logger.info(f"Upload completed successfully. Returning response: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in upload_selfies: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload-clothing-scene', methods=['POST'])
def upload_clothing_scene():
    """Upload clothing and scene description for fashion photo generation"""
    try:
        logger.info("Starting clothing and scene upload process...")
        
        # Get session ID
        session_id = request.form.get('session_id')
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        # Check if avatar is ready
        session_dir = OUTPUT_DIR / session_id
        if not session_dir.exists():
            return jsonify({'error': 'Session not found'}), 400
        
        avatar_dir = session_dir / "avatar_output"
        if not avatar_dir.exists() or not list(avatar_dir.glob("*.png")):
            return jsonify({'error': 'Avatar generation not complete. Please complete step 1 first.'}), 400
        
        # Get clothing files (multiple files supported)
        clothing_files = request.files.getlist('clothing_files')
        if not clothing_files or all(f.filename == '' for f in clothing_files):
            return jsonify({'error': 'No clothing files provided'}), 400
        
        # Get reference pose file (optional)
        reference_pose_file = request.files.get('reference_pose_file')
        pose_preset = request.form.get('pose_preset')  # Optional preset pose
        
        # Get scene prompt
        scene_prompt = request.form.get('scene_prompt', '').strip()
        if not scene_prompt:
            return jsonify({'error': 'Scene prompt is required'}), 400
        
        # Get quality mode
        quality_mode = request.form.get('quality_mode', 'high_fidelity')
        
        # Save clothing files
        clothing_paths = []
        clothing_items = []
        
        for i, file in enumerate(clothing_files):
            if file and file.filename:
                # Validate file type
                if not allowed_file(file.filename, {'png', 'jpg', 'jpeg', 'webp', 'gif'}):
                    return jsonify({'error': f'Invalid file type for clothing {i+1}: {file.filename}'}), 400
                
                try:
                    # Save clothing file
                    filename = secure_filename_with_timestamp(file.filename)
                    clothing_path = os.path.join(str(session_dir), filename)
                    file.save(clothing_path)
                    clothing_paths.append(clothing_path)
                    
                    # Create clothing item data
                    clothing_item = {
                        'image_path': clothing_path,
                        'type': request.form.get(f'clothing_type_{i}', 'top'),
                        'layer': int(request.form.get(f'clothing_layer_{i}', 2)),
                        'name': request.form.get(f'clothing_name_{i}', f'Clothing {i+1}')
                    }
                    clothing_items.append(clothing_item)
                    
                    logger.info(f"Saved clothing file {i+1}: {clothing_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to save clothing file {i+1}: {e}")
                    return jsonify({'error': f'Failed to save clothing file {i+1}: {str(e)}'}), 500
        
        # Save reference pose file if provided
        reference_pose_path = None
        if reference_pose_file and reference_pose_file.filename:
            if not allowed_file(reference_pose_file.filename, {'png', 'jpg', 'jpeg', 'webp'}):
                return jsonify({'error': 'Invalid file type for reference pose'}), 400
            
            try:
                filename = secure_filename_with_timestamp(reference_pose_file.filename)
                reference_pose_path = os.path.join(str(session_dir), filename)
                reference_pose_file.save(reference_pose_path)
                logger.info(f"Saved reference pose file: {reference_pose_path}")
            except Exception as e:
                logger.error(f"Failed to save reference pose file: {e}")
                return jsonify({'error': f'Failed to save reference pose file: {str(e)}'}), 500
        
        # Validate clothing compatibility
        try:
            viton_processor = VitonHDProcessor()
            compatibility_result = viton_processor.validate_clothing_compatibility(clothing_items)
            
            if not compatibility_result['compatible']:
                conflicts = compatibility_result.get('conflicts', [])
                layer_conflicts = compatibility_result.get('layer_conflicts', [])
                
                error_messages = []
                for conflict in conflicts:
                    error_messages.append(f"{conflict['item1']} and {conflict['item2']}: {conflict['reason']}")
                for conflict in layer_conflicts:
                    error_messages.append(f"Layer conflict: {conflict['reason']}")
                
                return jsonify({
                    'error': 'Clothing compatibility issues',
                    'details': error_messages
                }), 400
            
            # Get clothing suggestions
            suggestions = viton_processor.get_clothing_suggestions(clothing_items)
            
        except Exception as e:
            logger.error(f"Clothing compatibility check failed: {e}")
            return jsonify({'error': f'Clothing compatibility check failed: {str(e)}'}), 500
        
        # Create job for fashion photo generation
        job_id = f"fashion_{session_id}_{int(time.time())}"
        
        job_data = {
            'type': 'fashion_photo',
            'status': 'pending',
            'progress': 0,
            'message': 'Initializing fashion photo generation...',
            'data': {
                'session_id': session_id,
                'clothing_items': clothing_items,
                'scene_prompt': scene_prompt,
                'quality_mode': quality_mode,
                'reference_pose_path': reference_pose_path,
                'pose_preset': pose_preset,
                'compatibility_result': compatibility_result,
                'suggestions': suggestions,
                'session_dir': str(session_dir)
            },
            'created_at': time.time()
        }
        
        jobs[job_id] = job_data
        
        # Start fashion photo generation in background
        thread = threading.Thread(
            target=generate_fashion_photo_job,
            args=(job_id, session_id, job_data)
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started fashion photo generation job: {job_id}")
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Fashion photo generation started',
            'compatibility_result': compatibility_result,
            'suggestions': suggestions
        })
        
    except Exception as e:
        logger.error(f"Error in upload_clothing_scene: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status/<job_id>')
def get_job_status(job_id):
    """Get job status"""
    with job_lock:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job = jobs[job_id]
        return jsonify(job)

@app.route('/download/<job_id>/<filename>')
def download_image(job_id, filename):
    """Download generated image"""
    try:
        with job_lock:
            if job_id not in jobs:
                abort(404)
            
            job = jobs[job_id]
            if job.get('status') != 'completed':
                return jsonify({'error': 'Job not completed'}), 400
            
            session_id = job.get('session_id')
            if not session_id:
                return jsonify({'error': 'Session not found'}), 400
        
        # Construct file path
        session_dir = OUTPUT_DIR / session_id
        
        # Check different output directories
        possible_paths = [
            session_dir / "avatar_output" / filename,
            session_dir / "fashion_output" / filename,
            session_dir / "output" / filename
        ]
        
        file_path = None
        for path in possible_paths:
            if path.exists():
                file_path = path
                break
        
        if not file_path:
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(str(file_path), as_attachment=True)
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/session/<session_id>')
def get_session_results(session_id):
    """Get session results and workflow status"""
    try:
        session_dir = OUTPUT_DIR / session_id
        if not session_dir.exists():
            return jsonify({'error': 'Session not found'}), 404
        
        # Check workflow progress
        workflow_status = {
            'step1': {'status': 'pending', 'completed': False},
            'step2': {'status': 'pending', 'completed': False},
            'step3': {'status': 'pending', 'completed': False}
        }
        
        # Check step 1 (avatar generation)
        avatar_dir = session_dir / "avatar_output"
        if avatar_dir.exists() and list(avatar_dir.glob("*.png")):
            workflow_status['step1']['status'] = 'completed'
            workflow_status['step1']['completed'] = True
        
        # Check step 2 (clothing upload)
        clothing_dir = session_dir / "clothing"
        if clothing_dir.exists() and list(clothing_dir.glob("*")):
            workflow_status['step2']['status'] = 'completed'
            workflow_status['step2']['completed'] = True
        
        # Check step 3 (fashion photo generation)
        fashion_dir = session_dir / "fashion_output"
        if fashion_dir.exists() and list(fashion_dir.glob("*.png")):
            workflow_status['step3']['status'] = 'completed'
            workflow_status['step3']['completed'] = True
        
        # Get result files
        results = {
            'avatar_files': [],
            'fashion_files': [],
            'clothing_files': []
        }
        
        if avatar_dir.exists():
            results['avatar_files'] = [f.name for f in avatar_dir.glob("*.png")]
        
        if fashion_dir.exists():
            results['fashion_files'] = [f.name for f in fashion_dir.glob("*.png")]
        
        if clothing_dir.exists():
            results['clothing_files'] = [f.name for f in clothing_dir.glob("*")]
        
        return jsonify({
            'session_id': session_id,
            'workflow_status': workflow_status,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Failed to get session results: {e}")
        return jsonify({'error': f'Failed to get session results: {str(e)}'}), 500

@app.route('/jobs')
def list_jobs():
    """List recent jobs"""
    with job_lock:
        recent_jobs = []
        for job_id, job in jobs.items():
            if time.time() - job.get('created_at', 0) < 3600:  # Last hour
                recent_jobs.append({
                    'job_id': job_id,
                    'session_id': job.get('session_id'),
                    'status': job.get('status'),
                    'created_at': job.get('created_at'),
                    'workflow_step': job.get('workflow_step'),
                    'quality_mode': job.get('quality_mode')
                })
        
        return jsonify({'jobs': recent_jobs})

@app.route('/fashion-styles')
def get_fashion_styles():
    """Get available fashion avatar styles"""
    return jsonify({'styles': FASHION_AVATAR_STYLES})

@app.route('/fashion-quality-modes')
def get_fashion_quality_modes():
    """Get available fashion quality modes"""
    return jsonify({'modes': FASHION_QUALITY_MODES})

@app.route('/api/pose-presets', methods=['GET'])
def get_pose_presets():
    """Get available pose presets for advanced pose control"""
    try:
        controlnet_processor = ControlNetProcessor()
        presets = controlnet_processor.get_available_pose_presets()
        
        return jsonify({
            'success': True,
            'presets': presets
        })
        
    except Exception as e:
        logger.error(f"Error getting pose presets: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clothing-types', methods=['GET'])
def get_clothing_types():
    """Get available clothing types and layers for multiple clothing items"""
    try:
        clothing_types = {
            'types': [
                {'value': 'top', 'label': 'Top (Shirt, Blouse, etc.)'},
                {'value': 'bottom', 'label': 'Bottom (Pants, Skirt, etc.)'},
                {'value': 'dress', 'label': 'Dress'},
                {'value': 'outerwear', 'label': 'Outerwear (Jacket, Coat, etc.)'},
                {'value': 'accessories', 'label': 'Accessories (Hat, Scarf, etc.)'},
                {'value': 'shoes', 'label': 'Shoes'}
            ],
            'layers': [
                {'value': 0, 'label': 'Underwear'},
                {'value': 1, 'label': 'Bottom'},
                {'value': 2, 'label': 'Top'},
                {'value': 3, 'label': 'Outerwear'},
                {'value': 4, 'label': 'Accessories'}
            ]
        }
        
        return jsonify({
            'success': True,
            'clothing_types': clothing_types
        })
        
    except Exception as e:
        logger.error(f"Error getting clothing types: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'timestamp': time.time(),
        'active_jobs': len([j for j in jobs.values() if j.get('status') in ['validated', 'loading_models', 'generating']]),
        'workflow_steps': FASHION_WORKFLOW_STEPS
    })

@app.route('/system-status')
def get_system_status():
    """Get system resource status"""
    try:
        import psutil
        
        # Get system info
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get GPU info if available
        gpu_info = {}
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = {
                    'available': True,
                    'device_count': torch.cuda.device_count(),
                    'current_device': torch.cuda.current_device(),
                    'device_name': torch.cuda.get_device_name(0),
                    'memory_allocated': torch.cuda.memory_allocated(0) / 1024**3,  # GB
                    'memory_reserved': torch.cuda.memory_reserved(0) / 1024**3,  # GB
                }
        except:
            gpu_info = {'available': False}
        
        return jsonify({
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / 1024**3,
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / 1024**3
            },
            'gpu': gpu_info,
            'jobs': {
                'total': len(jobs),
                'active': len([j for j in jobs.values() if j.get('status') in ['validated', 'loading_models', 'generating']]),
                'completed': len([j for j in jobs.values() if j.get('status') == 'completed']),
                'failed': len([j for j in jobs.values() if j.get('status') == 'failed'])
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        return jsonify({'error': f'Failed to get system status: {str(e)}'}), 500

@app.route('/cleanup', methods=['POST'])
def trigger_cleanup():
    """Trigger manual cleanup"""
    try:
        # Cleanup old jobs
        current_time = time.time()
        with job_lock:
            old_jobs = [job_id for job_id, job in jobs.items() 
                       if current_time - job.get('created_at', 0) > 3600]  # 1 hour
        
        for job_id in old_jobs:
            cleanup_job_files(job_id)
            del jobs[job_id]
        
        # Trigger resource manager cleanup
        resource_manager.cleanup_old_jobs()
        
        return jsonify({
            'message': f'Cleanup completed. Removed {len(old_jobs)} old jobs.',
            'removed_jobs': len(old_jobs)
        })
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

# Register shutdown handlers
import atexit
atexit.register(resource_manager.shutdown)

if __name__ == '__main__':
    logger.info("Starting Fashion Digital Twin Generator Flask app...")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    ) 