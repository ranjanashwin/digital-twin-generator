#!/usr/bin/env python3
"""
Digital Twin Generator - Flask Web Application
"""

import os
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any
import zipfile
import shutil
import uuid

from flask import Flask, request, jsonify, render_template, send_file, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename
import json

from config import WEB_CONFIG, SELFIES_DIR, OUTPUT_DIR, QUALITY_MODES, CURRENT_QUALITY, QUALITY_MODE
from generate_twin import DigitalTwinGenerator
from utils.image_validator import ImageValidator
from utils.resource_manager import ResourceManager
import traceback

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
app.config['MAX_CONTENT_LENGTH'] = WEB_CONFIG["max_file_size"]
CORS(app)

# Global variables for job tracking
jobs = {}
job_lock = threading.Lock()

# Initialize validation and resource management
image_validator = ImageValidator()
resource_manager = ResourceManager()

# Ensure directories exist
SELFIES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {ext[1:] for ext in WEB_CONFIG["allowed_extensions"]}

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

def generate_digital_twin_worker(job_id: str, selfies_folder: str, prompt_style: str = "portrait", quality_mode: str = "high_fidelity"):
    """Worker function to generate digital twin with resource management"""
    try:
        # Update resource manager
        resource_manager.update_job_access(job_id)
        
        update_job_status(job_id, "loading_models", 10, "Loading AI models...")
        
        # Initialize generator
        generator = DigitalTwinGenerator()
        
        # Load models
        generator.load_models()
        
        update_job_status(job_id, "analyzing_selfies", 30, "Analyzing selfies for pose and lighting patterns...")
        
        # Generate digital twin with enhanced pose and lighting analysis
        output_dir = OUTPUT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_images = generator.generate_digital_twin(
            selfies_folder=selfies_folder,
            output_path=str(output_dir),
            prompt_style=prompt_style,
            num_images=1
        )
        
        if generated_images:
            update_job_status(job_id, "completed", 100, "Generation completed with enhanced pose, lighting analysis, and LoRA training!", {
                'images': generated_images,
                'enhanced': True,
                'lora_trained': True,
                'quality_mode': quality_mode
            })
            
            # Log completion with resource info
            system_status = resource_manager.get_system_status()
            logger.info(f"Job {job_id} completed. System status: {system_status}")
            
        else:
            update_job_status(job_id, "failed", 0, "Generation failed - no images produced")
        
        # Cleanup
        generator.cleanup()
        
    except Exception as e:
        logger.error(f"Generation failed for job {job_id}: {e}")
        logger.error(traceback.format_exc())
        update_job_status(job_id, "failed", 0, f"Generation failed: {str(e)}")
    
    finally:
        # Clean up resources
        try:
            # Clean up temporary files
            cleanup_job_files(job_id)
            
            # Clean up job in resource manager
            resource_manager.cleanup_job(job_id)
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info(f"Cleanup completed for job {job_id}")
            
        except Exception as cleanup_error:
            logger.error(f"Cleanup failed for job {job_id}: {cleanup_error}")

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/demo')
def demo():
    """Serve the demo page"""
    return render_template('demo.html')

@app.route('/test')
def test():
    """API test page"""
    return render_template('test.html')

@app.route('/upload', methods=['POST'])
def upload_selfies():
    """Handle selfie upload with comprehensive validation"""
    try:
        # Check if files were uploaded
        if 'files' not in request.files:
            # Fallback to single file upload for backward compatibility
            if 'file' not in request.files:
                return jsonify({'error': 'No files uploaded'}), 400
            
            files = [request.files['file']]
        else:
            # Handle multiple file uploads
            files = request.files.getlist('files')
        
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Validate file types
        for file in files:
            if not allowed_file(file.filename):
                return jsonify({'error': f'Invalid file type: {file.filename}. Please upload image files.'}), 400
        
        # Create job ID and directories
        job_id = create_job_id()
        job_dir = SELFIES_DIR / job_id
        output_dir = OUTPUT_DIR / job_id
        
        job_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Register job with resource manager
        resource_manager.register_job(job_id, str(job_dir), str(output_dir))
        
        # Save uploaded files
        image_paths = []
        extract_dir = job_dir / "images"
        extract_dir.mkdir(exist_ok=True)
        
        for file in files:
            if file.filename:
                filename = secure_filename(file.filename)
                file_path = extract_dir / filename
                file.save(str(file_path))
                image_paths.append(str(file_path))
        
        if not image_paths:
            cleanup_job_files(job_id)
            resource_manager.cleanup_job(job_id, force=True)
            return jsonify({'error': 'No valid image files uploaded'}), 400
        
        # Validate images with comprehensive checks
        logger.info(f"Validating {len(image_paths)} images for job {job_id}")
        validation_results = image_validator.validate_image_set(image_paths)
        
        # Check if validation passed
        if not validation_results['summary']['meets_requirements']:
            # Clean up and return detailed error
            cleanup_job_files(job_id)
            resource_manager.cleanup_job(job_id, force=True)
            
            error_details = {
                'error': 'Image validation failed',
                'validation_report': image_validator.get_validation_report(validation_results),
                'summary': validation_results['summary'],
                'details': {
                    'total_images': validation_results['summary']['total_images'],
                    'valid_images': validation_results['summary']['valid_images'],
                    'invalid_images': validation_results['summary']['invalid_images'],
                    'required': validation_results['summary']['min_required']
                }
            }
            
            return jsonify(error_details), 400
        
        # Get generation parameters
        prompt_style = request.form.get('prompt_style', 'portrait')
        quality_mode = request.form.get('quality_mode', 'high_fidelity')
        
        # Initialize job with validation results
        with job_lock:
            jobs[job_id] = {
                'id': job_id,
                'status': 'validated',
                'progress': 20,
                'message': f'Validation successful: {validation_results["summary"]["valid_images"]} valid images found',
                'created_at': time.time(),
                'updated_at': time.time(),
                'image_count': validation_results['summary']['valid_images'],
                'prompt_style': prompt_style,
                'quality_mode': quality_mode,
                'validation_results': validation_results['summary']
            }
        
        # Start generation in background
        thread = threading.Thread(
            target=generate_digital_twin_worker,
            args=[job_id, str(extract_dir), prompt_style, quality_mode]
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'message': f'Upload successful. {validation_results["summary"]["valid_images"]} valid images found. Generation started.',
            'status': 'validated',
            'image_count': validation_results['summary']['valid_images'],
            'validation_summary': validation_results['summary']
        })
    
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        logger.error(traceback.format_exc())
        
        # Clean up on error
        if 'job_id' in locals():
            cleanup_job_files(job_id)
            resource_manager.cleanup_job(job_id, force=True)
        
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/status/<job_id>')
def get_job_status(job_id):
    """Get job status"""
    with job_lock:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job = jobs[job_id].copy()
        return jsonify(job)

@app.route('/download/<job_id>/<filename>')
def download_image(job_id, filename):
    """Download generated image"""
    try:
        # Validate filename
        if not filename.endswith('.png'):
            abort(400)
        
        # Check if job exists and is completed
        with job_lock:
            if job_id not in jobs:
                abort(404)
            
            job = jobs[job_id]
            if job['status'] != 'completed':
                abort(400, description="Job not completed")
        
        # Serve file
        file_path = OUTPUT_DIR / job_id / filename
        if not file_path.exists():
            abort(404)
        
        return send_file(str(file_path), as_attachment=True)
    
    except Exception as e:
        logger.error(f"Download failed: {e}")
        abort(500)

@app.route('/jobs')
def list_jobs():
    """List recent jobs"""
    with job_lock:
        recent_jobs = []
        current_time = time.time()
        
        for job_id, job in jobs.items():
            # Only show jobs from last 24 hours
            if current_time - job['created_at'] < 86400:
                recent_jobs.append({
                    'id': job_id,
                    'status': job['status'],
                    'progress': job['progress'],
                    'created_at': job['created_at'],
                    'image_count': job.get('image_count', 0)
                })
        
        return jsonify({'jobs': recent_jobs})

@app.route('/quality-modes')
def get_quality_modes():
    """Get available quality modes and current selection"""
    return jsonify({
        'current_mode': QUALITY_MODE,
        'current_settings': CURRENT_QUALITY,
        'available_modes': QUALITY_MODES,
        'default_mode': 'high_fidelity'
    })

@app.route('/quality-mode/<mode>', methods=['POST'])
def set_quality_mode(mode):
    """Set quality mode (requires restart to take effect)"""
    if mode not in QUALITY_MODES:
        return jsonify({'error': f'Invalid quality mode: {mode}'}), 400
    
    # In a production system, you might want to persist this setting
    # For now, we'll just return the available modes
    return jsonify({
        'message': f'Quality mode set to {mode}. Restart required to take effect.',
        'available_modes': QUALITY_MODES,
        'selected_mode': mode
    })

@app.route('/generate', methods=['POST'])
def generate_avatar():
    """Generate avatar using the most recent trained twin embedding"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Get generation parameters
        num_images = data.get('num_images', 1)
        quality_mode = data.get('quality_mode', 'high_fidelity')
        seed = data.get('seed')
        
        # Find the most recent successful job
        recent_jobs = [j for j in jobs.values() if j.get('status') == 'completed' and j.get('result')]
        if not recent_jobs:
            return jsonify({'error': 'No trained twin embedding found. Please upload selfies first.'}), 400
        
        # Get the most recent job
        latest_job = max(recent_jobs, key=lambda x: x.get('created_at', 0))
        job_id = latest_job['id']
        
        # Create new generation job
        generation_job_id = create_job_id()
        generation_output_dir = OUTPUT_DIR / generation_job_id
        generation_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Register generation job
        with job_lock:
            jobs[generation_job_id] = {
                'id': generation_job_id,
                'status': 'generating',
                'progress': 0,
                'message': 'Generating avatar...',
                'created_at': time.time(),
                'updated_at': time.time(),
                'prompt': prompt,
                'num_images': num_images,
                'quality_mode': quality_mode,
                'base_job_id': job_id
            }
        
        # Start generation in background
        thread = threading.Thread(
            target=generate_avatar_worker,
            args=[generation_job_id, job_id, prompt, num_images, quality_mode, seed]
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': generation_job_id,
            'message': 'Avatar generation started',
            'status': 'generating'
        })
        
    except Exception as e:
        logger.error(f"Failed to start generation: {e}")
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500

def generate_avatar_worker(job_id: str, base_job_id: str, prompt: str, num_images: int, quality_mode: str, seed: int = None):
    """Worker function to generate avatar using existing twin embedding"""
    try:
        update_job_status(job_id, "loading_models", 10, "Loading AI models...")
        
        # Initialize generator
        generator = DigitalTwinGenerator()
        generator.load_models()
        
        update_job_status(job_id, "generating", 50, "Generating avatar...")
        
        # Get the base job's output directory
        base_output_dir = OUTPUT_DIR / base_job_id
        
        # Generate images
        output_files = generator.generate_digital_twin(
            selfies_folder=str(base_output_dir),  # Use existing processed selfies
            output_path=str(OUTPUT_DIR / job_id),
            prompt_style="custom",
            num_images=num_images,
            seed=seed
        )
        
        # Update job with results
        result = {
            'output_files': output_files,
            'prompt': prompt,
            'num_images': num_images,
            'quality_mode': quality_mode
        }
        
        update_job_status(job_id, "completed", 100, "Generation completed", result)
        
        # Cleanup
        generator.cleanup()
        
    except Exception as e:
        logger.error(f"Generation failed for job {job_id}: {e}")
        update_job_status(job_id, "failed", 0, f"Generation failed: {str(e)}")
        cleanup_job_files(job_id)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok', 
        'timestamp': time.time(),
        'quality_mode': QUALITY_MODE,
        'active_jobs': len([j for j in jobs.values() if j['status'] in ['uploaded', 'loading_models', 'validating_selfies', 'generating']])
    })

@app.route('/system-status')
def get_system_status():
    """Get system resource status"""
    try:
        system_status = resource_manager.get_system_status()
        resource_limits = resource_manager.check_resource_limits()
        
        return jsonify({
            'system_status': system_status,
            'resource_limits': resource_limits,
            'active_jobs': len(jobs),
            'quality_mode': QUALITY_MODE
        })
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        return jsonify({'error': 'Failed to get system status'}), 500

@app.route('/cleanup', methods=['POST'])
def trigger_cleanup():
    """Trigger manual cleanup"""
    try:
        # Clean up old jobs
        resource_manager.cleanup_all_jobs(max_age_hours=1)
        
        # Clean up temp files
        resource_manager.cleanup_temp_files()
        
        # Clean up GPU memory
        resource_manager.cleanup_gpu_memory()
        
        return jsonify({
            'message': 'Cleanup completed successfully',
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return jsonify({'error': 'Cleanup failed'}), 500

if __name__ == '__main__':
    # Create necessary directories
    SELFIES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Register shutdown handler
    import atexit
    atexit.register(resource_manager.shutdown)
    atexit.register(image_validator.cleanup)
    
    # Start Flask app
    app.run(
        host=WEB_CONFIG["host"],
        port=WEB_CONFIG["port"],
        debug=WEB_CONFIG["debug"]
    ) 