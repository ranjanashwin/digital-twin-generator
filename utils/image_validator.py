"""
Image Validation System
Validates uploaded images for resolution, face detection, and quality requirements
"""

import cv2
import numpy as np
from PIL import Image
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import insightface
from insightface.app import FaceAnalysis
import os
import shutil
import tempfile

logger = logging.getLogger(__name__)

class ImageValidator:
    """Validates images for avatar generation requirements"""
    
    def __init__(self):
        self.face_analyzer = None
        self._initialize_face_analyzer()
        
        # Validation parameters
        self.min_resolution = (512, 512)  # Minimum width x height
        self.min_face_size = 100  # Minimum face size in pixels
        self.max_file_size = 10 * 1024 * 1024  # 10MB max per file
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        self.min_valid_images = 15
        
    def _initialize_face_analyzer(self):
        """Initialize InsightFace for face detection"""
        try:
            self.face_analyzer = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("Face analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize face analyzer: {e}")
            self.face_analyzer = None
    
    def validate_image_set(self, image_paths: List[str]) -> Dict:
        """Validate a set of images for avatar generation"""
        logger.info(f"Validating {len(image_paths)} images...")
        
        validation_results = {
            'valid_images': [],
            'invalid_images': [],
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        if len(image_paths) < self.min_valid_images:
            error_msg = f"Not enough images provided. Found {len(image_paths)}, need at least {self.min_valid_images}."
            validation_results['errors'].append(error_msg)
            logger.error(error_msg)
            return validation_results
        
        # Validate each image
        for image_path in image_paths:
            try:
                result = self._validate_single_image(image_path)
                if result['is_valid']:
                    validation_results['valid_images'].append({
                        'path': image_path,
                        'metadata': result['metadata']
                    })
                else:
                    validation_results['invalid_images'].append({
                        'path': image_path,
                        'errors': result['errors']
                    })
                    
            except Exception as e:
                logger.error(f"Failed to validate image {image_path}: {e}")
                validation_results['invalid_images'].append({
                    'path': image_path,
                    'errors': [f"Validation failed: {str(e)}"]
                })
        
        # Generate summary
        validation_results['summary'] = self._generate_validation_summary(validation_results)
        
        # Check if we have enough valid images
        if len(validation_results['valid_images']) < self.min_valid_images:
            error_msg = f"Not enough valid images. Found {len(validation_results['valid_images'])}, need at least {self.min_valid_images}."
            validation_results['errors'].append(error_msg)
        
        logger.info(f"Validation complete: {len(validation_results['valid_images'])} valid, {len(validation_results['invalid_images'])} invalid")
        return validation_results
    
    def _validate_single_image(self, image_path: str) -> Dict:
        """Validate a single image"""
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'metadata': {}
        }
        
        try:
            # Check file format
            if not self._validate_file_format(image_path):
                result['is_valid'] = False
                result['errors'].append("Unsupported file format")
                return result
            
            # Check file size
            if not self._validate_file_size(image_path):
                result['is_valid'] = False
                result['errors'].append("File too large (max 10MB)")
                return result
            
            # Load and validate image
            image = self._load_image(image_path)
            if image is None:
                result['is_valid'] = False
                result['errors'].append("Failed to load image")
                return result
            
            # Check resolution
            resolution_valid, resolution_info = self._validate_resolution(image)
            if not resolution_valid:
                result['is_valid'] = False
                result['errors'].append(f"Resolution too low: {resolution_info}")
            else:
                result['metadata']['resolution'] = resolution_info
            
            # Check image quality
            quality_valid, quality_info = self._validate_image_quality(image)
            if not quality_valid:
                result['warnings'].append(f"Image quality issues: {quality_info}")
            result['metadata']['quality'] = quality_info
            
            # Check face detection
            face_valid, face_info = self._validate_face_detection(image_path)
            if not face_valid:
                result['is_valid'] = False
                result['errors'].append(f"Face detection failed: {face_info}")
            else:
                result['metadata']['face'] = face_info
            
            # Check for multiple faces
            if face_valid and face_info.get('face_count', 0) > 1:
                result['warnings'].append("Multiple faces detected - using primary face")
            
        except Exception as e:
            result['is_valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_file_format(self, image_path: str) -> bool:
        """Check if file format is supported"""
        file_ext = Path(image_path).suffix.lower()
        return file_ext in self.supported_formats
    
    def _validate_file_size(self, image_path: str) -> bool:
        """Check if file size is within limits"""
        try:
            file_size = os.path.getsize(image_path)
            return file_size <= self.max_file_size
        except Exception:
            return False
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image and convert to numpy array"""
        try:
            # Load with PIL first for format validation
            pil_image = Image.open(image_path)
            pil_image.verify()  # Verify image integrity
            
            # Load with OpenCV for processing
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def _validate_resolution(self, image: np.ndarray) -> Tuple[bool, Dict]:
        """Validate image resolution"""
        height, width = image.shape[:2]
        min_width, min_height = self.min_resolution
        
        resolution_info = {
            'width': width,
            'height': height,
            'aspect_ratio': width / height if height > 0 else 0
        }
        
        if width < min_width or height < min_height:
            return False, resolution_info
        
        return True, resolution_info
    
    def _validate_image_quality(self, image: np.ndarray) -> Tuple[bool, Dict]:
        """Validate image quality metrics"""
        quality_info = {}
        
        # Check brightness
        brightness = np.mean(image)
        quality_info['brightness'] = brightness
        
        # Check contrast
        contrast = np.std(image)
        quality_info['contrast'] = contrast
        
        # Check blur (using Laplacian variance)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality_info['blur_score'] = blur_score
        
        # Determine if quality is acceptable
        quality_issues = []
        
        if brightness < 30:
            quality_issues.append("Image too dark")
        elif brightness > 220:
            quality_issues.append("Image too bright")
        
        if contrast < 20:
            quality_issues.append("Low contrast")
        
        if blur_score < 100:
            quality_issues.append("Image may be blurry")
        
        quality_info['issues'] = quality_issues
        quality_valid = len(quality_issues) == 0
        
        return quality_valid, quality_info
    
    def _validate_face_detection(self, image_path: str) -> Tuple[bool, Dict]:
        """Validate face detection using InsightFace"""
        if self.face_analyzer is None:
            return False, {"error": "Face analyzer not initialized"}
        
        try:
            # Load image for face analysis
            image = cv2.imread(image_path)
            if image is None:
                return False, {"error": "Failed to load image for face detection"}
            
            # Detect faces
            faces = self.face_analyzer.get(image)
            
            if not faces:
                return False, {"error": "No face detected"}
            
            # Get primary face (largest face)
            primary_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
            
            face_info = {
                'face_count': len(faces),
                'primary_face_size': primary_face.bbox[2] * primary_face.bbox[3],
                'primary_face_bbox': primary_face.bbox.tolist(),
                'face_landmarks': primary_face.kps.tolist() if hasattr(primary_face, 'kps') else None
            }
            
            # Check if face is too small
            if face_info['primary_face_size'] < self.min_face_size * self.min_face_size:
                return False, {"error": f"Face too small: {face_info['primary_face_size']} pixels"}
            
            return True, face_info
            
        except Exception as e:
            logger.error(f"Face detection failed for {image_path}: {e}")
            return False, {"error": f"Face detection error: {str(e)}"}
    
    def _generate_validation_summary(self, validation_results: Dict) -> Dict:
        """Generate validation summary"""
        valid_count = len(validation_results['valid_images'])
        invalid_count = len(validation_results['invalid_images'])
        total_count = valid_count + invalid_count
        
        summary = {
            'total_images': total_count,
            'valid_images': valid_count,
            'invalid_images': invalid_count,
            'success_rate': (valid_count / total_count * 100) if total_count > 0 else 0,
            'meets_requirements': valid_count >= self.min_valid_images,
            'min_required': self.min_valid_images
        }
        
        return summary
    
    def get_validation_report(self, validation_results: Dict) -> str:
        """Generate a human-readable validation report"""
        summary = validation_results['summary']
        
        report = f"""
📊 Image Validation Report
========================

📁 Total Images: {summary['total_images']}
✅ Valid Images: {summary['valid_images']}
❌ Invalid Images: {summary['invalid_images']}
📈 Success Rate: {summary['success_rate']:.1f}%

🎯 Requirements: {summary['min_required']} valid images minimum
{'✅' if summary['meets_requirements'] else '❌'} Requirements Met: {summary['meets_requirements']}

"""
        
        if validation_results['errors']:
            report += "🚨 Errors:\n"
            for error in validation_results['errors']:
                report += f"  • {error}\n"
        
        if validation_results['warnings']:
            report += "\n⚠️ Warnings:\n"
            for warning in validation_results['warnings']:
                report += f"  • {warning}\n"
        
        if validation_results['invalid_images']:
            report += "\n❌ Invalid Images:\n"
            for invalid in validation_results['invalid_images'][:5]:  # Show first 5
                report += f"  • {Path(invalid['path']).name}: {', '.join(invalid['errors'])}\n"
            if len(validation_results['invalid_images']) > 5:
                report += f"  ... and {len(validation_results['invalid_images']) - 5} more\n"
        
        return report
    
    def cleanup(self):
        """Clean up resources"""
        if self.face_analyzer:
            del self.face_analyzer
            self.face_analyzer = None 