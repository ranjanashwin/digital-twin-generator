"""
Face processing utilities for Digital Twin Generator
"""

import cv2
import numpy as np
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
import os
from pathlib import Path
from typing import List, Tuple, Optional
import logging

from config import PROCESSING_CONFIG

logger = logging.getLogger(__name__)

class FaceProcessor:
    """Handles face detection, alignment, and validation"""
    
    def __init__(self):
        self.app = None
        self._initialize_face_analysis()
    
    def _initialize_face_analysis(self):
        """Initialize InsightFace for face detection"""
        try:
            self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("Face analysis initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize face analysis: {e}")
            self.app = None
    
    def detect_faces(self, image_path: str) -> List[dict]:
        """Detect faces in an image"""
        if self.app is None:
            return []
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            faces = self.app.get(img)
            return faces
        except Exception as e:
            logger.error(f"Face detection failed for {image_path}: {e}")
            return []
    
    def validate_selfie(self, image_path: str) -> Tuple[bool, str]:
        """Validate if an image is a good selfie for training"""
        faces = self.detect_faces(image_path)
        
        if not faces:
            return False, "No face detected"
        
        if len(faces) > 1:
            return False, "Multiple faces detected"
        
        face = faces[0]
        
        # Check face size
        bbox = face.bbox
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        
        if face_width < PROCESSING_CONFIG["face_size_threshold"] or \
           face_height < PROCESSING_CONFIG["face_size_threshold"]:
            return False, "Face too small"
        
        # Check detection confidence
        if hasattr(face, 'det_score') and face.det_score < PROCESSING_CONFIG["face_detection_confidence"]:
            return False, "Low detection confidence"
        
        # Check if face is reasonably centered
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        face_center_x = (bbox[0] + bbox[2]) / 2
        face_center_y = (bbox[1] + bbox[3]) / 2
        
        # Face should be in the center 70% of the image
        if face_center_x < w * 0.15 or face_center_x > w * 0.85 or \
           face_center_y < h * 0.15 or face_center_y > h * 0.85:
            return False, "Face not centered"
        
        return True, "Valid selfie"
    
    def extract_face_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Extract face embedding for identity verification"""
        faces = self.detect_faces(image_path)
        
        if not faces:
            return None
        
        face = faces[0]
        if hasattr(face, 'embedding'):
            return face.embedding
        
        return None
    
    def align_face(self, image_path: str, output_path: str) -> bool:
        """Align face to standard position"""
        faces = self.detect_faces(image_path)
        
        if not faces:
            return False
        
        face = faces[0]
        
        try:
            # Get face landmarks
            landmarks = face.kps
            
            # Calculate face angle
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            
            # Calculate angle to rotate
            angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
            
            # Load and rotate image
            img = Image.open(image_path)
            img_rotated = img.rotate(angle, resample=Image.BICUBIC)
            
            # Save aligned image
            img_rotated.save(output_path, quality=95)
            return True
            
        except Exception as e:
            logger.error(f"Face alignment failed: {e}")
            return False
    
    def crop_face(self, image_path: str, output_path: str, margin: float = 0.3) -> bool:
        """Crop image to focus on face with margin"""
        faces = self.detect_faces(image_path)
        
        if not faces:
            return False
        
        face = faces[0]
        bbox = face.bbox
        
        try:
            img = Image.open(image_path)
            w, h = img.size
            
            # Calculate crop coordinates with margin
            x1 = max(0, int(bbox[0] - margin * (bbox[2] - bbox[0])))
            y1 = max(0, int(bbox[1] - margin * (bbox[3] - bbox[1])))
            x2 = min(w, int(bbox[2] + margin * (bbox[2] - bbox[0])))
            y2 = min(h, int(bbox[3] + margin * (bbox[3] - bbox[1])))
            
            # Crop image
            img_cropped = img.crop((x1, y1, x2, y2))
            img_cropped.save(output_path, quality=95)
            return True
            
        except Exception as e:
            logger.error(f"Face cropping failed: {e}")
            return False

def validate_selfie_folder(folder_path: str) -> Tuple[bool, List[str], List[str]]:
    """Validate all selfies in a folder"""
    processor = FaceProcessor()
    valid_images = []
    invalid_images = []
    
    folder = Path(folder_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    for file_path in folder.iterdir():
        if file_path.suffix.lower() in image_extensions:
            is_valid, message = processor.validate_selfie(str(file_path))
            
            if is_valid:
                valid_images.append(str(file_path))
            else:
                invalid_images.append(f"{file_path.name}: {message}")
    
    min_selfies = PROCESSING_CONFIG["min_selfies"]
    is_sufficient = len(valid_images) >= min_selfies
    
    return is_sufficient, valid_images, invalid_images

def process_selfies_for_training(selfies_folder: str, output_folder: str) -> List[str]:
    """Process selfies for IPAdapter training"""
    processor = FaceProcessor()
    processed_images = []
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    folder = Path(selfies_folder)
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    for i, file_path in enumerate(folder.iterdir()):
        if file_path.suffix.lower() in image_extensions:
            # Validate selfie
            is_valid, message = processor.validate_selfie(str(file_path))
            
            if not is_valid:
                logger.warning(f"Skipping {file_path.name}: {message}")
                continue
            
            # Process image
            output_file = output_path / f"processed_{i:03d}.png"
            
            # Align and crop face
            temp_aligned = output_path / f"temp_aligned_{i}.png"
            if processor.align_face(str(file_path), str(temp_aligned)):
                if processor.crop_face(str(temp_aligned), str(output_file)):
                    processed_images.append(str(output_file))
                
                # Clean up temp file
                temp_aligned.unlink(missing_ok=True)
    
    return processed_images

def calculate_face_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between face embeddings"""
    if embedding1 is None or embedding2 is None:
        return 0.0
    
    # Normalize embeddings
    embedding1_norm = embedding1 / np.linalg.norm(embedding1)
    embedding2_norm = embedding2 / np.linalg.norm(embedding2)
    
    # Calculate cosine similarity
    similarity = np.dot(embedding1_norm, embedding2_norm)
    return float(similarity)

def verify_same_person(selfies_folder: str, threshold: float = 0.6) -> Tuple[bool, List[str]]:
    """Verify that all selfies are of the same person"""
    processor = FaceProcessor()
    embeddings = []
    image_paths = []
    
    folder = Path(selfies_folder)
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    # Extract embeddings
    for file_path in folder.iterdir():
        if file_path.suffix.lower() in image_extensions:
            embedding = processor.extract_face_embedding(str(file_path))
            if embedding is not None:
                embeddings.append(embedding)
                image_paths.append(str(file_path))
    
    if len(embeddings) < 2:
        return False, ["Not enough valid faces for comparison"]
    
    # Compare all pairs
    reference_embedding = embeddings[0]
    mismatches = []
    
    for i, embedding in enumerate(embeddings[1:], 1):
        similarity = calculate_face_similarity(reference_embedding, embedding)
        
        if similarity < threshold:
            mismatches.append(f"Image {i+1} may be of different person (similarity: {similarity:.3f})")
    
    is_same_person = len(mismatches) == 0
    return is_same_person, mismatches 