"""
ControlNet Integration for Avatar Generation
Provides pose and depth conditioning for enhanced avatar realism
"""

import cv2
import numpy as np
from PIL import Image
import torch
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ControlNetIntegrator:
    """Integrates ControlNet conditioning for pose and depth control"""
    
    def __init__(self):
        self.pose_estimator = None
        self.depth_estimator = None
        self._initialize_estimators()
    
    def _initialize_estimators(self):
        """Initialize pose and depth estimation models"""
        try:
            # Initialize MediaPipe pose estimation
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose_estimator = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5
            )
            logger.info("Pose estimator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize pose estimator: {e}")
            self.pose_estimator = None
    
    def generate_pose_conditioning(self, image_paths: List[str], target_pose: Dict) -> Optional[np.ndarray]:
        """Generate pose conditioning image based on target pose"""
        try:
            # Use the first image as base for pose conditioning
            if not image_paths:
                return None
            
            base_image = cv2.imread(image_paths[0])
            if base_image is None:
                return None
            
            # Extract pose from base image
            pose_landmarks = self._extract_pose_landmarks(base_image)
            if not pose_landmarks:
                return None
            
            # Create pose conditioning image
            pose_conditioning = self._create_pose_conditioning_image(
                base_image, pose_landmarks, target_pose
            )
            
            return pose_conditioning
            
        except Exception as e:
            logger.error(f"Failed to generate pose conditioning: {e}")
            return None
    
    def generate_depth_conditioning(self, image_paths: List[str]) -> Optional[np.ndarray]:
        """Generate depth conditioning image"""
        try:
            # Use the first image for depth estimation
            if not image_paths:
                return None
            
            base_image = cv2.imread(image_paths[0])
            if base_image is None:
                return None
            
            # Create depth conditioning
            depth_conditioning = self._create_depth_conditioning_image(base_image)
            
            return depth_conditioning
            
        except Exception as e:
            logger.error(f"Failed to generate depth conditioning: {e}")
            return None
    
    def _extract_pose_landmarks(self, image: np.ndarray) -> Optional[List]:
        """Extract pose landmarks from image"""
        if not self.pose_estimator:
            return None
        
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = self.pose_estimator.process(image_rgb)
            
            if not results.pose_landmarks:
                return None
            
            # Extract landmarks
            landmarks = []
            h, w = image.shape[:2]
            
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = landmark.z
                visibility = landmark.visibility
                landmarks.append([x, y, z, visibility])
            
            return landmarks
            
        except Exception as e:
            logger.error(f"Pose landmark extraction failed: {e}")
            return None
    
    def _create_pose_conditioning_image(self, image: np.ndarray, landmarks: List, target_pose: Dict) -> np.ndarray:
        """Create pose conditioning image with target pose adjustments"""
        try:
            h, w = image.shape[:2]
            
            # Create blank canvas
            pose_canvas = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Draw pose skeleton
            self._draw_pose_skeleton(pose_canvas, landmarks, target_pose)
            
            # Apply pose adjustments based on target pose
            adjusted_canvas = self._apply_pose_adjustments(pose_canvas, target_pose)
            
            return adjusted_canvas
            
        except Exception as e:
            logger.error(f"Pose conditioning image creation failed: {e}")
            return np.zeros((512, 512, 3), dtype=np.uint8)
    
    def _draw_pose_skeleton(self, canvas: np.ndarray, landmarks: List, target_pose: Dict):
        """Draw pose skeleton on canvas"""
        try:
            # Define pose connections (MediaPipe pose landmarks)
            pose_connections = [
                (11, 12),  # Shoulders
                (11, 13), (13, 15),  # Left arm
                (12, 14), (14, 16),  # Right arm
                (11, 23), (12, 24),  # Torso
                (23, 24),  # Hips
                (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
                (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
            ]
            
            # Draw connections
            for connection in pose_connections:
                start_idx, end_idx = connection
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
                    end_point = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
                    
                    # Only draw if both points are visible
                    if landmarks[start_idx][3] > 0.5 and landmarks[end_idx][3] > 0.5:
                        cv2.line(canvas, start_point, end_point, (255, 255, 255), 2)
            
            # Draw key points
            for i, landmark in enumerate(landmarks):
                if landmark[3] > 0.5:  # Only draw visible landmarks
                    x, y = int(landmark[0]), int(landmark[1])
                    cv2.circle(canvas, (x, y), 3, (255, 255, 255), -1)
                    
        except Exception as e:
            logger.error(f"Pose skeleton drawing failed: {e}")
    
    def _apply_pose_adjustments(self, canvas: np.ndarray, target_pose: Dict) -> np.ndarray:
        """Apply pose adjustments based on target pose analysis"""
        try:
            # Get target orientation
            orientation = target_pose.get('orientation', 'front-facing')
            
            # Apply orientation-specific adjustments
            if orientation == 'side-facing':
                # Apply side-facing adjustments
                canvas = self._apply_side_facing_adjustments(canvas)
            elif orientation == 'tilted':
                # Apply tilted adjustments
                canvas = self._apply_tilted_adjustments(canvas)
            elif orientation == 'rotated':
                # Apply rotated adjustments
                canvas = self._apply_rotated_adjustments(canvas)
            
            return canvas
            
        except Exception as e:
            logger.error(f"Pose adjustments failed: {e}")
            return canvas
    
    def _apply_side_facing_adjustments(self, canvas: np.ndarray) -> np.ndarray:
        """Apply side-facing pose adjustments"""
        try:
            h, w = canvas.shape[:2]
            
            # Create transformation matrix for side-facing
            # This simulates a head turn to the side
            center = (w // 2, h // 2)
            
            # Apply slight rotation and scaling
            M = cv2.getRotationMatrix2D(center, 15, 1.0)  # 15 degree rotation
            adjusted = cv2.warpAffine(canvas, M, (w, h))
            
            return adjusted
            
        except Exception as e:
            logger.error(f"Side-facing adjustments failed: {e}")
            return canvas
    
    def _apply_tilted_adjustments(self, canvas: np.ndarray) -> np.ndarray:
        """Apply tilted pose adjustments"""
        try:
            h, w = canvas.shape[:2]
            
            # Create transformation matrix for tilted head
            center = (w // 2, h // 2)
            
            # Apply tilt (roll rotation)
            M = cv2.getRotationMatrix2D(center, 10, 1.0)  # 10 degree tilt
            adjusted = cv2.warpAffine(canvas, M, (w, h))
            
            return adjusted
            
        except Exception as e:
            logger.error(f"Tilted adjustments failed: {e}")
            return canvas
    
    def _apply_rotated_adjustments(self, canvas: np.ndarray) -> np.ndarray:
        """Apply rotated pose adjustments"""
        try:
            h, w = canvas.shape[:2]
            
            # Create transformation matrix for rotated head
            center = (w // 2, h // 2)
            
            # Apply slight rotation
            M = cv2.getRotationMatrix2D(center, 5, 1.0)  # 5 degree rotation
            adjusted = cv2.warpAffine(canvas, M, (w, h))
            
            return adjusted
            
        except Exception as e:
            logger.error(f"Rotated adjustments failed: {e}")
            return canvas
    
    def _create_depth_conditioning_image(self, image: np.ndarray) -> np.ndarray:
        """Create depth conditioning image using edge detection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Dilate edges to make them more prominent
            kernel = np.ones((2, 2), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Convert to 3-channel image
            depth_conditioning = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
            
            return depth_conditioning
            
        except Exception as e:
            logger.error(f"Depth conditioning creation failed: {e}")
            return np.zeros((512, 512, 3), dtype=np.uint8)
    
    def prepare_controlnet_inputs(self, image_paths: List[str], analysis: Dict) -> Dict:
        """Prepare ControlNet inputs for generation"""
        try:
            # Generate pose conditioning
            pose_conditioning = self.generate_pose_conditioning(image_paths, analysis.get('pose', {}))
            
            # Generate depth conditioning
            depth_conditioning = self.generate_depth_conditioning(image_paths)
            
            # Prepare ControlNet parameters
            controlnet_params = self._prepare_controlnet_params(analysis)
            
            return {
                'pose_conditioning': pose_conditioning,
                'depth_conditioning': depth_conditioning,
                'controlnet_params': controlnet_params
            }
            
        except Exception as e:
            logger.error(f"ControlNet input preparation failed: {e}")
            return {}
    
    def _prepare_controlnet_params(self, analysis: Dict) -> Dict:
        """Prepare ControlNet parameters based on analysis"""
        try:
            pose = analysis.get('pose', {})
            lighting = analysis.get('lighting', {})
            
            # Get orientation and lighting direction
            orientation = pose.get('dominant_orientation', 'front-facing')
            lighting_direction = lighting.get('dominant_direction', 'front')
            
            # Determine ControlNet strengths
            pose_strength = self._get_pose_strength(orientation)
            depth_strength = self._get_depth_strength(lighting_direction)
            
            return {
                'pose_strength': pose_strength,
                'depth_strength': depth_strength,
                'start_percent': 0.0,
                'end_percent': 1.0,
                'guidance_scale': 1.0
            }
            
        except Exception as e:
            logger.error(f"ControlNet parameter preparation failed: {e}")
            return {
                'pose_strength': 0.8,
                'depth_strength': 0.6,
                'start_percent': 0.0,
                'end_percent': 1.0,
                'guidance_scale': 1.0
            }
    
    def _get_pose_strength(self, orientation: str) -> float:
        """Get ControlNet pose strength based on orientation and quality mode"""
        from config import CONTROLNET_CONFIG
        
        # Base strengths
        strengths = {
            'front-facing': 0.7,
            'side-facing': 0.9,
            'tilted': 0.8,
            'rotated': 0.75,
            'slightly-angled': 0.7
        }
        
        base_strength = strengths.get(orientation, 0.7)
        
        # Apply quality mode scaling
        quality_scale = CONTROLNET_CONFIG["pose_strength"]
        return base_strength * quality_scale
    
    def _get_depth_strength(self, lighting_direction: str) -> float:
        """Get ControlNet depth strength based on lighting direction and quality mode"""
        from config import CONTROLNET_CONFIG
        
        # Base strengths
        strengths = {
            'front': 0.5,
            'left': 0.7,
            'right': 0.7,
            'back': 0.8
        }
        
        base_strength = strengths.get(lighting_direction, 0.6)
        
        # Apply quality mode scaling
        quality_scale = CONTROLNET_CONFIG["depth_strength"]
        return base_strength * quality_scale
    
    def save_conditioning_images(self, conditioning_data: Dict, output_dir: str):
        """Save conditioning images for debugging"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save pose conditioning
            if conditioning_data.get('pose_conditioning') is not None:
                pose_path = output_path / 'pose_conditioning.png'
                cv2.imwrite(str(pose_path), conditioning_data['pose_conditioning'])
                logger.info(f"Saved pose conditioning to {pose_path}")
            
            # Save depth conditioning
            if conditioning_data.get('depth_conditioning') is not None:
                depth_path = output_path / 'depth_conditioning.png'
                cv2.imwrite(str(depth_path), conditioning_data['depth_conditioning'])
                logger.info(f"Saved depth conditioning to {depth_path}")
            
            # Save parameters
            params_path = output_path / 'controlnet_params.json'
            with open(params_path, 'w') as f:
                json.dump(conditioning_data.get('controlnet_params', {}), f, indent=2)
            logger.info(f"Saved ControlNet parameters to {params_path}")
            
        except Exception as e:
            logger.error(f"Failed to save conditioning images: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.pose_estimator:
            self.pose_estimator.close() 