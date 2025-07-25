"""
Pose and Lighting Analyzer for Avatar Generation
Detects head pose, facial orientation, and lighting direction from selfie sets
"""

import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class PoseLightingAnalyzer:
    """Analyzes head pose and lighting from selfie collections"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = None
        self._initialize_face_mesh()
    
    def _initialize_face_mesh(self):
        """Initialize MediaPipe face mesh for pose detection"""
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            logger.info("Face mesh initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize face mesh: {e}")
            self.face_mesh = None
    
    def analyze_selfie_set(self, image_paths: List[str]) -> Dict:
        """Analyze a set of selfies for pose and lighting patterns"""
        if not self.face_mesh:
            logger.error("Face mesh not initialized")
            return {}
        
        logger.info(f"Analyzing {len(image_paths)} selfies for pose and lighting")
        
        pose_data = []
        lighting_data = []
        
        for img_path in image_paths:
            try:
                # Load and analyze image
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Analyze pose
                pose_info = self._analyze_head_pose(image_rgb)
                if pose_info:
                    pose_data.append(pose_info)
                
                # Analyze lighting
                lighting_info = self._analyze_lighting(image_rgb)
                if lighting_info:
                    lighting_data.append(lighting_info)
                    
            except Exception as e:
                logger.warning(f"Failed to analyze {img_path}: {e}")
                continue
        
        # Aggregate results
        return self._aggregate_analysis(pose_data, lighting_data)
    
    def _analyze_head_pose(self, image: np.ndarray) -> Optional[Dict]:
        """Analyze head pose using facial landmarks"""
        try:
            results = self.face_mesh.process(image)
            
            if not results.multi_face_landmarks:
                return None
            
            landmarks = results.multi_face_landmarks[0]
            
            # Get key facial landmarks for pose estimation
            h, w = image.shape[:2]
            
            # Convert landmarks to pixel coordinates
            landmark_points = []
            for landmark in landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmark_points.append([x, y, landmark.z])
            
            # Calculate head pose using facial landmarks
            pose_angles = self._calculate_head_pose(landmark_points, w, h)
            
            # Determine head orientation
            orientation = self._determine_head_orientation(pose_angles)
            
            return {
                'yaw': pose_angles['yaw'],
                'pitch': pose_angles['pitch'],
                'roll': pose_angles['roll'],
                'orientation': orientation,
                'confidence': self._calculate_pose_confidence(landmark_points)
            }
            
        except Exception as e:
            logger.error(f"Head pose analysis failed: {e}")
            return None
    
    def _calculate_head_pose(self, landmarks: List[List], width: int, height: int) -> Dict:
        """Calculate head pose angles from facial landmarks"""
        try:
            # Use key facial landmarks for pose estimation
            # Eyes, nose, mouth corners
            left_eye = np.mean([landmarks[33], landmarks[7], landmarks[163], landmarks[144], landmarks[145], landmarks[153], landmarks[154], landmarks[155], landmarks[133], landmarks[173], landmarks[157], landmarks[158], landmarks[159], landmarks[160], landmarks[161], landmarks[246]], axis=0)
            right_eye = np.mean([landmarks[362], landmarks[382], landmarks[381], landmarks[380], landmarks[374], landmarks[373], landmarks[390], landmarks[249], landmarks[263], landmarks[466], landmarks[388], landmarks[387], landmarks[386], landmarks[385], landmarks[384], landmarks[398]], axis=0)
            nose_tip = landmarks[1]
            left_mouth = landmarks[61]
            right_mouth = landmarks[291]
            
            # Calculate face center
            face_center = np.mean([left_eye, right_eye, nose_tip], axis=0)
            
            # Calculate pose angles
            # Yaw (left-right rotation)
            eye_vector = right_eye - left_eye
            yaw = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi
            
            # Pitch (up-down rotation)
            eye_center = (left_eye + right_eye) / 2
            pitch_vector = nose_tip - eye_center
            pitch = np.arctan2(pitch_vector[1], pitch_vector[2]) * 180 / np.pi
            
            # Roll (tilt)
            mouth_vector = right_mouth - left_mouth
            roll = np.arctan2(mouth_vector[1], mouth_vector[0]) * 180 / np.pi
            
            return {
                'yaw': float(yaw),
                'pitch': float(pitch),
                'roll': float(roll),
                'face_center': face_center.tolist()
            }
            
        except Exception as e:
            logger.error(f"Pose calculation failed: {e}")
            return {'yaw': 0, 'pitch': 0, 'roll': 0}
    
    def _determine_head_orientation(self, pose_angles: Dict) -> str:
        """Determine head orientation category"""
        yaw = abs(pose_angles['yaw'])
        pitch = abs(pose_angles['pitch'])
        roll = abs(pose_angles['roll'])
        
        # Classify orientation
        if yaw < 10 and pitch < 10 and roll < 5:
            return 'front-facing'
        elif yaw > 30:
            return 'side-facing'
        elif pitch > 20:
            return 'tilted'
        elif roll > 10:
            return 'rotated'
        else:
            return 'slightly-angled'
    
    def _analyze_lighting(self, image: np.ndarray) -> Optional[Dict]:
        """Analyze lighting direction and intensity"""
        try:
            # Convert to grayscale for lighting analysis
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect face region for lighting analysis
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None
            
            # Use the largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_region = gray[y:y+h, x:x+w]
            
            # Analyze lighting patterns
            lighting_info = self._analyze_face_lighting(face_region)
            
            return {
                'direction': lighting_info['direction'],
                'intensity': lighting_info['intensity'],
                'contrast': lighting_info['contrast'],
                'softness': lighting_info['softness']
            }
            
        except Exception as e:
            logger.error(f"Lighting analysis failed: {e}")
            return None
    
    def _analyze_face_lighting(self, face_region: np.ndarray) -> Dict:
        """Analyze lighting characteristics in face region"""
        try:
            # Calculate lighting direction using gradient analysis
            grad_x = cv2.Sobel(face_region, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(face_region, cv2.CV_64F, 0, 1, ksize=3)
            
            # Average gradients to determine lighting direction
            avg_grad_x = np.mean(grad_x)
            avg_grad_y = np.mean(grad_y)
            
            # Calculate lighting direction angle
            lighting_angle = np.arctan2(avg_grad_y, avg_grad_x) * 180 / np.pi
            
            # Determine lighting direction category
            direction = self._classify_lighting_direction(lighting_angle)
            
            # Calculate lighting intensity and contrast
            intensity = np.mean(face_region)
            contrast = np.std(face_region)
            
            # Calculate softness (lower values = softer lighting)
            laplacian_var = cv2.Laplacian(face_region, cv2.CV_64F).var()
            softness = 1.0 / (1.0 + laplacian_var / 1000.0)  # Normalize softness
            
            return {
                'direction': direction,
                'angle': float(lighting_angle),
                'intensity': float(intensity),
                'contrast': float(contrast),
                'softness': float(softness)
            }
            
        except Exception as e:
            logger.error(f"Face lighting analysis failed: {e}")
            return {
                'direction': 'front',
                'angle': 0.0,
                'intensity': 128.0,
                'contrast': 50.0,
                'softness': 0.5
            }
    
    def _classify_lighting_direction(self, angle: float) -> str:
        """Classify lighting direction based on angle"""
        # Normalize angle to 0-360
        angle = (angle + 360) % 360
        
        if 315 <= angle or angle < 45:
            return 'front'
        elif 45 <= angle < 135:
            return 'left'
        elif 135 <= angle < 225:
            return 'back'
        elif 225 <= angle < 315:
            return 'right'
        else:
            return 'front'
    
    def _calculate_pose_confidence(self, landmarks: List[List]) -> float:
        """Calculate confidence score for pose detection"""
        try:
            # Simple confidence based on landmark visibility
            visible_landmarks = sum(1 for lm in landmarks if lm[2] > 0)
            confidence = visible_landmarks / len(landmarks)
            return float(confidence)
        except:
            return 0.5
    
    def _aggregate_analysis(self, pose_data: List[Dict], lighting_data: List[Dict]) -> Dict:
        """Aggregate analysis results from multiple images"""
        if not pose_data and not lighting_data:
            return {}
        
        # Aggregate pose data
        pose_summary = {}
        if pose_data:
            pose_summary = {
                'avg_yaw': np.mean([p['yaw'] for p in pose_data]),
                'avg_pitch': np.mean([p['pitch'] for p in pose_data]),
                'avg_roll': np.mean([p['roll'] for p in pose_data]),
                'dominant_orientation': self._get_dominant_orientation([p['orientation'] for p in pose_data]),
                'pose_confidence': np.mean([p['confidence'] for p in pose_data])
            }
        
        # Aggregate lighting data
        lighting_summary = {}
        if lighting_data:
            lighting_summary = {
                'dominant_direction': self._get_dominant_lighting([l['direction'] for l in lighting_data]),
                'avg_intensity': np.mean([l['intensity'] for l in lighting_data]),
                'avg_contrast': np.mean([l['contrast'] for l in lighting_data]),
                'avg_softness': np.mean([l['softness'] for l in lighting_data])
            }
        
        return {
            'pose': pose_summary,
            'lighting': lighting_summary,
            'sample_count': len(pose_data)
        }
    
    def _get_dominant_orientation(self, orientations: List[str]) -> str:
        """Get the most common head orientation"""
        from collections import Counter
        counter = Counter(orientations)
        return counter.most_common(1)[0][0]
    
    def _get_dominant_lighting(self, directions: List[str]) -> str:
        """Get the most common lighting direction"""
        from collections import Counter
        counter = Counter(directions)
        return counter.most_common(1)[0][0]
    
    def generate_controlnet_prompt(self, analysis: Dict) -> Dict:
        """Generate ControlNet conditioning based on analysis"""
        if not analysis:
            return {}
        
        pose = analysis.get('pose', {})
        lighting = analysis.get('lighting', {})
        
        # Generate pose-specific prompts
        orientation = pose.get('dominant_orientation', 'front-facing')
        pose_prompt = self._get_pose_prompt(orientation, pose)
        
        # Generate lighting-specific prompts
        lighting_direction = lighting.get('dominant_direction', 'front')
        lighting_prompt = self._get_lighting_prompt(lighting_direction, lighting)
        
        # Generate ControlNet parameters
        controlnet_params = self._get_controlnet_params(orientation, lighting_direction)
        
        return {
            'pose_prompt': pose_prompt,
            'lighting_prompt': lighting_prompt,
            'controlnet_params': controlnet_params,
            'orientation': orientation,
            'lighting_direction': lighting_direction
        }
    
    def _get_pose_prompt(self, orientation: str, pose_data: Dict) -> str:
        """Generate pose-specific prompt"""
        prompts = {
            'front-facing': 'front-facing portrait, looking directly at camera',
            'side-facing': 'profile portrait, head turned to the side',
            'tilted': 'head tilted portrait, looking up or down',
            'rotated': 'rotated head portrait, slight head turn',
            'slightly-angled': 'slightly angled portrait, natural head position'
        }
        return prompts.get(orientation, 'front-facing portrait')
    
    def _get_lighting_prompt(self, direction: str, lighting_data: Dict) -> str:
        """Generate lighting-specific prompt"""
        intensity = lighting_data.get('avg_intensity', 128)
        softness = lighting_data.get('avg_softness', 0.5)
        
        # Determine lighting quality
        if intensity > 150:
            lighting_quality = 'bright'
        elif intensity < 100:
            lighting_quality = 'dim'
        else:
            lighting_quality = 'balanced'
        
        if softness > 0.7:
            lighting_type = 'soft'
        elif softness < 0.3:
            lighting_type = 'harsh'
        else:
            lighting_type = 'natural'
        
        direction_prompts = {
            'front': f'{lighting_type} front lighting',
            'left': f'{lighting_type} left side lighting',
            'right': f'{lighting_type} right side lighting',
            'back': f'{lighting_type} back lighting'
        }
        
        return f"{direction_prompts.get(direction, 'natural lighting')}, {lighting_quality} illumination"
    
    def _get_controlnet_params(self, orientation: str, lighting_direction: str) -> Dict:
        """Generate ControlNet parameters for pose and lighting"""
        # ControlNet strength based on orientation
        pose_strength = {
            'front-facing': 0.8,
            'side-facing': 0.9,
            'tilted': 0.85,
            'rotated': 0.8,
            'slightly-angled': 0.75
        }.get(orientation, 0.8)
        
        # Lighting strength based on direction
        lighting_strength = {
            'front': 0.6,
            'left': 0.7,
            'right': 0.7,
            'back': 0.8
        }.get(lighting_direction, 0.6)
        
        return {
            'pose_strength': pose_strength,
            'lighting_strength': lighting_strength,
            'start_percent': 0.0,
            'end_percent': 1.0
        } 