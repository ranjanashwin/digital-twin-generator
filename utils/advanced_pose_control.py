#!/usr/bin/env python3
"""
Advanced Pose Control for Fashion Content Creator
Provides fine-grained pose manipulation and control
"""

import os
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import cv2
from PIL import Image
import json
import mediapipe as mp

logger = logging.getLogger(__name__)

class AdvancedPoseController:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pose_estimator = None
        self.face_mesh = None
        self.hands = None
        self._initialize_mediapipe()
        
        # Pose presets for fashion
        self.fashion_pose_presets = {
            'fashion_portrait': {
                'head_angle': 15,
                'shoulder_rotation': 10,
                'arm_position': 'natural',
                'torso_angle': 5,
                'confidence_threshold': 0.7
            },
            'street_style': {
                'head_angle': 0,
                'shoulder_rotation': 0,
                'arm_position': 'relaxed',
                'torso_angle': 0,
                'confidence_threshold': 0.6
            },
            'studio_fashion': {
                'head_angle': 20,
                'shoulder_rotation': 15,
                'arm_position': 'posed',
                'torso_angle': 10,
                'confidence_threshold': 0.8
            },
            'editorial': {
                'head_angle': 30,
                'shoulder_rotation': 25,
                'arm_position': 'dramatic',
                'torso_angle': 15,
                'confidence_threshold': 0.9
            }
        }
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe pose estimation models"""
        try:
            # Initialize pose estimation
            self.pose_estimator = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5
            )
            
            # Initialize face mesh for detailed facial pose
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Initialize hands for detailed hand pose
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            logger.info("MediaPipe pose estimation models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe models: {e}")
            self.pose_estimator = None
            self.face_mesh = None
            self.hands = None
    
    def extract_detailed_pose(self, image_path: str) -> Dict[str, Any]:
        """Extract detailed pose information including face, body, and hands"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            pose_data = {
                'image_size': (width, height),
                'body_pose': self._extract_body_pose(image_rgb),
                'face_pose': self._extract_face_pose(image_rgb),
                'hand_poses': self._extract_hand_poses(image_rgb),
                'overall_pose': {}
            }
            
            # Calculate overall pose characteristics
            pose_data['overall_pose'] = self._calculate_overall_pose(pose_data)
            
            return pose_data
            
        except Exception as e:
            logger.error(f"Failed to extract detailed pose: {e}")
            raise
    
    def _extract_body_pose(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """Extract body pose landmarks"""
        if not self.pose_estimator:
            return {}
        
        try:
            results = self.pose_estimator.process(image_rgb)
            
            if not results.pose_landmarks:
                return {}
            
            landmarks = []
            h, w = image_rgb.shape[:2]
            
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = landmark.z
                visibility = landmark.visibility
                landmarks.append([x, y, z, visibility])
            
            # Extract key pose characteristics
            pose_characteristics = self._analyze_body_pose(landmarks, (w, h))
            
            return {
                'landmarks': landmarks,
                'characteristics': pose_characteristics,
                'confidence': np.mean([lm[3] for lm in landmarks if lm[3] > 0])
            }
            
        except Exception as e:
            logger.error(f"Failed to extract body pose: {e}")
            return {}
    
    def _extract_face_pose(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """Extract detailed facial pose landmarks"""
        if not self.face_mesh:
            return {}
        
        try:
            results = self.face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                return {}
            
            face_landmarks = results.multi_face_landmarks[0]
            h, w = image_rgb.shape[:2]
            
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = landmark.z
                landmarks.append([x, y, z])
            
            # Extract facial pose characteristics
            face_characteristics = self._analyze_face_pose(landmarks, (w, h))
            
            return {
                'landmarks': landmarks,
                'characteristics': face_characteristics
            }
            
        except Exception as e:
            logger.error(f"Failed to extract face pose: {e}")
            return {}
    
    def _extract_hand_poses(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """Extract hand pose landmarks"""
        if not self.hands:
            return []
        
        try:
            results = self.hands.process(image_rgb)
            
            if not results.multi_hand_landmarks:
                return []
            
            hand_poses = []
            h, w = image_rgb.shape[:2]
            
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    z = landmark.z
                    landmarks.append([x, y, z])
                
                hand_characteristics = self._analyze_hand_pose(landmarks, (w, h))
                hand_poses.append({
                    'landmarks': landmarks,
                    'characteristics': hand_characteristics
                })
            
            return hand_poses
            
        except Exception as e:
            logger.error(f"Failed to extract hand poses: {e}")
            return []
    
    def _analyze_body_pose(self, landmarks: List[List[float]], image_size: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze body pose characteristics"""
        if len(landmarks) < 33:  # MediaPipe pose has 33 landmarks
            return {}
        
        w, h = image_size
        
        # Extract key points
        nose = landmarks[0] if landmarks[0][3] > 0.5 else None
        left_shoulder = landmarks[11] if landmarks[11][3] > 0.5 else None
        right_shoulder = landmarks[12] if landmarks[12][3] > 0.5 else None
        left_elbow = landmarks[13] if landmarks[13][3] > 0.5 else None
        right_elbow = landmarks[14] if landmarks[14][3] > 0.5 else None
        left_wrist = landmarks[15] if landmarks[15][3] > 0.5 else None
        right_wrist = landmarks[16] if landmarks[16][3] > 0.5 else None
        
        characteristics = {
            'head_angle': 0,
            'shoulder_rotation': 0,
            'arm_position': 'natural',
            'torso_angle': 0,
            'pose_confidence': 0
        }
        
        # Calculate head angle (if nose and shoulders are visible)
        if nose and left_shoulder and right_shoulder:
            # Calculate head tilt based on shoulder line vs horizontal
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            
            # Calculate angle between nose and shoulder center
            dx = nose[0] - shoulder_center_x
            dy = nose[1] - shoulder_center_y
            head_angle = np.degrees(np.arctan2(dy, dx))
            characteristics['head_angle'] = head_angle
        
        # Calculate shoulder rotation
        if left_shoulder and right_shoulder:
            shoulder_dx = right_shoulder[0] - left_shoulder[0]
            shoulder_dy = right_shoulder[1] - left_shoulder[1]
            shoulder_angle = np.degrees(np.arctan2(shoulder_dy, shoulder_dx))
            characteristics['shoulder_rotation'] = shoulder_angle
        
        # Analyze arm positions
        if left_elbow and right_elbow:
            left_arm_angle = self._calculate_arm_angle(left_shoulder, left_elbow, left_wrist)
            right_arm_angle = self._calculate_arm_angle(right_shoulder, right_elbow, right_wrist)
            
            if left_arm_angle > 90 or right_arm_angle > 90:
                characteristics['arm_position'] = 'raised'
            elif left_arm_angle < 45 or right_arm_angle < 45:
                characteristics['arm_position'] = 'lowered'
            else:
                characteristics['arm_position'] = 'natural'
        
        # Calculate overall pose confidence
        visible_landmarks = [lm for lm in landmarks if lm[3] > 0.5]
        characteristics['pose_confidence'] = len(visible_landmarks) / len(landmarks)
        
        return characteristics
    
    def _analyze_face_pose(self, landmarks: List[List[float]], image_size: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze facial pose characteristics"""
        if len(landmarks) < 468:  # MediaPipe face mesh has 468 landmarks
            return {}
        
        w, h = image_size
        
        # Extract key facial points
        nose_tip = landmarks[4]  # Nose tip
        left_eye = landmarks[33]  # Left eye corner
        right_eye = landmarks[263]  # Right eye corner
        left_ear = landmarks[234]  # Left ear
        right_ear = landmarks[454]  # Right ear
        
        characteristics = {
            'face_angle': 0,
            'eye_level': 0,
            'face_confidence': 0
        }
        
        # Calculate face angle (head rotation)
        if left_eye and right_eye:
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            eye_center_y = (left_eye[1] + right_eye[1]) / 2
            
            # Calculate angle from eye center to nose tip
            dx = nose_tip[0] - eye_center_x
            dy = nose_tip[1] - eye_center_y
            face_angle = np.degrees(np.arctan2(dy, dx))
            characteristics['face_angle'] = face_angle
        
        # Calculate eye level (head tilt)
        if left_eye and right_eye:
            eye_dx = right_eye[0] - left_eye[0]
            eye_dy = right_eye[1] - left_eye[1]
            eye_angle = np.degrees(np.arctan2(eye_dy, eye_dx))
            characteristics['eye_level'] = eye_angle
        
        # Calculate face confidence (based on landmark visibility)
        characteristics['face_confidence'] = 0.8  # Simplified for now
        
        return characteristics
    
    def _analyze_hand_pose(self, landmarks: List[List[float]], image_size: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze hand pose characteristics"""
        if len(landmarks) < 21:  # MediaPipe hands has 21 landmarks
            return {}
        
        w, h = image_size
        
        # Extract key hand points
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        characteristics = {
            'hand_gesture': 'neutral',
            'finger_positions': {},
            'hand_confidence': 0
        }
        
        # Analyze finger positions
        finger_tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
        for i, (tip, name) in enumerate(zip(finger_tips, finger_names)):
            # Calculate finger extension (distance from wrist)
            distance = np.sqrt((tip[0] - wrist[0])**2 + (tip[1] - wrist[1])**2)
            characteristics['finger_positions'][name] = {
                'extended': distance > 50,  # Threshold for extension
                'position': tip
            }
        
        # Determine hand gesture
        extended_fingers = sum(1 for pos in characteristics['finger_positions'].values() if pos['extended'])
        
        if extended_fingers == 0:
            characteristics['hand_gesture'] = 'fist'
        elif extended_fingers == 1:
            characteristics['hand_gesture'] = 'pointing'
        elif extended_fingers == 2:
            characteristics['hand_gesture'] = 'peace'
        elif extended_fingers == 5:
            characteristics['hand_gesture'] = 'open'
        else:
            characteristics['hand_gesture'] = 'partial'
        
        characteristics['hand_confidence'] = 0.7  # Simplified for now
        
        return characteristics
    
    def _calculate_arm_angle(self, shoulder: List[float], elbow: List[float], wrist: List[float]) -> float:
        """Calculate arm angle in degrees"""
        if not all([shoulder, elbow, wrist]):
            return 0
        
        # Calculate vectors
        upper_arm = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
        lower_arm = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])
        
        # Calculate angle
        dot_product = np.dot(upper_arm, lower_arm)
        upper_arm_norm = np.linalg.norm(upper_arm)
        lower_arm_norm = np.linalg.norm(lower_arm)
        
        if upper_arm_norm == 0 or lower_arm_norm == 0:
            return 0
        
        cos_angle = dot_product / (upper_arm_norm * lower_arm_norm)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def _calculate_overall_pose(self, pose_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall pose characteristics"""
        body_pose = pose_data.get('body_pose', {}).get('characteristics', {})
        face_pose = pose_data.get('face_pose', {}).get('characteristics', {})
        
        overall = {
            'pose_type': 'neutral',
            'confidence': 0,
            'fashion_ready': False,
            'suggestions': []
        }
        
        # Determine pose type
        head_angle = abs(body_pose.get('head_angle', 0))
        shoulder_rotation = abs(body_pose.get('shoulder_rotation', 0))
        arm_position = body_pose.get('arm_position', 'natural')
        
        if head_angle > 30 or shoulder_rotation > 45:
            overall['pose_type'] = 'dynamic'
        elif head_angle > 15 or shoulder_rotation > 20:
            overall['pose_type'] = 'semi_dynamic'
        else:
            overall['pose_type'] = 'neutral'
        
        # Calculate overall confidence
        body_conf = body_pose.get('pose_confidence', 0)
        face_conf = face_pose.get('face_confidence', 0)
        overall['confidence'] = (body_conf + face_conf) / 2
        
        # Determine if pose is fashion-ready
        overall['fashion_ready'] = (
            overall['confidence'] > 0.7 and
            body_pose.get('pose_confidence', 0) > 0.6
        )
        
        # Generate suggestions
        if overall['confidence'] < 0.6:
            overall['suggestions'].append('Low pose confidence - consider clearer pose')
        if head_angle > 45:
            overall['suggestions'].append('Extreme head angle - consider more neutral pose')
        if arm_position == 'raised':
            overall['suggestions'].append('Arms raised - consider more natural arm position')
        
        return overall
    
    def apply_pose_preset(self, pose_data: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
        """Apply a fashion pose preset to the extracted pose data"""
        preset = self.fashion_pose_presets.get(preset_name, self.fashion_pose_presets['fashion_portrait'])
        
        # Create modified pose data
        modified_pose = pose_data.copy()
        body_characteristics = modified_pose.get('body_pose', {}).get('characteristics', {}).copy()
        
        # Apply preset adjustments
        body_characteristics['head_angle'] = preset['head_angle']
        body_characteristics['shoulder_rotation'] = preset['shoulder_rotation']
        body_characteristics['arm_position'] = preset['arm_position']
        body_characteristics['torso_angle'] = preset['torso_angle']
        
        # Update confidence threshold
        if modified_pose.get('body_pose'):
            modified_pose['body_pose']['characteristics'] = body_characteristics
            modified_pose['body_pose']['confidence_threshold'] = preset['confidence_threshold']
        
        return modified_pose
    
    def generate_pose_conditioning(self, pose_data: Dict[str, Any], 
                                 target_pose: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Generate pose conditioning image for ControlNet"""
        try:
            # Get image size
            image_size = pose_data.get('image_size', (512, 512))
            w, h = image_size
            
            # Create blank canvas
            pose_canvas = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Draw pose skeleton
            self._draw_pose_skeleton(pose_canvas, pose_data, target_pose)
            
            # Apply pose adjustments if target pose is provided
            if target_pose:
                pose_canvas = self._apply_pose_adjustments(pose_canvas, pose_data, target_pose)
            
            return pose_canvas
            
        except Exception as e:
            logger.error(f"Failed to generate pose conditioning: {e}")
            return np.zeros((512, 512, 3), dtype=np.uint8)
    
    def _draw_pose_skeleton(self, canvas: np.ndarray, pose_data: Dict[str, Any], 
                           target_pose: Optional[Dict[str, Any]] = None):
        """Draw pose skeleton on canvas"""
        try:
            body_landmarks = pose_data.get('body_pose', {}).get('landmarks', [])
            
            if not body_landmarks:
                return
            
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
                if start_idx < len(body_landmarks) and end_idx < len(body_landmarks):
                    start_point = (int(body_landmarks[start_idx][0]), int(body_landmarks[start_idx][1]))
                    end_point = (int(body_landmarks[end_idx][0]), int(body_landmarks[end_idx][1]))
                    
                    # Only draw if both points are visible
                    if body_landmarks[start_idx][3] > 0.5 and body_landmarks[end_idx][3] > 0.5:
                        cv2.line(canvas, start_point, end_point, (255, 255, 255), 2)
            
            # Draw key points
            for i, landmark in enumerate(body_landmarks):
                if landmark[3] > 0.5:  # Only draw visible landmarks
                    x, y = int(landmark[0]), int(landmark[1])
                    cv2.circle(canvas, (x, y), 3, (255, 255, 255), -1)
                    
        except Exception as e:
            logger.error(f"Failed to draw pose skeleton: {e}")
    
    def _apply_pose_adjustments(self, canvas: np.ndarray, pose_data: Dict[str, Any], 
                               target_pose: Dict[str, Any]) -> np.ndarray:
        """Apply pose adjustments based on target pose"""
        try:
            # Get current and target characteristics
            current_body = pose_data.get('body_pose', {}).get('characteristics', {})
            target_body = target_pose.get('body_pose', {}).get('characteristics', {})
            
            # Calculate adjustments
            head_adjustment = target_body.get('head_angle', 0) - current_body.get('head_angle', 0)
            shoulder_adjustment = target_body.get('shoulder_rotation', 0) - current_body.get('shoulder_rotation', 0)
            
            # Apply adjustments to canvas
            if abs(head_adjustment) > 5 or abs(shoulder_adjustment) > 5:
                h, w = canvas.shape[:2]
                center = (w // 2, h // 2)
                
                # Create transformation matrix
                M = cv2.getRotationMatrix2D(center, head_adjustment, 1.0)
                adjusted_canvas = cv2.warpAffine(canvas, M, (w, h))
                
                return adjusted_canvas
            
            return canvas
            
        except Exception as e:
            logger.error(f"Failed to apply pose adjustments: {e}")
            return canvas
    
    def get_pose_suggestions(self, pose_data: Dict[str, Any]) -> List[str]:
        """Get pose improvement suggestions"""
        suggestions = []
        overall_pose = pose_data.get('overall_pose', {})
        
        confidence = overall_pose.get('confidence', 0)
        pose_type = overall_pose.get('pose_type', 'neutral')
        
        if confidence < 0.6:
            suggestions.append("Low pose confidence - ensure clear, well-lit image")
        
        if pose_type == 'dynamic':
            suggestions.append("Dynamic pose detected - consider more neutral pose for fashion")
        
        body_pose = pose_data.get('body_pose', {}).get('characteristics', {})
        head_angle = abs(body_pose.get('head_angle', 0))
        
        if head_angle > 30:
            suggestions.append("Extreme head angle - consider more neutral head position")
        
        return suggestions
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.pose_estimator:
                self.pose_estimator.close()
            if self.face_mesh:
                self.face_mesh.close()
            if self.hands:
                self.hands.close()
            
            logger.info("Advanced pose controller cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup advanced pose controller: {e}")
    
    def get_available_presets(self) -> List[str]:
        """Get list of available pose presets"""
        return list(self.fashion_pose_presets.keys())
    
    def get_preset_details(self, preset_name: str) -> Dict[str, Any]:
        """Get detailed information about a pose preset"""
        preset = self.fashion_pose_presets.get(preset_name, {})
        return {
            'name': preset_name,
            'description': f"Fashion {preset_name.replace('_', ' ').title()} pose",
            'parameters': preset,
            'suitable_for': self._get_preset_suitability(preset_name)
        }
    
    def _get_preset_suitability(self, preset_name: str) -> List[str]:
        """Get suitable use cases for a pose preset"""
        suitability_map = {
            'fashion_portrait': ['headshots', 'professional', 'editorial'],
            'street_style': ['casual', 'lifestyle', 'social_media'],
            'studio_fashion': ['fashion_shoots', 'commercial', 'high_end'],
            'editorial': ['magazine', 'artistic', 'dramatic']
        }
        return suitability_map.get(preset_name, ['general']) 