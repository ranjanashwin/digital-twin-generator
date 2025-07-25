#!/usr/bin/env python3
"""
Multiple Clothing Items Manager for Fashion Content Creator
Handles multiple clothing pieces, layering, and coordination
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
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)

class ClothingType(Enum):
    """Clothing type enumeration"""
    TOP = "top"
    BOTTOM = "bottom"
    DRESS = "dress"
    OUTERWEAR = "outerwear"
    ACCESSORIES = "accessories"
    SHOES = "shoes"

class ClothingLayer(Enum):
    """Clothing layer enumeration"""
    UNDERWEAR = 0
    BOTTOM = 1
    TOP = 2
    OUTERWEAR = 3
    ACCESSORIES = 4

@dataclass
class ClothingItem:
    """Represents a single clothing item"""
    id: str
    name: str
    type: ClothingType
    layer: ClothingLayer
    image_path: str
    processed_path: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class MultipleClothingManager:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clothing_items = {}
        self.outfits = {}
        self.current_outfit = None
        
        # Clothing compatibility rules
        self.compatibility_rules = {
            ClothingType.TOP: [ClothingType.BOTTOM, ClothingType.DRESS],
            ClothingType.BOTTOM: [ClothingType.TOP, ClothingType.DRESS],
            ClothingType.DRESS: [],
            ClothingType.OUTERWEAR: [ClothingType.TOP, ClothingType.DRESS],
            ClothingType.ACCESSORIES: [ClothingType.TOP, ClothingType.DRESS],
            ClothingType.SHOES: [ClothingType.BOTTOM, ClothingType.DRESS]
        }
        
        # Layer ordering for rendering
        self.layer_order = [
            ClothingLayer.UNDERWEAR,
            ClothingLayer.BOTTOM,
            ClothingLayer.TOP,
            ClothingLayer.OUTERWEAR,
            ClothingLayer.ACCESSORIES
        ]
    
    def add_clothing_item(self, image_path: str, item_type: ClothingType, 
                         layer: ClothingLayer, name: str = None) -> str:
        """Add a clothing item to the manager"""
        try:
            # Generate unique ID
            item_id = f"{item_type.value}_{len(self.clothing_items)}_{int(os.path.getmtime(image_path))}"
            
            # Create clothing item
            clothing_item = ClothingItem(
                id=item_id,
                name=name or f"{item_type.value.title()} {len(self.clothing_items) + 1}",
                type=item_type,
                layer=layer,
                image_path=image_path
            )
            
            # Process the clothing item
            processed_path = self._process_clothing_item(clothing_item)
            clothing_item.processed_path = processed_path
            
            # Add to manager
            self.clothing_items[item_id] = clothing_item
            
            logger.info(f"Added clothing item: {clothing_item.name} (ID: {item_id})")
            return item_id
            
        except Exception as e:
            logger.error(f"Failed to add clothing item: {e}")
            raise
    
    def _process_clothing_item(self, clothing_item: ClothingItem) -> str:
        """Process a clothing item for virtual try-on"""
        try:
            # Load and preprocess image
            img = cv2.imread(clothing_item.image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {clothing_item.image_path}")
            
            # Resize to standard size
            img = cv2.resize(img, (512, 512))
            
            # Extract clothing characteristics
            characteristics = self._extract_clothing_characteristics(img, clothing_item.type)
            clothing_item.metadata['characteristics'] = characteristics
            
            # Save processed image
            processed_dir = Path(clothing_item.image_path).parent / "processed"
            processed_dir.mkdir(exist_ok=True)
            
            processed_path = processed_dir / f"processed_{clothing_item.id}.png"
            cv2.imwrite(str(processed_path), img)
            
            return str(processed_path)
            
        except Exception as e:
            logger.error(f"Failed to process clothing item: {e}")
            raise
    
    def _extract_clothing_characteristics(self, img: np.ndarray, item_type: ClothingType) -> Dict[str, Any]:
        """Extract characteristics from clothing image"""
        characteristics = {
            'type': item_type.value,
            'color_dominant': self._extract_dominant_color(img),
            'texture_type': self._classify_texture(img),
            'fit_type': self._classify_fit(img),
            'style_category': self._classify_style(img, item_type)
        }
        
        return characteristics
    
    def _extract_dominant_color(self, img: np.ndarray) -> str:
        """Extract dominant color from clothing image"""
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Reshape for color analysis
        pixels = img_rgb.reshape(-1, 3)
        
        # Calculate mean color
        mean_color = np.mean(pixels, axis=0)
        
        # Classify color
        r, g, b = mean_color
        
        if r > g and r > b:
            return 'red'
        elif g > r and g > b:
            return 'green'
        elif b > r and b > g:
            return 'blue'
        elif r > 200 and g > 200 and b > 200:
            return 'white'
        elif r < 50 and g < 50 and b < 50:
            return 'black'
        else:
            return 'neutral'
    
    def _classify_texture(self, img: np.ndarray) -> str:
        """Classify texture type"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture features
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var > 100:
            return 'textured'
        elif laplacian_var > 50:
            return 'slightly_textured'
        else:
            return 'smooth'
    
    def _classify_fit(self, img: np.ndarray) -> str:
        """Classify fit type based on image analysis"""
        # Simplified classification
        # In production, this would use more sophisticated analysis
        
        # Analyze image edges to determine fit
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density > 0.1:
            return 'fitted'
        elif edge_density > 0.05:
            return 'semi_fitted'
        else:
            return 'loose'
    
    def _classify_style(self, img: np.ndarray, item_type: ClothingType) -> str:
        """Classify style category"""
        # Simplified style classification
        # In production, this would use ML models
        
        color = self._extract_dominant_color(img)
        texture = self._classify_texture(img)
        fit = self._classify_fit(img)
        
        # Determine style based on characteristics
        if color in ['black', 'white'] and texture == 'smooth':
            return 'minimalist'
        elif texture == 'textured':
            return 'casual'
        elif fit == 'fitted':
            return 'formal'
        else:
            return 'casual'
    
    def create_outfit(self, name: str, item_ids: List[str]) -> str:
        """Create an outfit from multiple clothing items"""
        try:
            # Validate outfit compatibility
            if not self._validate_outfit_compatibility(item_ids):
                raise ValueError("Incompatible clothing items in outfit")
            
            # Generate outfit ID
            outfit_id = f"outfit_{len(self.outfits)}_{int(time.time())}"
            
            # Create outfit
            outfit = {
                'id': outfit_id,
                'name': name,
                'item_ids': item_ids,
                'layers': self._organize_layers(item_ids),
                'metadata': self._generate_outfit_metadata(item_ids)
            }
            
            self.outfits[outfit_id] = outfit
            logger.info(f"Created outfit: {name} (ID: {outfit_id})")
            
            return outfit_id
            
        except Exception as e:
            logger.error(f"Failed to create outfit: {e}")
            raise
    
    def _validate_outfit_compatibility(self, item_ids: List[str]) -> bool:
        """Validate that clothing items are compatible"""
        try:
            items = [self.clothing_items[item_id] for item_id in item_ids]
            
            # Check for conflicts
            for i, item1 in enumerate(items):
                for j, item2 in enumerate(items):
                    if i != j:
                        # Check if items are compatible
                        if not self._are_items_compatible(item1, item2):
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate outfit compatibility: {e}")
            return False
    
    def _are_items_compatible(self, item1: ClothingItem, item2: ClothingItem) -> bool:
        """Check if two clothing items are compatible"""
        # Check compatibility rules
        compatible_types = self.compatibility_rules.get(item1.type, [])
        
        if item2.type in compatible_types:
            return True
        
        # Check layer conflicts
        if item1.layer == item2.layer and item1.type != item2.type:
            return False
        
        return True
    
    def _organize_layers(self, item_ids: List[str]) -> Dict[ClothingLayer, List[str]]:
        """Organize clothing items by layer"""
        layers = {layer: [] for layer in ClothingLayer}
        
        for item_id in item_ids:
            item = self.clothing_items[item_id]
            layers[item.layer].append(item_id)
        
        return layers
    
    def _generate_outfit_metadata(self, item_ids: List[str]) -> Dict[str, Any]:
        """Generate metadata for the outfit"""
        items = [self.clothing_items[item_id] for item_id in item_ids]
        
        metadata = {
            'total_items': len(items),
            'item_types': [item.type.value for item in items],
            'color_scheme': self._analyze_color_scheme(items),
            'style_category': self._determine_outfit_style(items),
            'seasonality': self._determine_seasonality(items),
            'formality_level': self._determine_formality(items)
        }
        
        return metadata
    
    def _analyze_color_scheme(self, items: List[ClothingItem]) -> str:
        """Analyze the color scheme of an outfit"""
        colors = [item.metadata.get('characteristics', {}).get('color_dominant', 'neutral') 
                 for item in items]
        
        # Determine color scheme
        if all(color == 'black' for color in colors):
            return 'monochrome_black'
        elif all(color == 'white' for color in colors):
            return 'monochrome_white'
        elif len(set(colors)) == 1:
            return 'monochrome'
        elif len(set(colors)) <= 2:
            return 'complementary'
        else:
            return 'mixed'
    
    def _determine_outfit_style(self, items: List[ClothingItem]) -> str:
        """Determine the overall style of an outfit"""
        styles = [item.metadata.get('characteristics', {}).get('style_category', 'casual') 
                 for item in items]
        
        if all(style == 'formal' for style in styles):
            return 'formal'
        elif all(style == 'minimalist' for style in styles):
            return 'minimalist'
        elif 'casual' in styles:
            return 'casual'
        else:
            return 'mixed'
    
    def _determine_seasonality(self, items: List[ClothingItem]) -> str:
        """Determine the seasonality of an outfit"""
        # Simplified seasonality detection
        # In production, this would use more sophisticated analysis
        
        has_outerwear = any(item.type == ClothingType.OUTERWEAR for item in items)
        has_light_items = any(item.type in [ClothingType.TOP, ClothingType.DRESS] for item in items)
        
        if has_outerwear:
            return 'winter'
        elif has_light_items:
            return 'summer'
        else:
            return 'all_season'
    
    def _determine_formality(self, items: List[ClothingItem]) -> str:
        """Determine the formality level of an outfit"""
        styles = [item.metadata.get('characteristics', {}).get('style_category', 'casual') 
                 for item in items]
        
        if all(style == 'formal' for style in styles):
            return 'formal'
        elif any(style == 'formal' for style in styles):
            return 'semi_formal'
        else:
            return 'casual'
    
    def apply_outfit_to_avatar(self, outfit_id: str, avatar_path: str, 
                              pose_data: Dict[str, Any]) -> str:
        """Apply an outfit to an avatar"""
        try:
            outfit = self.outfits.get(outfit_id)
            if not outfit:
                raise ValueError(f"Outfit not found: {outfit_id}")
            
            # Get layered clothing items
            layers = outfit['layers']
            
            # Apply clothing items in layer order
            result_image = cv2.imread(avatar_path)
            if result_image is None:
                raise ValueError(f"Failed to load avatar: {avatar_path}")
            
            # Apply each layer
            for layer in self.layer_order:
                if layer in layers and layers[layer]:
                    for item_id in layers[layer]:
                        result_image = self._apply_clothing_layer(
                            result_image, item_id, pose_data
                        )
            
            # Save result
            output_path = self._save_outfit_result(avatar_path, outfit_id, result_image)
            
            logger.info(f"Applied outfit {outfit_id} to avatar")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to apply outfit to avatar: {e}")
            raise
    
    def _apply_clothing_layer(self, base_image: np.ndarray, item_id: str, 
                             pose_data: Dict[str, Any]) -> np.ndarray:
        """Apply a single clothing layer to the base image"""
        try:
            clothing_item = self.clothing_items[item_id]
            
            # Load clothing image
            clothing_img = cv2.imread(clothing_item.processed_path or clothing_item.image_path)
            if clothing_img is None:
                return base_image
            
            # Resize clothing to match base image
            clothing_img = cv2.resize(clothing_img, (base_image.shape[1], base_image.shape[0]))
            
            # Apply clothing based on type
            if clothing_item.type == ClothingType.TOP:
                result = self._apply_top_clothing(base_image, clothing_img, pose_data)
            elif clothing_item.type == ClothingType.BOTTOM:
                result = self._apply_bottom_clothing(base_image, clothing_img, pose_data)
            elif clothing_item.type == ClothingType.DRESS:
                result = self._apply_dress_clothing(base_image, clothing_img, pose_data)
            elif clothing_item.type == ClothingType.OUTERWEAR:
                result = self._apply_outerwear_clothing(base_image, clothing_img, pose_data)
            else:
                result = self._apply_accessory_clothing(base_image, clothing_img, pose_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to apply clothing layer: {e}")
            return base_image
    
    def _apply_top_clothing(self, base_image: np.ndarray, clothing_img: np.ndarray, 
                           pose_data: Dict[str, Any]) -> np.ndarray:
        """Apply top clothing (shirt, blouse, etc.)"""
        # Simplified top application
        # In production, this would use sophisticated body part segmentation
        
        # Create mask for upper body region
        h, w = base_image.shape[:2]
        upper_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Define upper body region (simplified)
        upper_y_start = int(h * 0.2)
        upper_y_end = int(h * 0.7)
        upper_mask[upper_y_start:upper_y_end, :] = 255
        
        # Apply clothing with mask
        result = base_image.copy()
        result[upper_mask > 0] = cv2.addWeighted(
            result[upper_mask > 0], 0.3,
            clothing_img[upper_mask > 0], 0.7, 0
        )
        
        return result
    
    def _apply_bottom_clothing(self, base_image: np.ndarray, clothing_img: np.ndarray, 
                              pose_data: Dict[str, Any]) -> np.ndarray:
        """Apply bottom clothing (pants, skirt, etc.)"""
        # Simplified bottom application
        
        h, w = base_image.shape[:2]
        lower_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Define lower body region
        lower_y_start = int(h * 0.5)
        lower_y_end = int(h * 0.9)
        lower_mask[lower_y_start:lower_y_end, :] = 255
        
        # Apply clothing with mask
        result = base_image.copy()
        result[lower_mask > 0] = cv2.addWeighted(
            result[lower_mask > 0], 0.3,
            clothing_img[lower_mask > 0], 0.7, 0
        )
        
        return result
    
    def _apply_dress_clothing(self, base_image: np.ndarray, clothing_img: np.ndarray, 
                             pose_data: Dict[str, Any]) -> np.ndarray:
        """Apply dress clothing"""
        # Simplified dress application
        
        h, w = base_image.shape[:2]
        dress_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Define dress region (covers most of body)
        dress_y_start = int(h * 0.2)
        dress_y_end = int(h * 0.9)
        dress_mask[dress_y_start:dress_y_end, :] = 255
        
        # Apply clothing with mask
        result = base_image.copy()
        result[dress_mask > 0] = cv2.addWeighted(
            result[dress_mask > 0], 0.3,
            clothing_img[dress_mask > 0], 0.7, 0
        )
        
        return result
    
    def _apply_outerwear_clothing(self, base_image: np.ndarray, clothing_img: np.ndarray, 
                                 pose_data: Dict[str, Any]) -> np.ndarray:
        """Apply outerwear clothing (jacket, coat, etc.)"""
        # Simplified outerwear application
        
        h, w = base_image.shape[:2]
        outerwear_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Define outerwear region (covers most of body)
        outerwear_y_start = int(h * 0.1)
        outerwear_y_end = int(h * 0.8)
        outerwear_mask[outerwear_y_start:outerwear_y_end, :] = 255
        
        # Apply clothing with mask
        result = base_image.copy()
        result[outerwear_mask > 0] = cv2.addWeighted(
            result[outerwear_mask > 0], 0.4,
            clothing_img[outerwear_mask > 0], 0.6, 0
        )
        
        return result
    
    def _apply_accessory_clothing(self, base_image: np.ndarray, clothing_img: np.ndarray, 
                                 pose_data: Dict[str, Any]) -> np.ndarray:
        """Apply accessory clothing (hats, scarves, etc.)"""
        # Simplified accessory application
        
        h, w = base_image.shape[:2]
        accessory_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Define accessory region (varies by accessory type)
        accessory_y_start = int(h * 0.05)
        accessory_y_end = int(h * 0.3)
        accessory_mask[accessory_y_start:accessory_y_end, :] = 255
        
        # Apply clothing with mask
        result = base_image.copy()
        result[accessory_mask > 0] = cv2.addWeighted(
            result[accessory_mask > 0], 0.5,
            clothing_img[accessory_mask > 0], 0.5, 0
        )
        
        return result
    
    def _save_outfit_result(self, avatar_path: str, outfit_id: str, 
                           result_image: np.ndarray) -> str:
        """Save the outfit application result"""
        avatar_path_obj = Path(avatar_path)
        output_dir = avatar_path_obj.parent / "outfits"
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / f"outfit_{outfit_id}_{avatar_path_obj.name}"
        cv2.imwrite(str(output_path), result_image)
        
        return str(output_path)
    
    def get_outfit_suggestions(self, outfit_id: str) -> List[str]:
        """Get suggestions for improving an outfit"""
        outfit = self.outfits.get(outfit_id)
        if not outfit:
            return []
        
        suggestions = []
        metadata = outfit.get('metadata', {})
        
        # Color scheme suggestions
        color_scheme = metadata.get('color_scheme', 'mixed')
        if color_scheme == 'mixed':
            suggestions.append("Consider a more cohesive color scheme")
        
        # Style suggestions
        style = metadata.get('style_category', 'casual')
        if style == 'mixed':
            suggestions.append("Consider matching style categories for consistency")
        
        # Formality suggestions
        formality = metadata.get('formality_level', 'casual')
        if formality == 'mixed':
            suggestions.append("Consider matching formality levels")
        
        return suggestions
    
    def get_compatible_items(self, item_id: str) -> List[str]:
        """Get list of compatible clothing items"""
        if item_id not in self.clothing_items:
            return []
        
        current_item = self.clothing_items[item_id]
        compatible_items = []
        
        for other_id, other_item in self.clothing_items.items():
            if other_id != item_id and self._are_items_compatible(current_item, other_item):
                compatible_items.append(other_id)
        
        return compatible_items
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Clear clothing items
            self.clothing_items.clear()
            self.outfits.clear()
            self.current_outfit = None
            
            logger.info("Multiple clothing manager cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup multiple clothing manager: {e}")
    
    def get_outfit_summary(self, outfit_id: str) -> Dict[str, Any]:
        """Get detailed summary of an outfit"""
        outfit = self.outfits.get(outfit_id)
        if not outfit:
            return {}
        
        items = [self.clothing_items[item_id] for item_id in outfit['item_ids']]
        
        summary = {
            'id': outfit_id,
            'name': outfit['name'],
            'items': [{'id': item.id, 'name': item.name, 'type': item.type.value} for item in items],
            'metadata': outfit['metadata'],
            'total_items': len(items),
            'layers': {layer.value: len(item_ids) for layer, item_ids in outfit['layers'].items()}
        }
        
        return summary 