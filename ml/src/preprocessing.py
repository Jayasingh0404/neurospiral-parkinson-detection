"""
Preprocessing and Augmentation Pipeline for Spiral Drawing Images.
Complies with specifications in Data_Preprocessing.pdf and CNN_Model_Pretrained_models.pdf:
- Aspect-ratio preserving letterboxing with white padding (drawing-specific)
- Model-specific input normalizations (Custom CNN vs VGG16 vs ResNet50)
- Label-preserving data augmentation strictly confined to training data
"""

import numpy as np
from PIL import Image, ImageEnhance
import random
from typing import Tuple, Optional, Union


def letterbox_image(image: Image.Image, target_size: Tuple[int, int] = (224, 224), bg_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """
    Resizes an image preserving aspect ratio and pads remaining space with white background
    (appropriate for spiral drawings on white paper to prevent geometric distortion).
    """
    target_w, target_h = target_size
    img_w, img_h = image.size
    
    scale = min(target_w / img_w, target_h / img_h)
    new_w = max(1, int(img_w * scale))
    new_h = max(1, int(img_h * scale))
    
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    padded_img = Image.new("RGB", (target_w, target_h), bg_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    padded_img.paste(resized, (paste_x, paste_y))
    return padded_img


def augment_image(image: Image.Image,
                  enable_flip: bool = True,
                  max_rotation: float = 10.0,
                  zoom_range: Tuple[float, float] = (0.95, 1.05),
                  brightness_range: Tuple[float, float] = (0.9, 1.1),
                  contrast_range: Tuple[float, float] = (0.9, 1.1)) -> Image.Image:
    """
    Conservative, label-preserving augmentation designed specifically for handwriting/spiral patterns.
    - Never uses vertical flips or aggressive shear/rotations that alter motor tremor geometry.
    - Strictly applied ONLY during training.
    """
    img = image.copy()
    
    # 1. Subtle rotation within [-max_rotation, +max_rotation] degrees
    if max_rotation > 0:
        angle = random.uniform(-max_rotation, max_rotation)
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255))
        
    # 2. Subtle zoom / crop
    if zoom_range and zoom_range != (1.0, 1.0):
        scale = random.uniform(zoom_range[0], zoom_range[1])
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        if scale > 1.0:
            # Crop center
            img_zoomed = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            img = img_zoomed.crop((left, top, left + w, top + h))
        elif scale < 1.0:
            # Pad to center
            img_zoomed = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            img_padded = Image.new("RGB", (w, h), (255, 255, 255))
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            img_padded.paste(img_zoomed, (left, top))
            img = img_padded
            
    # 3. Horizontal flip
    if enable_flip and random.random() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
    # 4. Subtle brightness jitter
    if brightness_range:
        factor = random.uniform(brightness_range[0], brightness_range[1])
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)
        
    # 5. Subtle contrast jitter
    if contrast_range:
        factor = random.uniform(contrast_range[0], contrast_range[1])
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)
        
    return img


def preprocess_for_model(img_array: np.ndarray, model_type: str = "cnn") -> np.ndarray:
    """
    Applies the exact mathematically appropriate normalization for each architecture:
    - 'cnn': Scaled to [0, 1] via 1/255.0
    - 'vgg16' / 'resnet50': ImageNet standard Caffe-style zero-centering:
        RGB -> BGR, subtract ImageNet channel means: [103.939, 116.779, 123.68]
    """
    arr = img_array.astype(np.float32)
    model_type = model_type.lower()
    
    if model_type == "cnn":
        return arr / 255.0
    elif model_type in ["vgg16", "resnet50"]:
        # Convert RGB to BGR
        arr_bgr = arr[..., ::-1].copy()
        # Subtract ImageNet means (BGR format)
        arr_bgr[..., 0] -= 103.939
        arr_bgr[..., 1] -= 116.779
        arr_bgr[..., 2] -= 123.680
        return arr_bgr
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from 'cnn', 'vgg16', 'resnet50'.")


def prepare_image(image_input: Union[str, Image.Image],
                  target_size: Tuple[int, int] = (224, 224),
                  model_type: str = "cnn",
                  augment: bool = False) -> Tuple[np.ndarray, Image.Image]:
    """
    Full preprocessing pipeline from file or PIL image to standardized model input tensor:
    1. Load and convert to RGB
    2. Aspect-ratio preserving letterbox padding
    3. Optional training augmentation
    4. Model-specific normalization
    Returns: (preprocessed_array_with_batch_dim, letterboxed_pil_image)
    """
    if isinstance(image_input, str):
        with Image.open(image_input) as img:
            image = img.convert("RGB")
    else:
        image = image_input.convert("RGB")
        
    # 1. Letterbox
    standardized = letterbox_image(image, target_size=target_size)
    
    # 2. Augment if requested
    if augment:
        standardized = augment_image(standardized)
        
    # 3. Model Preprocess
    arr = np.asarray(standardized, dtype=np.float32)
    norm_arr = preprocess_for_model(arr, model_type=model_type)
    
    # Add batch dimension: (1, 224, 224, 3)
    batch_arr = np.expand_dims(norm_arr, axis=0)
    return batch_arr, standardized

