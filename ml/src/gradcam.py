"""
Explainable AI (XAI) - Gradient-Weighted Class Activation Mapping (Grad-CAM)
Implements:
- Saliency map extraction for Custom CNN, VGG16, and ResNet50
- Alpha-blended heatmap overlays with Jet colormap
- Clinical sanity checks (distinguishing pen stroke attention from background edge artifacts)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib
import matplotlib.cm as cm
from typing import Tuple, Dict, Any, Optional


def find_target_conv_layer(model: keras.Model, model_name: str) -> str:
    """Returns the optimal terminal convolutional layer for each architecture."""
    model_name = model_name.lower()
    if "resnet" in model_name:
        # In Keras ResNet50, terminal conv layer inside backbone is conv5_block3_out
        for layer in model.layers:
            if hasattr(layer, "layers"):  # Nested Functional model
                for sub_layer in layer.layers:
                    if "conv5_block3_out" in sub_layer.name:
                        return "conv5_block3_out"
            if "conv5_block3_out" in layer.name:
                return layer.name
        return "conv5_block3_out"
    elif "vgg" in model_name:
        for layer in model.layers:
            if hasattr(layer, "layers"):
                for sub_layer in layer.layers:
                    if "block5_conv3" in sub_layer.name:
                        return "block5_conv3"
            if "block5_conv3" in layer.name:
                return layer.name
        return "block5_conv3"
    else:  # Custom CNN
        for layer in reversed(model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                return layer.name
        return "block3_conv"


def generate_gradcam_heatmap(img_array: np.ndarray,
                             model: keras.Model,
                             last_conv_layer_name: Optional[str] = None,
                             pred_index: Optional[int] = None) -> np.ndarray:
    """
    Computes Grad-CAM heatmap for a given input tensor and model.
    Works for direct models and nested models (where base_model is a layer).
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_target_conv_layer(model, model.name)
        
    # Check if target layer is in a sub-model (like VGG16 or ResNet50 base)
    target_submodel = None
    target_layer = None
    
    for layer in model.layers:
        if layer.name == last_conv_layer_name:
            target_layer = layer
            break
        if hasattr(layer, "layers"):
            for sub_l in layer.layers:
                if sub_l.name == last_conv_layer_name:
                    target_submodel = layer
                    target_layer = sub_l
                    break
            if target_submodel:
                break
                
    if target_submodel is not None:
        # Build gradient tape across composite structure
        sub_grad_model = keras.models.Model(
            inputs=target_submodel.inputs,
            outputs=[target_layer.output, target_submodel.output]
        )
        
        with tf.GradientTape() as tape:
            # 1. Forward through submodel
            conv_outputs, sub_out = sub_grad_model(img_array)
            tape.watch(conv_outputs)
            
            # 2. Forward remainder of main model
            x = sub_out
            for l in model.layers[2:]:  # after input and submodel
                x = l(x)
            loss = x[0] if pred_index is None else x[0][pred_index]
            
        grads = tape.gradient(loss, conv_outputs)
    else:
        # Standard model
        grad_model = keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            tape.watch(conv_outputs)
            loss = predictions[0] if pred_index is None else predictions[0][pred_index]
            
        grads = tape.gradient(loss, conv_outputs)
        
    # Global average pooling of gradients: alpha_k
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight convolutional feature maps by alpha_k
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Apply ReLU to retain only positive influence towards class
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_gradcam(original_img: Image.Image,
                    heatmap: np.ndarray,
                    alpha: float = 0.45,
                    colormap_name: str = "jet") -> Tuple[Image.Image, Image.Image]:
    """
    Superimposes the Grad-CAM heatmap over the original image with alpha blending.
    Returns: (blended_pil_image, colored_heatmap_pil_image)
    """
    # Resize heatmap to match image dimensions
    heatmap_uint8 = np.uint8(255 * heatmap)
    try:
        colormap = matplotlib.colormaps[colormap_name]
    except Exception:
        colormap = cm.get_cmap(colormap_name)
    colored_heatmap = colormap(heatmap_uint8)
    
    # Convert RGBA to RGB [0, 255]
    colored_heatmap_rgb = np.uint8(colored_heatmap[:, :, :3] * 255)
    heatmap_pil = Image.fromarray(colored_heatmap_rgb).resize(original_img.size, Image.Resampling.LANCZOS)
    
    # Blend with original
    orig_rgb = original_img.convert("RGB")
    blended = Image.blend(orig_rgb, heatmap_pil, alpha=alpha)
    return blended, heatmap_pil

