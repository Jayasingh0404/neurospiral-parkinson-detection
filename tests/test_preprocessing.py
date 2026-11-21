"""
Unit tests for Image Preprocessing and Standardizations.
"""

import numpy as np
from PIL import Image
from ml.src.preprocessing import (
    letterbox_image,
    preprocess_for_model,
    augment_image,
    prepare_image
)


def test_letterbox_preserves_dimensions_and_pads_white():
    """Verifies that non-square images are padded with pure white canvas to 224x224."""
    # Create rectangular test image (512x200) with dark line
    rect_img = Image.new("RGB", (512, 200), (255, 255, 255))
    letterboxed = letterbox_image(rect_img, target_size=(224, 224), bg_color=(255, 255, 255))
    
    assert letterboxed.size == (224, 224), f"Expected (224, 224), got {letterboxed.size}"
    arr = np.asarray(letterboxed)
    # Check that top and bottom padding rows are pure white (255, 255, 255)
    assert np.all(arr[0, :, :] == 255), "Top border padding should be white"
    assert np.all(arr[-1, :, :] == 255), "Bottom border padding should be white"


def test_custom_cnn_rescaling():
    """Verifies that CNN preprocessing scales pixel values strictly to [0.0, 1.0]."""
    test_arr = np.array([[[0, 128, 255]]], dtype=np.float32)
    res = preprocess_for_model(test_arr, model_type="cnn")
    assert res.min() >= 0.0 and res.max() <= 1.0
    assert np.isclose(res[0, 0, 0], 0.0)
    assert np.isclose(res[0, 0, 2], 1.0)


def test_vgg16_resnet50_caffe_subtraction():
    """Verifies BGR channel swap and ImageNet mean subtraction."""
    test_rgb = np.zeros((1, 1, 3), dtype=np.float32)
    test_rgb[0, 0, 0] = 100.0  # R
    test_rgb[0, 0, 1] = 150.0  # G
    test_rgb[0, 0, 2] = 200.0  # B
    
    res_vgg = preprocess_for_model(test_rgb, model_type="vgg16")
    # Expected: BGR order -> [B - 103.939, G - 116.779, R - 123.68]
    assert np.isclose(res_vgg[0, 0, 0], 200.0 - 103.939, atol=1e-3)
    assert np.isclose(res_vgg[0, 0, 1], 150.0 - 116.779, atol=1e-3)
    assert np.isclose(res_vgg[0, 0, 2], 100.0 - 123.680, atol=1e-3)


def test_augmentation_output_validity():
    """Verifies that augmentation generates valid PIL images of identical dimensions."""
    img = Image.new("RGB", (224, 224), (200, 200, 200))
    aug = augment_image(img)
    assert aug.size == (224, 224)
    assert aug.mode == "RGB"

