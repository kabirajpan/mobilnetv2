# improved_augment_config.py
# Enhanced augmentation config with better validation split support and more robust preprocessing

import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------------------
# Extra preprocessing helper - IMPROVED
# ------------------------------
def extra_preprocessing(img):
    """
    Enhanced per-image preprocessing with more sophisticated augmentations.
    Robust to input range and adds more realistic variations.
    Returns image in float32 with values in [0, 1].
    """
    # Ensure float32 and normalize if needed
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0

    # ---------- 1) Enhanced contrast adjustment ----------
    # More sophisticated contrast using CLAHE occasionally
    if np.random.rand() < 0.3:
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        lab[..., 0] = clahe.apply(lab[..., 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    else:
        # Standard contrast adjustment
        alpha = np.random.uniform(0.85, 1.15)
        img = np.clip(alpha * img, 0.0, 1.0)

    # ---------- 2) Enhanced color augmentations ----------
    if np.random.rand() < 0.8:  # Apply color changes more frequently
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Enhanced saturation adjustment (±15%)
        sat_scale = np.random.uniform(0.85, 1.15)
        hsv[..., 1] = hsv[..., 1] * sat_scale
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        
        # Enhanced hue shift (±8 degrees)
        hue_shift = np.random.randint(-8, 9)
        hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
        
        # Enhanced value/brightness adjustment
        val_scale = np.random.uniform(0.9, 1.1)
        hsv[..., 2] = hsv[..., 2] * val_scale
        hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
        
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    # ---------- 3) Enhanced noise simulation ----------
    noise_prob = np.random.rand()
    if noise_prob < 0.4:  # Gaussian noise
        noise_std = np.random.uniform(0.005, 0.025)
        noise = np.random.normal(0.0, noise_std, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)
    elif noise_prob < 0.5:  # Salt and pepper noise (rare)
        noise_mask = np.random.random(img.shape[:2]) < 0.002
        img[noise_mask] = np.random.choice([0.0, 1.0], size=img[noise_mask].shape)

    # ---------- 4) Enhanced blur effects ----------
    blur_prob = np.random.rand()
    if blur_prob < 0.25:  # Gaussian blur
        kernel_size = np.random.choice([3, 5])
        sigma = np.random.uniform(0.3, 1.0)
        img = cv2.GaussianBlur((img * 255).astype(np.uint8), (kernel_size, kernel_size), sigma).astype(np.float32) / 255.0
    elif blur_prob < 0.35:  # Motion blur
        size = np.random.choice([3, 5, 7])
        angle = np.random.uniform(0, 180)
        kernel = np.zeros((size, size))
        if np.random.rand() < 0.5:  # Linear motion blur
            kernel[int((size-1)/2), :] = 1.0
        else:  # Diagonal motion blur
            np.fill_diagonal(kernel, 1.0)
        kernel = kernel / kernel.sum()
        img = cv2.filter2D((img * 255).astype(np.uint8), -1, kernel).astype(np.float32) / 255.0

    # ---------- 5) Lighting effects ----------
    if np.random.rand() < 0.2:
        # Simulate uneven lighting by creating a gradient
        H, W = img.shape[:2]
        center_x, center_y = np.random.uniform(0.3, 0.7) * W, np.random.uniform(0.3, 0.7) * H
        y, x = np.ogrid[:H, :W]
        
        # Create radial gradient
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(H**2 + W**2)
        gradient = 1.0 - (distance / max_distance) * np.random.uniform(0.1, 0.3)
        gradient = np.clip(gradient, 0.7, 1.0)
        
        # Apply gradient to each channel
        for c in range(3):
            img[..., c] = img[..., c] * gradient

    # ---------- 6) Shadow simulation ----------
    if np.random.rand() < 0.15:
        # Create random shadow pattern
        H, W = img.shape[:2]
        shadow_mask = np.ones((H, W), dtype=np.float32)
        
        # Create irregular shadow shape
        num_points = np.random.randint(3, 7)
        points = np.random.randint(0, min(H, W), size=(num_points, 2))
        cv2.fillPoly(shadow_mask, [points], 0.6)
        
        # Blur the shadow for realism
        shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 0)
        
        # Apply shadow
        for c in range(3):
            img[..., c] = img[..., c] * shadow_mask

    return np.clip(img.astype(np.float32), 0.0, 1.0)

# ------------------------------
# Enhanced ImageDataGenerator config
# ------------------------------
def augment_config(for_validation=False):
    """
    Returns a Keras ImageDataGenerator with enhanced configurations.
    
    Args:
        for_validation (bool): If True, returns minimal augmentation for validation
    
    Returns:
        ImageDataGenerator: Configured data generator
    """
    
    if for_validation:
        # Minimal augmentation for validation
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
    else:
        # Full augmentation for training
        datagen = ImageDataGenerator(
            # Enhanced geometric transforms
            rotation_range=25,              # ±25 degrees rotation
            width_shift_range=0.15,         # horizontal translations up to 15%
            height_shift_range=0.15,        # vertical translations up to 15%
            shear_range=0.1,               # shear transformation
            zoom_range=[0.8, 1.2],         # enhanced zoom range
            horizontal_flip=True,           # horizontal flip
            vertical_flip=True,             # vertical flip
            fill_mode='reflect',            # better fill mode
            
            # Enhanced brightness range
            brightness_range=[0.7, 1.3],   # ±30% brightness variation
            
            # Channel shift for color variation
            channel_shift_range=20.0,       # random channel shifts
            
            # Set validation split
            validation_split=0.2,
            
            # Apply our enhanced preprocessing
            preprocessing_function=extra_preprocessing
        )

    return datagen

# ------------------------------
# Utility function for different augmentation levels
# ------------------------------
def get_augmentation_config(level='medium'):
    """
    Get augmentation configuration based on desired intensity level.
    
    Args:
        level (str): 'light', 'medium', 'heavy'
    
    Returns:
        ImageDataGenerator: Configured data generator
    """
    
    configs = {
        'light': {
            'rotation_range': 15,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'zoom_range': [0.9, 1.1],
            'brightness_range': [0.8, 1.2],
            'preprocessing_function': None
        },
        'medium': {
            'rotation_range': 25,
            'width_shift_range': 0.15,
            'height_shift_range': 0.15,
            'shear_range': 0.1,
            'zoom_range': [0.8, 1.2],
            'brightness_range': [0.7, 1.3],
            'channel_shift_range': 20.0,
            'preprocessing_function': extra_preprocessing
        },
        'heavy': {
            'rotation_range': 35,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'shear_range': 0.15,
            'zoom_range': [0.7, 1.3],
            'brightness_range': [0.6, 1.4],
            'channel_shift_range': 30.0,
            'preprocessing_function': extra_preprocessing
        }
    }
    
    config = configs.get(level, configs['medium'])
    
    return ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect',
        validation_split=0.2,
        **config
    )
