import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
import os

def load_images_from_directory(base_path, image_size=(224, 224)):
    """Load images from directory structure: base_path/gender/*.jpg"""
    images = []
    labels = []
    for gender in ['male', 'female']:
        gender_path = os.path.join(base_path, gender)
        if not os.path.exists(gender_path): continue
        for img_file in os.listdir(gender_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(gender_path, img_file))
                if img is not None:
                    img = cv2.resize(img, image_size)
                    img = img.astype('float32') / 255.0
                    images.append(img)
                    labels.append(gender)
    return np.array(images), np.array(labels)

def extract_features(images):
    """Extract HOG and LBP features for a batch of images."""
    features_list = []
    for img in images:
        # Convert to uint8 for feature extraction
        img_uint8 = (img * 255).astype('uint8')
        feat = _get_raw_features(img_uint8)
        features_list.append(feat)
    return np.array(features_list)

def preprocess_single_image(image_path, image_size=(224, 224)):
    """Required by app.py: Loads and resizes a single image."""
    img = cv2.imread(image_path)
    if img is None: return None
    img = cv2.resize(img, image_size)
    img = img.astype('float32') / 255.0
    return img

def extract_single_image_features(img):
    """Extracts features for one image. Ensures exactly 26,253 features."""
    img_uint8 = (img * 255).astype('uint8') if img.dtype == 'float32' else img
    feat = _get_raw_features(img_uint8)
    return feat.reshape(1, -1)

def _get_raw_features(img_uint8):
    """Internal helper to ensure feature consistency."""
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY) if len(img_uint8.shape) == 3 else img_uint8
    
    # 1. HOG Features (Produces 26,244 features)
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    
    # 2. LBP Features (Must produce EXACTLY 9 features for PCA compatibility)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    # Using bins=np.arange(0, 10) creates 9 bins (0-1, 1-2 ... 8-9)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    
    return np.hstack([hog_feat, hist])

def validate_pelvic_xray(image_path):
    """Adaptive validation for pediatric and adult pelvic X-rays."""
    try:
        if not os.path.exists(image_path): return False, "File not found"
        img = cv2.imread(image_path)
        if img is None: return False, "Invalid image file"
        
        h, w = img.shape[:2]
        if h < 512 or w < 512: return False, "Resolution too low for medical analysis."
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        p2, p98 = np.percentile(gray, (2, 98))
        if (p98 - p2) < 50: return False, "Insufficient dynamic range (Low contrast)."

        mean_val, std_val = np.mean(gray), np.std(gray)
        # Pediatric-friendly thresholds
        dark_thresh = min(65, mean_val - (0.4 * std_val))
        bright_thresh = max(145, mean_val + (0.4 * std_val))
        
        if (np.sum(gray < dark_thresh) / gray.size) < 0.03:
            return False, "Image lacks characteristic X-ray background."
        if (np.sum(gray > bright_thresh) / gray.size) < 0.02:
            return False, "Image lacks characteristic bone density."

        return True, "Valid pelvic X-ray"
    except Exception as e:
        return False, f"Validation error: {str(e)}"