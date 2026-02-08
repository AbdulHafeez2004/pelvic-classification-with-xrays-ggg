import cv2
import numpy as np
import os
from utils.preprocessing import validate_pelvic_xray

def create_realistic_test_images():
    """Create more realistic synthetic test images that will pass validation"""
    test_dir = 'test_images'
    os.makedirs(test_dir, exist_ok=True)
    
    # 1. Create a realistic pelvic X-ray that will PASS
    # X-rays have characteristic bimodal histogram (very dark and very bright areas)
    pelvic_img = np.zeros((1024, 1024), dtype=np.uint8)
    
    # Add very dark background (typical of X-ray background)
    pelvic_img[:, :] = 30  # Very dark
    
    # Add bright bone structures (typical of X-ray bones)
    # Hip bones - bright white
    cv2.ellipse(pelvic_img, (350, 512), (180, 120), 0, 0, 180, 220, -1)  # Left hip
    cv2.ellipse(pelvic_img, (674, 512), (180, 120), 0, 0, 180, 220, -1)  # Right hip
    
    # Sacrum - bright
    cv2.rectangle(pelvic_img, (475, 300), (549, 600), 200, -1)
    
    # Femur heads - very bright
    cv2.circle(pelvic_img, (350, 700), 60, 240, -1)  # Left femur
    cv2.circle(pelvic_img, (674, 700), 60, 240, -1)  # Right femur
    
    # Add some noise to simulate X-ray grain
    noise = np.random.normal(0, 5, (1024, 1024)).astype(np.uint8)
    pelvic_img = cv2.add(pelvic_img, noise)
    
    cv2.imwrite(os.path.join(test_dir, 'realistic_pelvic_xray.jpg'), pelvic_img)
    
    # 2. Create a real pelvic X-ray-like image using actual X-ray characteristics
    # X-rays have specific intensity distribution
    xray_like = np.zeros((1024, 1024), dtype=np.uint8)
    
    # Create bimodal distribution: very dark background + bright bones
    # 70% dark background (0-50 intensity)
    xray_like[:716, :] = np.random.randint(0, 50, (716, 1024))
    
    # 25% medium intensity (50-150) for soft tissues
    xray_like[716:972, :] = np.random.randint(50, 150, (256, 1024))
    
    # 5% very bright (200-255) for bones
    xray_like[972:, :] = np.random.randint(200, 255, (52, 1024))
    
    # Add actual bone-like structures
    cv2.ellipse(xray_like, (512, 512), (300, 200), 0, 0, 360, 180, 40)
    cv2.ellipse(xray_like, (300, 600), (80, 60), 0, 0, 360, 220, -1)
    cv2.ellipse(xray_like, (724, 600), (80, 60), 0, 0, 360, 220, -1)
    
    cv2.imwrite(os.path.join(test_dir, 'xray_like_image.jpg'), xray_like)
    
    # 3. Keep your existing test images for comparison
    # Regular photo (should FAIL)
    photo_img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(test_dir, 'regular_photo.jpg'), photo_img)
    
    # Small image (should FAIL)
    small_img = np.random.normal(128, 30, (300, 300)).astype(np.uint8)
    cv2.imwrite(os.path.join(test_dir, 'small_image.jpg'), small_img)
    
    print(f"✅ Created realistic test images in '{test_dir}' folder")

def analyze_image_histogram(image_path):
    """Analyze why an image is failing validation"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Could not read image")
        return
    
    # Calculate histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    
    print(f"\n📊 Histogram analysis for {os.path.basename(image_path)}:")
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    print(f"Intensity range: {img.min()} - {img.max()}")
    print(f"Mean intensity: {img.mean():.1f}")
    print(f"Standard deviation: {img.std():.1f}")
    
    # Find peaks
    peaks = []
    for i in range(1, 255):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0.005:
            peaks.append((i, hist[i]))
    
    peaks.sort(key=lambda x: x[1], reverse=True)
    print(f"Peaks found: {peaks[:5]}")  # Top 5 peaks
    
    # Check if it has the bimodal distribution of X-rays
    if len(peaks) >= 2:
        peak1, peak2 = peaks[0][0], peaks[1][0]
        peak_diff = abs(peak1 - peak2)
        print(f"Peak difference: {peak_diff}")
        if peak_diff > 10:  # X-rays typically have widely separated peaks
            print("✅ Has bimodal distribution (X-ray characteristic)")
        else:
            print("❌ Lacks strong bimodal distribution")
    else:
        print("❌ Not enough peaks for bimodal distribution")

def test_validation():
    """Test the pelvic X-ray validation"""
    test_images = [
        ("test_images/real_pelvic_xray_1.jpg", "Realistic synthetic X-ray (should PASS)", True),
        ("test_images/real_pelvic_xray_2.jpg", "X-ray-like image (should PASS)", True),
        ("test_images/IMG_9805.jpg", "Regular photo (should FAIL)", False),
        ("test_images/small_image.jpg", "Small image (should FAIL)", False),
    ]
    
    # Add your real pelvic X-ray if it exists
    real_xray_path = "test_images/real_pelvic_xray_1.jpg"
    if os.path.exists(real_xray_path):
        test_images.insert(0, (real_xray_path, "Real pelvic X-ray (should PASS)", True))
    
    print("🧪 Testing Pelvic X-ray Validation")
    print("=" * 60)
    
    all_pass = True
    for filename, description, expected in test_images:
        if os.path.exists(filename):
            # First analyze why it might pass/fail
            analyze_image_histogram(filename)
            
            # Then test validation
            is_valid, message = validate_pelvic_xray(filename)
            
            status = "✅ PASS" if is_valid == expected else "❌ FAIL"
            result_color = "PASS" if is_valid == expected else "FAIL"
            
            print(f"{status} {description}: {message}")
            
            if is_valid != expected:
                all_pass = False
                print(f"   Expected: {'PASS' if expected else 'FAIL'}, Got: {'PASS' if is_valid else 'FAIL'}")
            
            print("-" * 40)
        else:
            print(f"⚠️  SKIP {description}: File not found")
    
    print("=" * 60)
    if all_pass:
        print("🎉 All tests passed! Validation is working correctly.")
    else:
        print("❌ Some tests failed. Check the validation logic.")
    
    return all_pass

if __name__ == "__main__":
    # Create realistic test images
    create_realistic_test_images()
    
    # Run the tests with analysis
    test_validation()