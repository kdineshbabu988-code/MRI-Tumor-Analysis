import os
import numpy as np
from PIL import Image
import utils
import config

def create_test_images():
    ensure_dir("test_images")
    
    # 1. Valid MRI-like image (Grayscale)
    mri_like = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    Image.fromarray(mri_like).save("test_images/valid_mri.png")
    
    # 2. Invalid Color image (Highly saturated)
    color_img = np.zeros((224, 224, 3), dtype=np.uint8)
    color_img[:, :, 0] = 255 # Bright red
    Image.fromarray(color_img).save("test_images/invalid_color.png")
    
    # 3. Invalid Blank image
    blank_img = np.zeros((224, 224), dtype=np.uint8)
    Image.fromarray(blank_img).save("test_images/invalid_blank.png")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def test_validation():
    print("--- Testing Image Validation ---")
    
    test_cases = [
        ("test_images/valid_mri.png", True),
        ("test_images/invalid_color.png", False),
        ("test_images/invalid_blank.png", False)
    ]
    
    for path, expected in test_cases:
        is_valid, msg = utils.validate_image_content(path)
        status = "PASS" if is_valid == expected else "FAIL"
        print(f"File: {path:30} | Expected: {str(expected):5} | Result: {str(is_valid):5} | {status}")
        if not is_valid:
            print(f"  Error message: {msg}")

if __name__ == "__main__":
    create_test_images()
    test_validation()
    print("\nVerification complete.")
