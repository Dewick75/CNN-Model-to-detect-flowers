# Quick test to verify the notebook can run without errors
import os
import sys

def test_imports():
    """Test if all required libraries can be imported"""
    try:
        import tensorflow as tf
        import matplotlib.pyplot as plt
        import numpy as np
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
        from tensorflow.keras.preprocessing import image
        print("✅ All imports successful!")
        print(f"TensorFlow version: {tf.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_dataset():
    """Test if dataset exists and is properly structured"""
    dataset_path = './Dataset'
    if not os.path.exists(dataset_path):
        print("❌ Dataset folder not found!")
        return False
    
    classes = os.listdir(dataset_path)
    if len(classes) == 0:
        print("❌ No classes found in dataset!")
        return False
    
    print(f"✅ Dataset found with {len(classes)} classes: {classes}")
    
    # Count images in each class
    total_images = 0
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            print(f"   📁 {class_name}: {len(images)} images")
            total_images += len(images)
    
    print(f"📊 Total images: {total_images}")
    return True

def test_notebook_ready():
    """Test if the notebook environment is ready"""
    print("🧪 Testing Notebook Environment")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        return False
    
    print()
    
    # Test dataset
    if not test_dataset():
        return False
    
    print()
    print("🎉 Environment is ready!")
    print("💡 You can now run the CNN.ipynb notebook cell by cell")
    print("📝 Instructions:")
    print("   1. Open CNN.ipynb in VS Code")
    print("   2. Select Python kernel when prompted")
    print("   3. Run cells sequentially using Shift+Enter")
    print("   4. In the last cell, uncomment one testing option")
    
    return True

if __name__ == "__main__":
    test_notebook_ready()
