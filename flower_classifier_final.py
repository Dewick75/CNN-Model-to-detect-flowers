#!/usr/bin/env python3
"""
Complete Flower Classification CNN with Image Selection
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog

# Global variables
model = None
class_names = None
img_size = (64, 64)

def train_model():
    """Train the flower classification model"""
    global model, class_names
    
    print("=== Training Flower Classification Model ===")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check dataset
    dataset_path = './Dataset'
    if not os.path.exists(dataset_path):
        print("❌ Dataset not found!")
        return False
    
    classes = os.listdir(dataset_path)
    print(f"✅ Dataset found with classes: {classes}")
    
    # Data preparation
    train_dir = './Dataset'
    batch_size = 32

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Build model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("✅ Model built successfully")
    
    # Train model
    print("🚀 Training model (this may take 1-2 minutes)...")
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        verbose=1
    )

    # Get results
    final_train_accuracy = history.history['accuracy'][-1] * 100
    final_val_accuracy = history.history['val_accuracy'][-1] * 100
    print(f"✅ Training Complete!")
    print(f"📊 Training Accuracy: {final_train_accuracy:.2f}%")
    print(f"📊 Validation Accuracy: {final_val_accuracy:.2f}%")
    
    # Save model and get class names
    model.save('flower_classifier_model.h5')
    class_names = list(train_generator.class_indices.keys())
    print(f"💾 Model saved. Classes: {class_names}")
    
    return True

def select_and_predict_image():
    """Select and predict image"""
    global model, class_names, img_size
    
    if model is None:
        print("❌ Model not trained yet! Run train_model() first.")
        return
    
    try:
        print("📁 Opening file dialog...")
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        img_path = filedialog.askopenfilename(
            title="Select a flower image for classification",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if img_path:
            print(f"✅ Selected: {os.path.basename(img_path)}")
            
            # Load and preprocess image
            img = image.load_img(img_path, target_size=img_size)
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            print("🧠 Making prediction...")
            prediction = model.predict(img_array, verbose=0)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            # Display results
            plt.figure(figsize=(12, 5))
            
            # Show image
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Selected Image\\nPredicted: {predicted_class}\\nConfidence: {confidence:.1f}%', 
                     fontsize=12, fontweight='bold')
            
            # Show probabilities
            plt.subplot(1, 2, 2)
            probabilities = prediction[0] * 100
            colors = ['green' if i == np.argmax(prediction) else 'lightblue' for i in range(len(class_names))]
            bars = plt.bar(class_names, probabilities, color=colors)
            plt.ylabel('Confidence (%)')
            plt.title('Prediction Confidence')
            plt.ylim(0, 100)
            
            # Add percentage labels on bars
            for bar, prob in zip(bars, probabilities):
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                        f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
            # Print results
            print("\\n" + "="*50)
            print("🌺 PREDICTION RESULTS")
            print("="*50)
            print(f"🖼️  Image: {os.path.basename(img_path)}")
            print(f"🌸 Predicted: {predicted_class}")
            print(f"📊 Confidence: {confidence:.1f}%")
            print("\\n📈 All probabilities:")
            for i, class_name in enumerate(class_names):
                emoji = "🎯" if i == np.argmax(prediction) else "  "
                print(f"   {emoji} {class_name}: {probabilities[i]:.1f}%")
            print("="*50)
                
        else:
            print("❌ No image selected")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("💡 Make sure the image file is valid and accessible")
    finally:
        try:
            root.destroy()
        except:
            pass

def main():
    """Main function - complete workflow"""
    print("🌺 Welcome to Flower Classification CNN!")
    print("This program will:")
    print("1. Train a CNN model on flower images")
    print("2. Allow you to test it with your own images")
    print()
    
    # Train the model
    if train_model():
        print("\\n🎉 Training successful! Ready for testing.")
        
        # Testing loop
        while True:
            print("\\n" + "="*40)
            print("🧪 TESTING PHASE")
            print("="*40)
            choice = input("Do you want to test with your own image? (y/n): ").lower().strip()
            
            if choice == 'y' or choice == 'yes':
                select_and_predict_image()
            elif choice == 'n' or choice == 'no':
                print("👋 Thank you for using Flower Classification CNN!")
                break
            else:
                print("❌ Please enter 'y' for yes or 'n' for no")
    else:
        print("❌ Training failed. Please check your dataset.")

if __name__ == "__main__":
    main()
