#!/usr/bin/env python3
"""
Simple Flower Classification CNN
Trains a model and allows testing with your own images
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

def check_dataset():
    """Check if dataset exists"""
    dataset_path = './Dataset'
    if os.path.exists(dataset_path):
        classes = os.listdir(dataset_path)
        print(f"Dataset found with classes: {classes}")
        return True
    else:
        print("Dataset not found!")
        return False

def prepare_data():
    """Prepare training and validation data"""
    train_dir = './Dataset'
    img_size = (64, 64)
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
    
    return train_generator, validation_generator, img_size

def build_model():
    """Build CNN model"""
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
    print("Model built successfully")
    return model

def train_model(model, train_generator, validation_generator):
    """Train the model"""
    print("Training model...")
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator
    )

    final_train_accuracy = history.history['accuracy'][-1] * 100
    final_val_accuracy = history.history['val_accuracy'][-1] * 100
    print(f"Training Accuracy: {final_train_accuracy:.2f}%")
    print(f"Validation Accuracy: {final_val_accuracy:.2f}%")
    
    return history

def save_model(model, train_generator):
    """Save model and get class names"""
    model.save('flower_classifier_model.h5')
    class_names = list(train_generator.class_indices.keys())
    print(f"Model saved. Classes: {class_names}")
    return class_names

def plot_training_history(history):
    """Plot training results"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def select_and_predict_image(model, class_names, img_size):
    """Select and predict image"""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        print("Opening file dialog...")
        img_path = filedialog.askopenfilename(
            title="Select a flower image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if img_path:
            print(f"Selected: {os.path.basename(img_path)}")
            
            # Load and preprocess image
            img = image.load_img(img_path, target_size=img_size)
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            print("Making prediction...")
            prediction = model.predict(img_array, verbose=0)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            # Display results
            plt.figure(figsize=(12, 5))
            
            # Show image
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Selected Image\nPredicted: {predicted_class}\nConfidence: {confidence:.1f}%', fontsize=12)
            
            # Show probabilities
            plt.subplot(1, 2, 2)
            probabilities = prediction[0] * 100
            colors = ['green' if i == np.argmax(prediction) else 'lightblue' for i in range(len(class_names))]
            bars = plt.bar(class_names, probabilities, color=colors)
            plt.ylabel('Confidence (%)')
            plt.title('Prediction Confidence')
            plt.ylim(0, 100)
            
            for bar, prob in zip(bars, probabilities):
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                        f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
            print(f"\nResult: {predicted_class} ({confidence:.1f}% confidence)")
            print("All probabilities:")
            for i, class_name in enumerate(class_names):
                print(f"  {class_name}: {probabilities[i]:.1f}%")
                
        else:
            print("No image selected")
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        try:
            root.destroy()
        except:
            pass

def main():
    """Main function"""
    print("=== Flower Classification CNN ===")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check dataset
    if not check_dataset():
        return
    
    # Prepare data
    train_generator, validation_generator, img_size = prepare_data()
    
    # Build model
    model = build_model()
    
    # Train model
    history = train_model(model, train_generator, validation_generator)
    
    # Save model
    class_names = save_model(model, train_generator)
    
    # Plot results
    plot_training_history(history)
    
    # Test with user image
    print("\n=== Testing Phase ===")
    while True:
        choice = input("\nDo you want to test with your own image? (y/n): ").lower()
        if choice == 'y':
            select_and_predict_image(model, class_names, img_size)
        elif choice == 'n':
            print("Goodbye!")
            break
        else:
            print("Please enter 'y' or 'n'")

if __name__ == "__main__":
    main()
