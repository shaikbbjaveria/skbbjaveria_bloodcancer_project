import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob

# Load the trained model
print("Loading trained model...")
model = tf.keras.models.load_model('best_model.h5')
print("Model loaded successfully!")

# Class mapping based on training
CLASS_MAPPING = {
    0: "Benign",
    1: "[Malignant] Pre-B", 
    2: "[Malignant] Pro-B",
    3: "[Malignant] early Pre-B"
}

# Reverse mapping for predictions
PREDICTION_MAPPING = {
    0: "Benign",
    1: "[Malignant] Pre-B", 
    2: "[Malignant] Pro-B",
    3: "[Malignant] early Pre-B"
}

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    image = Image.open(image_path)
    image = image.resize((224, 224))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(image_path):
    """Make prediction on a single image"""
    try:
        processed_image = preprocess_image(image_path)
        prediction = model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return predicted_class, confidence, prediction[0]
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None, None

def test_predicted_split_folder():
    """Test all images in PREDICTED_SPLIT folder"""
    print("\n=== TESTING PREDICTED_SPLIT FOLDER ===")
    
    test_folders = [
        "PREDICTED_SPLIT/Benign",
        "PREDICTED_SPLIT/Early_Pre_B", 
        "PREDICTED_SPLIT/Pre_B",
        "PREDICTED_SPLIT/Pro_B"
    ]
    
    folder_mapping = {
        "Benign": "Benign",
        "Early_Pre_B": "[Malignant] early Pre-B",
        "Pre_B": "[Malignant] Pre-B", 
        "Pro_B": "[Malignant] Pro-B"
    }
    
    total_correct = 0
    total_images = 0
    
    for folder in test_folders:
        if os.path.exists(folder):
            folder_name = os.path.basename(folder)
            expected_class = folder_mapping[folder_name]
            
            # Find all images in the folder
            image_files = glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))
            
            print(f"\n--- Testing {folder_name} (Expected: {expected_class}) ---")
            print(f"Found {len(image_files)} images")
            
            folder_correct = 0
            folder_total = 0
            
            # Test first 5 images from each folder (for quick testing)
            for i, image_path in enumerate(image_files[:5]):
                predicted_class, confidence, predictions = predict_image(image_path)
                
                if predicted_class is not None:
                    predicted_label = PREDICTION_MAPPING[predicted_class]
                    is_correct = (predicted_label == expected_class)
                    
                    if is_correct:
                        folder_correct += 1
                        total_correct += 1
                    
                    folder_total += 1
                    total_images += 1
                    
                    print(f"  {os.path.basename(image_path)}: {predicted_label} ({confidence:.2%}) {'✓' if is_correct else '✗'}")
                    
                    # Show all class probabilities
                    prob_str = "    Probabilities: "
                    for j, prob in enumerate(predictions):
                        prob_str += f"{PREDICTION_MAPPING[j]}: {prob:.2%}  "
                    print(prob_str)
            
            accuracy = (folder_correct / folder_total * 100) if folder_total > 0 else 0
            print(f"  Folder Accuracy: {accuracy:.1f}% ({folder_correct}/{folder_total})")
    
    overall_accuracy = (total_correct / total_images * 100) if total_images > 0 else 0
    print(f"\n=== OVERALL ACCURACY: {overall_accuracy:.1f}% ({total_correct}/{total_images}) ===")
    
    return overall_accuracy

def test_sample_images():
    """Test a few sample images from each category"""
    print("\n=== TESTING SAMPLE IMAGES FROM EACH CATEGORY ===")
    
    sample_images = [
        ("Blood cell Cancer [ALL]/Benign/Sap_013 (1).jpg", "Benign"),
        ("Blood cell Cancer [ALL]/[Malignant] Pre-B/Snap_001.jpg", "[Malignant] Pre-B"),
        ("Blood cell Cancer [ALL]/[Malignant] Pro-B/Snap_001.jpg", "[Malignant] Pro-B"),
        ("Blood cell Cancer [ALL]/[Malignant] early Pre-B/Snap_001.jpg", "[Malignant] early Pre-B")
    ]
    
    for image_path, expected_class in sample_images:
        if os.path.exists(image_path):
            predicted_class, confidence, predictions = predict_image(image_path)
            
            if predicted_class is not None:
                predicted_label = PREDICTION_MAPPING[predicted_class]
                is_correct = (predicted_label == expected_class)
                
                print(f"\n{os.path.basename(image_path)}:")
                print(f"  Expected: {expected_class}")
                print(f"  Predicted: {predicted_label} ({confidence:.2%}) {'✓' if is_correct else '✗'}")
                print(f"  All probabilities: {dict(zip(PREDICTION_MAPPING.values(), predictions))}")
        else:
            print(f"Image not found: {image_path}")

def convert_to_tflite():
    """Convert the trained model to TensorFlow Lite format"""
    print("\n=== CONVERTING TO TENSORFLOW LITE ===")
    
    try:
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Save the TensorFlow Lite model
        with open('model1 (MobileNetV2).tflite', 'wb') as f:
            f.write(tflite_model)
        
        print("TensorFlow Lite model saved as 'model1 (MobileNetV2).tflite'")
        
        # Print model size
        h5_size = os.path.getsize('best_model.h5') / (1024 * 1024)
        tflite_size = os.path.getsize('model1 (MobileNetV2).tflite') / (1024 * 1024)
        print(f"H5 model size: {h5_size:.2f} MB")
        print(f"TFLite model size: {tflite_size:.2f} MB")
        
        return True
    except Exception as e:
        print(f"Error converting to TFLite: {e}")
        return False

if __name__ == "__main__":
    print("=== BLOOD CANCER DETECTION MODEL TESTING ===")
    
    # Test sample images
    test_sample_images()
    
    # Test PREDICTED_SPLIT folder
    accuracy = test_predicted_split_folder()
    
    # Convert to TFLite if accuracy is good
    if accuracy > 70:
        print("\nModel accuracy is good. Converting to TensorFlow Lite...")
        convert_to_tflite()
        print("\n=== TESTING COMPLETED SUCCESSFULLY ===")
        print("Your model is ready for use!")
    else:
        print(f"\nModel accuracy ({accuracy:.1f}%) could be improved.")
        print("Consider retraining with more epochs or different parameters.")
