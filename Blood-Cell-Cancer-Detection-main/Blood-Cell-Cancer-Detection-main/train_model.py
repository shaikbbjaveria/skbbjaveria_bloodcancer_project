import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import os
import matplotlib.pyplot as plt

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DATASET_PATH = "Blood cell Cancer [ALL]"

# Class mapping based on analysis
CLASS_MAPPING = {
    0: "Malignant early Pre-B",
    1: "Malignant Pre-B", 
    2: "Malignant Pro-B",
    3: "Benign"
}

def create_data_generators():
    """Create training and validation data generators"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )
    
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

def create_model():
    """Create MobileNetV2 based model"""
    # Load MobileNetV2 with pre-trained weights, excluding top layer
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze the base model layers initially
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(4, activation='softmax')(x)  # 4 classes
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def fine_tune_model(model, base_model):
    """Fine-tune the model by unfreezing some layers"""
    # Unfreeze the top layers of the base model
    base_model.trainable = True
    
    # Freeze all layers except the last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Re-compile with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Main training function"""
    print("=== BLOOD CANCER DETECTION MODEL TRAINING ===")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset path {DATASET_PATH} does not exist!")
        print("Please make sure your dataset is in the 'Blood cell Cancer [ALL]' folder")
        return None, None
    
    # Create data generators
    print("\nCreating data generators...")
    train_generator, validation_generator = create_data_generators()
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Class indices: {train_generator.class_indices}")
    
    # Create model
    print("\nCreating model...")
    model, base_model = create_model()
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
    ]
    
    # Phase 1: Train with frozen base model
    print("\n=== PHASE 1: Training with frozen base model ===")
    history1 = model.fit(
        train_generator,
        epochs=EPOCHS//2,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tuning
    print("\n=== PHASE 2: Fine-tuning the model ===")
    model = fine_tune_model(model, base_model)
    
    history2 = model.fit(
        train_generator,
        epochs=EPOCHS//2,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.save('blood_cancer_model.h5')
    print("\nModel saved as 'blood_cancer_model.h5'")
    
    return model, (history1, history2)

def convert_to_tflite():
    """Convert the trained model to TensorFlow Lite format"""
    print("\n=== CONVERTING TO TENSORFLOW LITE ===")
    
    # Load the trained model
    model = tf.keras.models.load_model('blood_cancer_model.h5')
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Create a representative dataset for quantization
    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
            yield [data]
    
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    
    tflite_model = converter.convert()
    
    # Save the TensorFlow Lite model
    with open('model1 (MobileNetV2).tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("TensorFlow Lite model saved as 'model1 (MobileNetV2).tflite'")
    
    # Print model size
    h5_size = os.path.getsize('blood_cancer_model.h5') / (1024 * 1024)
    tflite_size = os.path.getsize('model1 (MobileNetV2).tflite') / (1024 * 1024)
    print(f"H5 model size: {h5_size:.2f} MB")
    print(f"TFLite model size: {tflite_size:.2f} MB")

def plot_training_history(history1, history2):
    """Plot training history"""
    # Combine histories
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    print("Training history plot saved as 'training_history.png'")

if __name__ == "__main__":
    # Train the model
    model, histories = train_model()
    
    if model is not None:
        # Convert to TensorFlow Lite
        convert_to_tflite()
        
        # Plot training history
        if histories:
            plot_training_history(histories[0], histories[1])
        
        print("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
        print("Files created:")
        print("- blood_cancer_model.h5 (Keras model)")
        print("- model1 (MobileNetV2).tflite (TensorFlow Lite model)")
        print("- training_history.png (Training plots)")
    else:
        print("\n=== TRAINING FAILED ===")
        print("Please check your dataset and try again.")
