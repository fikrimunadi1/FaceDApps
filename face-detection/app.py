from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda
import numpy as np
import os
from PIL import Image
import cv2
import tensorflow as tf
import h5py
import json
import base64

from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

model_path = 'best_vgg16_balanced_strategy6_1 (2).h5'
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Custom function for the Lambda layer with explicit output shape
def grayscale_to_rgb(x):
    return tf.repeat(x, 3, axis=-1)

# Register the custom function with Keras
@tf.keras.utils.register_keras_serializable(package='Custom')
def grayscale_to_rgb_registered(x):
    return grayscale_to_rgb(x)

# Custom objects for model loading
custom_objects = {
    'grayscale_to_rgb': grayscale_to_rgb_registered,
    'Lambda': Lambda(grayscale_to_rgb_registered, output_shape=(None, 48, 48, 3))
}

model = None

try:
    # First try loading with custom objects
    model = load_model(model_path, custom_objects=custom_objects, compile=False)
    print("✅ Model loaded successfully with custom objects!")
except Exception as e:
    print(f"[ERROR] First load attempt failed: {str(e)}")
    
    try:
        # Alternative approach: Rebuild the model architecture and load weights
        # Define the model architecture first
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
        
        # Input layer
        input_layer = Input(shape=(48, 48, 1), name='grayscale_input')
        
        # Lambda layer to convert grayscale to RGB
        x = Lambda(grayscale_to_rgb_registered, output_shape=(48, 48, 3), name='grayscale_to_rgb')(input_layer)
        
        # VGG16 blocks
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        
        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        
        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        
        # Top layers
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = Dense(256, activation='relu', name='fc1')(x)
        x = BatchNormalization(name='bn1')(x)
        x = Dropout(0.5, name='dropout1')(x)
        output_layer = Dense(7, activation='softmax', name='predictions')(x)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output_layer, name='VGG16_Simplified_EmotionRecognition')
        
        # Load weights
        model.load_weights(model_path)
        print("✅ Model loaded using architecture reconstruction method!")
    except Exception as e2:
        print(f"[ERROR] Architecture reconstruction method failed: {str(e2)}")
        model = None

def preprocess_image(image, target_size=(48, 48)):
    """
    Preprocess image for model prediction
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
    # Resize with padding to maintain aspect ratio
    height, width = image.shape
    target_width, target_height = target_size
        
    # Calculate padding
    if width > height:
        new_width = target_width
        new_height = int(height * (target_width / width))
    else:
        new_height = target_height
        new_width = int(width * (target_height / height))
        
    # Resize
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
    # Create canvas with target size
    canvas = np.zeros((target_height, target_width), dtype=np.uint8)
        
    # Center the image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
    # Normalize and add channel dimension
    normalized = canvas / 255.0
    normalized = np.expand_dims(normalized, axis=-1)
    normalized = np.expand_dims(normalized, axis=0)  # Add batch dimension
        
    return normalized

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
            
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    try:
        # Read image
        img = Image.open(file.stream)
        img = np.array(img)
                
        # Preprocess image
        processed_img = preprocess_image(img)
                
        # Make prediction
        predictions = model.predict(processed_img)[0]
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx]) * 100
                
        # Get top 3 predictions
        top3_indices = np.argsort(predictions)[-3:][::-1]
        top3 = [{
            "label": CLASS_NAMES[i],
            "confidence": float(predictions[i]) * 100
        } for i in top3_indices]

        return jsonify({
            "predicted_emotion": CLASS_NAMES[predicted_class_idx],
            "confidence": confidence,
            "top_predictions": top3
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
