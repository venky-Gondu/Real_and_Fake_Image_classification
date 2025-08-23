from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your trained model
model = tf.keras.models.load_model('C:\\Users\\HP\\OneDrive\\Desktop\\best_custom_cnn.h5')


def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        # Load and resize image to match training size (224x224)
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize like in training
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        processed_image = preprocess_image(filepath)
        
        if processed_image is not None:
            # Make prediction
            prediction = model.predict(processed_image)[0][0]
            confidence = float(prediction)
            
            # Convert to class label (adjust based on your model's output)
            if confidence > 0.5:
                result = "Real Art"
                confidence_percent = confidence * 100
            else:
                result = "AI Art"
                confidence_percent = (1 - confidence) * 100
            
            return jsonify({
                'prediction': result,
                'confidence': f"{confidence_percent:.2f}%",
                'raw_score': confidence
            })
        else:
            return jsonify({'error': 'Error processing image'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
