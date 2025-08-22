import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
from collections import Counter
import traceback

# 📁 Define model paths (corrected names)
model_paths = [
    'models/real_vs_fake_custommodel.keras',
    'models/real_vs_fake_custommode2.keras',
    'models/real_vs_fake_custommode3.keras'
]

# 🧠 Load models safely
def load_all_models(paths):
    models = []
    for path in paths:
        if not os.path.exists(path):
            st.warning(f"⚠️ Model file not found: {path}")
            continue
        try:
            model = keras.models.load_model(path, compile=False)
            models.append(model)
            st.success(f"✅ Loaded model: {os.path.basename(path)}")
        except Exception as e:
            st.error(f"❌ Error loading model: {path}")
            st.text(traceback.format_exc())
    return models

# 🖼️ Preprocess image for prediction
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Convert RGBA to RGB
    image = image.resize((224, 224))
    img_array = keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# 🔍 Get predictions from all models
def get_ensemble_predictions(image_array, models):
    predictions = []
    for model in models:
        pred = model.predict(image_array)
        predictions.append(pred[0][0])  # Binary classification: probability of 'real'
    return predictions

# 🚀 Streamlit UI
st.set_page_config(page_title="Real vs Fake Image Classifier", layout="centered")
st.title("🧠 Real vs Fake Face Classifier")
st.write("Upload an image to get predictions from an ensemble of models trained to detect authenticity.")

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

# Load models once
loaded_models = load_all_models(model_paths)

if uploaded_file is not None and loaded_models:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)
    st.write("🔍 Classifying...")

    processed_image = preprocess_image(image)
    model_predictions = get_ensemble_predictions(processed_image, loaded_models)

    st.subheader("📊 Individual Model Predictions")
    model_labels = []
    for i, prediction in enumerate(model_predictions):
        confidence = prediction if prediction > 0.5 else (1 - prediction)
        label = 'real' if prediction > 0.7 else 'fake'
        model_labels.append(label)
        st.metric(label=f"Model {i+1}", value=f"{label.capitalize()}", delta=f"{confidence:.2%}")

    # 🧬 Majority voting
    label_counts = Counter(model_labels)
    final_prediction = label_counts.most_common(1)[0][0]

    st.subheader("🧬 Ensemble Verdict")
    st.markdown(f"**Final Prediction:** The image is **{final_prediction.upper()}**.")

elif uploaded_file is not None and not loaded_models:
    st.error("🚫 No models loaded. Please check your model files.")