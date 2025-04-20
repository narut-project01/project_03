from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import gdown

# --- Class Labels ---
CLASS_LABELS = ['Mien_pattern_01', 'Mien_pattern_02', 'Mien_pattern_03', 'Mien_pattern_04']

YOUTUBE_LINKS = {
    'Mien_pattern_01': 'https://youtu.be/zkrltLG0r9w',
    'Mien_pattern_02': 'https://youtu.be/example_for_02',
    'Mien_pattern_03': 'https://youtu.be/example_for_03',
    'Mien_pattern_04': 'https://youtu.be/example_for_04'
}

# --- Download model from Google Drive if not exist ---
MODEL_ID = '1AB3tFMw6K8iL-RnGt0TUwQlvA53-ccvd'
MODEL_PATH = 'mien_fabric_classifier_forapp.h5'

if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(f'https://drive.google.com/uc?id={MODEL_ID}', MODEL_PATH, quiet=False)
    print("✅ Model downloaded.")

# --- Load model ---
model = load_model(MODEL_PATH)

# --- Create Flask App ---
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Mien Fabric Classifier API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = np.array(data['input']).reshape(1, -1)  # Adjust input shape if needed
        predictions = model.predict(input_data)
        
        # Get predicted class index
        predicted_index = np.argmax(predictions[0])
        predicted_label = CLASS_LABELS[predicted_index]
        youtube_link = YOUTUBE_LINKS[predicted_label]

        return jsonify({
            'predicted_index': predicted_index + 1,  # for 1-based class (1 to 4)
            'predicted_label': predicted_label,
            'youtube_link': youtube_link
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
