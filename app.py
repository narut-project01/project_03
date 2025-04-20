import os
import numpy as np
import gdown
from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ดาวน์โหลดโมเดลจาก Google Drive หากยังไม่มี
MODEL_ID = '1AB3tFMw6K8iL-RnGt0TUwQlvA53-ccvd'
MODEL_PATH = 'mien_fabric_classifier_forapp.h5'

if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(f'https://drive.google.com/uc?id={MODEL_ID}', MODEL_PATH, quiet=False)
    print("✅ Model downloaded.")

# โหลดโมเดล
model = load_model(MODEL_PATH)
print("✅ Model loaded.")

# ชื่อคลาส
CLASS_LABELS = ['Mien_pattern_01', 'Mien_pattern_02', 'Mien_pattern_03', 'Mien_pattern_04']

# ลิงก์ YouTube ตามคลาส
YOUTUBE_LINKS = {
    'Mien_pattern_01': 'https://youtu.be/zkrltLG0r9w',
    'Mien_pattern_02': 'https://youtu.be/example_for_02',
    'Mien_pattern_03': 'https://youtu.be/example_for_03',
    'Mien_pattern_04': 'https://youtu.be/example_for_04'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    filename = None
    youtube_link = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # ปรับขนาดภาพให้ตรงกับ input ของโมเดล (แก้ให้ตรงกับของคุณ)
            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # ทำนายผล
            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            result = CLASS_LABELS[predicted_index]
            youtube_link = YOUTUBE_LINKS.get(result)

    return render_template('index.html', result=result, filename=filename, youtube_link=youtube_link)

if __name__ == '__main__':
    app.run(debug=True)
