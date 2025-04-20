import os
import numpy as np
from flask import Flask, render_template, request, url_for
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# โหลด TFLite model
interpreter = Interpreter(model_path="mien_fabric_classifier_forapp.tflite")
interpreter.allocate_tensors()

# ดึง input/output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels และลิงก์ YouTube
CLASS_LABELS = ['Mien_pattern_01', 'Mien_pattern_02', 'Mien_pattern_03', 'Mien_pattern_04']
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

            # โหลดภาพและเตรียมข้อมูล
            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

            # รัน TFLite model
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            predicted_index = np.argmax(output_data)
            result = CLASS_LABELS[predicted_index]
            youtube_link = YOUTUBE_LINKS.get(result)

    return render_template('index1.html', result=result, filename=filename, youtube_link=youtube_link)

if __name__ == '__main__':
    app.run(debug=True)
