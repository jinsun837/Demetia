import os
from flask import Flask, request, render_template, jsonify
from keras.models import load_model
from keras.utils import img_to_array, load_img
import tensorflow_addons as tfa
import numpy as np

app = Flask(__name__)
model_path = 'C:/Users/RXO/Desktop/daeseong_file/project_code/Dementia/model/cnn_model.h5'
# 모델을 로드할 때 사용자 정의 객체를 등록
model = load_model(model_path, custom_objects={"Addons>F1Score": tfa.metrics.F1Score})

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def prepare_image(image_path):
    IMG_SIZE = 128
    image = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

@app.route('/')
def home():
    return render_template('mri_index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        image = prepare_image(file_path)
        prediction = model.predict(image)
        predicted_class = CLASSES[np.argmax(prediction)]
        prediction_probabilities = prediction[0]  # 확률값을 출력하기 위해 추가

        return jsonify({'prediction': predicted_class, 'probabilities': prediction_probabilities.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
