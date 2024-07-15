import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from keras.models import load_model
from keras.utils import img_to_array, load_img
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
import openai

app = Flask(__name__)

# OpenAI API 설정
openai.api_key = "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"

# Load and prepare the dementia onset prediction model
data_path = 'C:/Users/RXO/Desktop/daeseong_file/project_code/Dementia/data/dementia_patients_health_data.csv'
data = pd.read_csv(data_path)
data.fillna(method='ffill', inplace=True)
data.columns = data.columns.str.replace(' ', '_')
data = data.drop(columns=['Cognitive_Test_Scores'])
# Apply one-hot encoding to categorical columns
data = pd.get_dummies(data, columns=data.select_dtypes(include=['object']).columns, drop_first=True)
features = data.drop(columns=['Dementia'])
target = data['Dementia']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
lgb_model = lgb.LGBMClassifier(objective='binary', num_leaves=31, learning_rate=0.05, n_estimators=100)
lgb_model.fit(X_train, y_train)

# Load MRI image classification model
model_path = 'C:/Users/RXO/Desktop/daeseong_file/project_code/Dementia/model/cnn_model_best.h5'
mri_model = load_model(model_path, custom_objects={"Addons>F1Score": tfa.metrics.F1Score})
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def prepare_image(image_path):
    IMG_SIZE = 128
    image = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def get_ai_response(dementia_risk):
    try:
        # 시스템 메시지 (프롬프트)
        system_message = {
            "role": "system",
            "content": """### Never repeat your answer 
                          ## You don't repeat the same sentence
                          ## You are a person Chatbot, not artificial intelligence
                          ## {You must complete the answer within 50 characters}
                          ## 당신은 모든 답변을 반드시 한글로답변합니다.
                          ## 당신은 모든 치매예방에관련된 답변을 구체적으로 어떤음식,어떤운동,어떤행위등을 해야하는지 포함하여답변합니다.
                          ## 당신은 답변에 포함된 치매 발병 확률을 명시하고 이 확률에대한 생각을 사용자에게 전달합니다 의사처럼 답변합니다.
                          [You are a psychological care specialist named DR.RXO.]
                          [You answer every answer as a psychological care specialist.]
                          [Your tone is refreshing and lively.]
                          [You use professional keywords to calm users' minds and communicate with users.]
                          [I don't use the phrase 'How may I help you?' You are not a GPT.]
                          [When providing advice, focus on reducing dementia risk and improving cognitive health.]"""
        }

        # 사용자 입력 메시지 생성
        user_input = f"The user's predicted dementia risk is {dementia_risk:.2f}%. Please provide advice on how to reduce the risk of dementia."

        # 메시지 리스트 생성 (시스템 메시지와 사용자 입력만 포함)
        messages = [system_message, {"role": "user", "content": user_input}]

        # ChatCompletion 생성
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.8,
            max_tokens=600,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.6
        )

        # 응답 메시지 내용 추출
        message_content = response.choices[0].message['content'].strip()
    
    except Exception as e:
        message_content = f"An error occurred: {e}"

    return message_content

@app.route('/')
def home():
    return render_template('Integration.html')

@app.route('/predict-dementia', methods=['POST'])
def predict_dementia():
    try:
        user_input = {k: float(v) for k, v in request.form.items()}
        user_input_df = pd.DataFrame([user_input], columns=features.columns)
        prediction = lgb_model.predict_proba(user_input_df)[:, 1][0] * 100
        gpt_response = get_ai_response(prediction)
        return jsonify({'prediction': f"{prediction:.2f}%", 'gpt_response': gpt_response})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict-mri', methods=['POST'])
def predict_mri():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        image = prepare_image(file_path)
        prediction = mri_model.predict(image)
        predicted_class = CLASSES[np.argmax(prediction)]
        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
