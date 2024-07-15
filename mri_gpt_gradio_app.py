import os
from PIL import Image
import numpy as np
import pandas as pd
import gradio as gr
import lightgbm as lgb
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
import tensorflow_addons as tfa
import openai

# 데이터 로드 및 모델 학습
data = pd.read_csv('C:/Users/RXO/Desktop/daeseong_file/project_code/Dementia/data/dementia_patients_health_data.csv')
data.fillna(method='ffill', inplace=True)
data.columns = data.columns.str.replace(' ', '_')
data = data.drop(columns=['Cognitive_Test_Scores'])

label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

features = data.drop(columns=['Dementia'])
target = data['Dementia']

params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'max_depth': -1,
    'min_child_samples': 20,
    'min_gain_to_split': 0.01,
    'force_col_wise': True,
}

cv_results = lgb.cv(
    params,
    lgb.Dataset(features, label=target),
    num_boost_round=100,
    nfold=5,
    stratified=True,
    shuffle=True,
    metrics='binary_logloss',
    callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(10)],
    seed=42
)

best_num_boost_round = len(cv_results['valid binary_logloss-mean'])

model = lgb.train(
    params,
    lgb.Dataset(features, label=target),
    num_boost_round=best_num_boost_round
)

# OpenAI API 설정
openai.api_key = os.getenv("OPENAI_API_KEY", "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")

def predict_dementia(user_input):
    user_input_df = pd.DataFrame([user_input], columns=features.columns)
    for column, le in label_encoders.items():
        if column in user_input_df.columns:
            valid_labels = list(le.classes_)
            user_input_df[column] = user_input_df[column].apply(lambda x: x if x in valid_labels else valid_labels[0])
            user_input_df[column] = le.transform(user_input_df[column])

    prediction = model.predict(user_input_df)
    prediction_percentage = prediction[0] * 100
    return prediction_percentage

def get_gpt_response(prediction_percentage):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a psychological care specialist."},
                {"role": "user", "content": f"The user's predicted dementia risk is {prediction_percentage:.2f}%. Please provide advice on how to reduce the risk of dementia."}
            ],
            temperature=0.8,
            max_tokens=300
        )
        message_content = response.choices[0].message['content'].strip()
    except Exception as e:
        message_content = f"An error occurred: {e}"
    return message_content

# MRI 모델 설정
model_path = 'C:/Users/RXO/Desktop/daeseong_file/project_code/Dementia/model/cnn_model.h5'
mri_model = load_model(model_path, custom_objects={"Addons>F1Score": tfa.metrics.F1Score})

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def prepare_image(image):
    IMG_SIZE = 128
    # 이미지 사이즈 조정과 array 변환을 처리
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def mri_predict(image):
    image_prepared = prepare_image(image)
    prediction = mri_model.predict(image_prepared)
    predicted_class = CLASSES[np.argmax(prediction)]
    prediction_probabilities = {CLASSES[i]: float(prediction[0][i] * 100) for i in range(len(CLASSES))}
    
    # 로그 출력
    print(f"Predicted Class: {predicted_class}")
    print(f"Probabilities: {prediction_probabilities}")

    return predicted_class, prediction_probabilities

def mri_interface(image):
    try:
        print("Received image:", image)  # 이미지 수신 확인
        predicted_class, probabilities = mri_predict(image)
        probabilities_df = pd.DataFrame(list(probabilities.items()), columns=['Class', 'Probability'])
        print("Prediction successful:", predicted_class, probabilities)  # 예측 성공 로그
        return predicted_class, probabilities_df
    except Exception as e:
        print("Error in prediction:", str(e))  # 예측 실패 시 로그
        return "Error in prediction: " + str(e), None



# Gradio 인터페이스 정의
def dementia_interface(diabetic, alcohol_level, heart_rate, blood_oxygen_level, body_temperature, weight, mri_delay, prescription, dosage_in_mg, age, smoking_status, apoe_e4, physical_activity, depression_status, medication_history, nutrition_diet, sleep_quality, chronic_health_conditions):
    user_input = {
        "Diabetic": diabetic,
        "AlcoholLevel": alcohol_level,
        "HeartRate": heart_rate,
        "BloodOxygenLevel": blood_oxygen_level,
        "BodyTemperature": body_temperature,
        "Weight": weight,
        "MRI_Delay": mri_delay,
        "Prescription": prescription,
        "Dosage_in_mg": dosage_in_mg,
        "Age": age,
        "Smoking_Status": smoking_status,
        "APOE_ε4": apoe_e4,
        "Physical_Activity": physical_activity,
        "Depression_Status": depression_status,
        "Medication_History": medication_history,
        "Nutrition_Diet": nutrition_diet,
        "Sleep_Quality": sleep_quality,
        "Chronic_Health_Conditions": chronic_health_conditions
    }
    prediction_percentage = predict_dementia(user_input)
    gpt_response = get_gpt_response(prediction_percentage)
    return f"치매 발병 확률: {prediction_percentage:.2f}%\n\nDR.RXO: {gpt_response}"

def mri_interface(image):
    predicted_class, probabilities = mri_predict(image)
    probabilities_df = pd.DataFrame(list(probabilities.items()), columns=['Class', 'Probability'])
    return predicted_class, probabilities_df

chatbot = gr.Interface(
    fn=dementia_interface,
    inputs=[
        gr.Number(label="Diabetic (당뇨병 여부)"),
        gr.Number(label="Alcohol Level (알코올 수치)"),
        gr.Number(label="Heart Rate (심박수)"),
        gr.Number(label="Blood Oxygen Level (혈중 산소 수치)"),
        gr.Number(label="Body Temperature (체온)"),
        gr.Number(label="Weight (체중)"),
        gr.Number(label="MRI Delay (MRI 촬영까지 지연시간)"),
        gr.Number(label="Prescription (처방받은 약물 개수)"),
        gr.Number(label="Dosage in mg (약물 복용량)"),
        gr.Number(label="Age (나이)"),
        gr.Number(label="Smoking Status (흡연 상태)"),
        gr.Number(label="APOE ε4 (유전자 보유 여부)"),
        gr.Number(label="Physical Activity (신체 활동 수준)"),
        gr.Number(label="Depression Status (우울증 상태)"),
        gr.Number(label="Medication History (약물 복용 이력)"),
        gr.Number(label="Nutrition Diet (영양 섭취 상태)"),
        gr.Number(label="Sleep Quality (수면의 질)"),
        gr.Number(label="Chronic Health Conditions (만성 질환 보유 여부)")
    ],
    outputs="text",
    title="RXO 치매 발병 확률 예측 및 심리 케어",
    description="이 챗봇은 RXO의 심리케어 전문가 관점에서 답변합니다. 질문을 입력해보세요."
)

mri = gr.Interface(
    fn=mri_interface,
    inputs=gr.Image(type="pil", label="Upload MRI Image"),
    outputs=[
        gr.Textbox(label="Predicted Class"),
        gr.Dataframe(label="Probabilities")
    ],
    title="MRI 이미지 기반 치매 단계 예측",
    description="MRI 이미지를 업로드하면 치매 단계를 예측합니다."
)

app = gr.TabbedInterface(
    [chatbot, mri],
    ["치매 발병 확률 예측", "MRI 이미지 기반 치매 단계 예측"]
)

# Gradio 인터페이스 실행
if __name__ == "__main__":
    app.launch()
