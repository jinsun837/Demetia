import gradio as gr
import openai
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# OpenAI API 설정
openai.api_key = "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"

# 데이터 로드 및 모델 학습 부분
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

# Gradio 인터페이스 정의
def chatbot_interface(Diabetic, AlcoholLevel, HeartRate, BloodOxygenLevel, BodyTemperature, Weight, MRI_Delay, Prescription, Dosage_in_mg, Age, Smoking_Status, APOE_ε4, Physical_Activity, Depression_Status, Medication_History, Nutrition_Diet, Sleep_Quality, Chronic_Health_Conditions):
    user_input = {
        "Diabetic": Diabetic,
        "AlcoholLevel": AlcoholLevel,
        "HeartRate": HeartRate,
        "BloodOxygenLevel": BloodOxygenLevel,
        "BodyTemperature": BodyTemperature,
        "Weight": Weight,
        "MRI_Delay": MRI_Delay,
        "Prescription": Prescription,
        "Dosage_in_mg": Dosage_in_mg,
        "Age": Age,
        "Smoking_Status": Smoking_Status,
        "APOE_ε4": APOE_ε4,
        "Physical_Activity": Physical_Activity,
        "Depression_Status": Depression_Status,
        "Medication_History": Medication_History,
        "Nutrition_Diet": Nutrition_Diet,
        "Sleep_Quality": Sleep_Quality,
        "Chronic_Health_Conditions": Chronic_Health_Conditions
    }
    
    # 치매 발병률 예측
    dementia_risk = predict_dementia(user_input)
    # GPT-4를 통한 조언 생성
    response = get_ai_response(dementia_risk)
    return f"치매 발병 확률: {dementia_risk:.2f}%\n\nDR.RXO: {response}"

iface = gr.Interface(
    fn=chatbot_interface,
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
    title="RXO Brain Pro(뇌 건강을 위한 종합 솔루션)",
    description="이 챗봇은 치매 발병률을 기반으로한 DR.RXO의 치매진단 및 예방을 제공합니다."
)

# Gradio 인터페이스 실행
if __name__ == "__main__":
    iface.launch()
