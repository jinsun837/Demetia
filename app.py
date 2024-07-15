import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, render_template, jsonify
from lightgbm.callback import early_stopping, log_evaluation

app = Flask(__name__)

# 데이터 로드
data = pd.read_csv('C:/Users/RXO/Desktop/daeseong_file/project_code/Dementia/data/dementia_patients_health_data.csv')
# 데이터 로드: dementia_patients_health_data.csv 파일을 읽어와 data 데이터프레임에 저장합니다.

# 결측치 처리
data.fillna(method='ffill', inplace=True)
# 결측치 처리: 결측치를 앞의 값으로 채웁니다.

# 피처 이름의 공백을 언더스코어로 변경
data.columns = data.columns.str.replace(' ', '_')
# 피처 이름 정리: 피처 이름의 공백을 언더스코어로 변경합니다.

# 'Cognitive_Test_Scores' 특성 제거
data = data.drop(columns=['Cognitive_Test_Scores'])
# 특정 피처 제거: Cognitive_Test_Scores 피처를 제거합니다.

# 카테고리형 데이터 인코딩
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
# 카테고리형 데이터 인코딩: 문자열 데이터를 숫자로 변환합니다.

# 특징 선택
features = data.drop(columns=['Dementia'])
target = data['Dementia']
# 특징과 타겟 분리: 예측에 사용할 피처와 타겟 변수를 분리합니다.

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# 데이터 분할: 학습 데이터와 테스트 데이터를 8:2 비율로 나눕니다.

# 모델 파라미터 설정
params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'binary_logloss',
    'num_leaves': 31,  # num_leaves를 적절히 설정 (예: 31)
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'max_depth': -1,  # max_depth를 적절히 설정 (예: -1로 무제한)
    'min_child_samples': 20,
    'min_gain_to_split': 0.01,
    'force_col_wise': True,  # force_col_wise 설정
}
# 모델 파라미터 설정: LightGBM 모델의 파라미터를 설정합니다.

# LightGBM 모델 학습을 위한 교차 검증 수행
cv_results = lgb.cv(
    params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=100,
    nfold=5,
    stratified=True,
    shuffle=True,
    metrics='binary_logloss',
    callbacks=[early_stopping(stopping_rounds=10), log_evaluation(10)],
    seed=42
)
# 교차 검증: 5-fold 교차 검증을 수행하여 모델의 성능을 검증하고 최적의 부스팅 라운드 수를 결정합니다.

# 최적의 부스팅 라운드 수
best_num_boost_round = len(cv_results['valid binary_logloss-mean'])

# 최적의 부스팅 라운드 수로 최종 모델 학습
model = lgb.train(
    params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=best_num_boost_round
)
# 모델 학습: 최적의 부스팅 라운드 수로 최종 모델을 학습합니다.

# 사용자 입력을 받아 치매 예측 확률 계산하는 함수
def predict_dementia(user_input):
    user_input_df = pd.DataFrame([user_input], columns=features.columns)
    
    # 카테고리형 데이터 인코딩
    for column, le in label_encoders.items():
        if column in user_input_df.columns:
            # 새로운 라벨이 있으면 가장 유사한 기존 라벨로 대체
            valid_labels = list(le.classes_)
            user_input_df[column] = user_input_df[column].apply(lambda x: x if x in valid_labels else valid_labels[0])
            user_input_df[column] = le.transform(user_input_df[column])
    
    prediction = model.predict(user_input_df)
    prediction_percentage = prediction[0] * 100  # 퍼센트로 변환
    return prediction_percentage

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form.to_dict()
    # Convert numeric values to appropriate types
    for key in user_input:
        try:
            user_input[key] = float(user_input[key])
        except ValueError:
            pass

    prediction_percentage = predict_dementia(user_input)
    return jsonify({'prediction': prediction_percentage})

if __name__ == '__main__':
    app.run(debug=True)
