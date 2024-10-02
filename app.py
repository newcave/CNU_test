import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Streamlit 페이지 설정
st.title("Water Quality Prediction using Random Forest")
st.write("""
### Educational Demo using Streamlit
- Predict water quality index using basic meteorological data
- Modify the Random Forest `n_estimators` parameter using the sidebar
""")

# 1. Sidebar 설정: n_estimators 값을 선택할 수 있도록 구성
n_estimators = st.sidebar.selectbox(
    'Select number of trees in Random Forest (n_estimators)',
    options=[50, 100, 200],
    index=1  # 기본값: 100
)

# 2. 가상 기상 데이터 생성 (날짜별 기온, 이슬점온도, 운량, 광량, 강수량, 풍속)
date_range = pd.date_range(start='2024-10-01', periods=30, freq='D')
data = {
    'Date': date_range,
    'Temperature': np.random.uniform(15, 30, size=len(date_range)),  # 기온
    'DewPointTemp': np.random.uniform(10, 25, size=len(date_range)),  # 이슬점 온도
    'CloudCover': np.random.uniform(0, 1, size=len(date_range)),  # 운량 (0~1)
    'Radiation': np.random.uniform(100, 1000, size=len(date_range)),  # 광량
    'Precipitation': np.random.uniform(0, 20, size=len(date_range)),  # 강수량 (mm)
    'WindSpeed': np.random.uniform(0, 10, size=len(date_range)),  # 풍속 (m/s)
    'WaterQualityIndex': np.random.uniform(50, 100, size=len(date_range))  # 가상의 수질 지표
}

# 3. DataFrame으로 변환
df = pd.DataFrame(data)

# 4. 피처와 타겟 변수 설정
X = df[['Temperature', 'DewPointTemp', 'CloudCover', 'Radiation', 'Precipitation', 'WindSpeed']]
y = df['WaterQualityIndex']

# 5. 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 랜덤 포레스트 모델 학습
model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

# 7. 예측 수행
y_pred = model.predict(X_test)

# 8. 예측 결과 시각화
st.subheader("Model Performance")
mse = mean_squared_error(y_test, y_pred)
r2_score = model.score(X_test, y_test)
st.write(f"**Mean Squared Error (MSE)**: {mse:.2f}")
st.write(f"**R^2 Score**: {r2_score:.2f}")

# 9. 실제값과 예측값 비교 그래프
st.subheader("Actual vs Predicted Water Quality Index")

# matplotlib를 Streamlit에 표시하기 위한 작업
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_test.values, label='Actual Values', marker='o', linestyle='-', color='blue')
ax.plot(y_pred, label='Predicted Values', marker='x', linestyle='--', color='orange')
ax.set_xlabel('Test Data Points')
ax.set_ylabel('Water Quality Index')
ax.set_title('Actual vs Predicted Water Quality Index')
ax.legend()
st.pyplot(fig)

# 10. 변수 중요도 시각화 (옵션)
st.subheader("Feature Importance")
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
st.bar_chart(feature_importance)
