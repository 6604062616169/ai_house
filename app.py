import pandas as pd
import numpy as np
import joblib
import streamlit as st

# โหลดโมเดลด้วยการจัดการข้อผิดพลาด
try:
    model = joblib.load('house_price_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("🏡 House Price Prediction")

# รับค่าอินพุตจากผู้ใช้
sqft = st.number_input("🏠 ขนาดพื้นที่ (ตร.ฟุต)", min_value=500, max_value=10000, value=1000)
bedrooms = st.number_input("🛏 จำนวนห้องนอน", min_value=1, max_value=10, value=1)
bathrooms = st.number_input("🛁 จำนวนห้องน้ำ", min_value=1, max_value=10, value=1)

# รับข้อมูลฟีเจอร์ทั้งหมดที่ใช้ฝึกฝนโมเดล
model_features = model.feature_names_in_

# สร้าง DataFrame ว่าง ที่มีฟีเจอร์ทั้งหมดที่โมเดลต้องการ
input_data = pd.DataFrame(np.zeros((1, len(model_features))), columns=model_features)

# ใส่ค่าที่ต้องการพยากรณ์
input_data.loc[0, 'GrLivArea'] = sqft   # พื้นที่ใช้สอย (ตร.ฟุต)
input_data.loc[0, 'BedroomAbvGr'] = bedrooms   # จำนวนห้องนอน
input_data.loc[0, 'FullBath'] = bathrooms   # จำนวนห้องน้ำ

# ทำนายราคาบ้าน
if st.button("📌 Predict Price"):
    try:
        # ทำนายราคาบ้าน
        predicted_price = model.predict(input_data)[0]
        st.success(f"🏡 ราคาบ้านที่คาดการณ์: ${predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
