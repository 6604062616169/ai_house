import streamlit as st
import numpy as np
import joblib

# โหลดโมเดลที่บันทึกไว้
model = joblib.load('house_price_model.pkl')

st.title("🏡 House Price Prediction")

# กรอกข้อมูลบ้าน
sqft = st.number_input("🏠 ขนาดพื้นที่ (ตร.ฟุต)", min_value=500, max_value=10000, value=1500)
bedrooms = st.number_input("🛏 จำนวนห้องนอน", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("🛁 จำนวนห้องน้ำ", min_value=1, max_value=10, value=2)

# ปุ่มทำนาย
if st.button("📌 Predict Price"):
    input_data = np.array([[sqft, bedrooms, bathrooms]])
    predicted_price = model.predict(input_data)[0]
    st.success(f"🏡 ราคาบ้านที่คาดการณ์: ${predicted_price:,.2f}")
