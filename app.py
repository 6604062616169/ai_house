import streamlit as st
import numpy as np
import joblib

# Load the model with error handling
try:
    model = joblib.load('house_price_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("🏡 House Price Prediction")

# Input fields
sqft = st.number_input("🏠 ขนาดพื้นที่ (ตร.ฟุต)", min_value=500, max_value=10000, value=1500)
bedrooms = st.number_input("🛏 จำนวนห้องนอน", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("🛁 จำนวนห้องน้ำ", min_value=1, max_value=10, value=2)

# Prediction button
if st.button("📌 Predict Price"):
    input_data = np.array([[sqft, bedrooms, bathrooms]])
    try:
        predicted_price = model.predict(input_data)[0]
        st.success(f"🏡 ราคาบ้านที่คาดการณ์: ${predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
