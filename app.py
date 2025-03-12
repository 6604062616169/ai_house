import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model with error handling
try:
    model = joblib.load('house_price_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("🏡 House Price Prediction")

# Input fields for user input
sqft = st.number_input("🏠 ขนาดพื้นที่ (ตร.ฟุต)", min_value=500, max_value=10000, value=1500)
bedrooms = st.number_input("🛏 จำนวนห้องนอน", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("🛁 จำนวนห้องน้ำ", min_value=1, max_value=10, value=2)

# Add missing features (as placeholders or with user inputs)
first_flr_sf = st.number_input("พื้นที่ชั้น 1 (ตร.ฟุต)", min_value=0, value=800)
second_flr_sf = st.number_input("พื้นที่ชั้น 2 (ตร.ฟุต)", min_value=0, value=700)
porch_sf = st.number_input("พื้นที่ระเบียง (ตร.ฟุต)", min_value=0, value=100)

# Handling categorical features (e.g., Alley)
alley_grvl = st.selectbox("ชนิดของทางเดิน (Gravel)", ['Yes', 'No'])
alley_pave = st.selectbox("ชนิดของทางเดิน (Paved)", ['Yes', 'No'])

# Map categorical values to 1 or 0 (assuming one-hot encoding was applied during training)
alley_grvl = 1 if alley_grvl == 'Yes' else 0
alley_pave = 1 if alley_pave == 'Yes' else 0

# Placeholder for X_train (replace this with actual column names from your model)
X_train_columns = ['GrLivArea', 'BedroomAbvGr', 'FullBath', '1stFlrSF', '2ndFlrSF', '3SsnPorch', 'Alley_Grvl', 'Alley_Pave']

# Prepare the DataFrame for prediction
input_data = pd.DataFrame(np.zeros((1, len(X_train_columns))), columns=X_train_columns)

# Set input values into the DataFrame
input_data.loc[0, 'GrLivArea'] = sqft
input_data.loc[0, 'BedroomAbvGr'] = bedrooms
input_data.loc[0, 'FullBath'] = bathrooms
input_data.loc[0, '1stFlrSF'] = first_flr_sf
input_data.loc[0, '2ndFlrSF'] = second_flr_sf
input_data.loc[0, '3SsnPorch'] = porch_sf
input_data.loc[0, 'Alley_Grvl'] = alley_grvl
input_data.loc[0, 'Alley_Pave'] = alley_pave

# Prediction button
if st.button("📌 Predict Price"):
    try:
        # Predict the house price
        predicted_price = model.predict(input_data)[0]
        st.success(f"🏡 ราคาบ้านที่คาดการณ์: ${predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
