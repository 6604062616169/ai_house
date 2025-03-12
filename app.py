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

# Placeholder for X_train (replace this with the actual training data or column names)
# Here we assume X_train is a dataframe with column names as required by your model.
# In practice, you'd load this data from a source or ensure it's available
X_train_columns = ['GrLivArea', 'BedroomAbvGr', 'FullBath']  # Replace with actual columns from your model

# Prepare the DataFrame for prediction
input_data = pd.DataFrame(np.zeros((1, len(X_train_columns))), columns=X_train_columns)

# Set input values into the DataFrame
input_data.loc[0, 'GrLivArea'] = sqft  # ขนาดพื้นที่
input_data.loc[0, 'BedroomAbvGr'] = bedrooms  # ห้องนอน
input_data.loc[0, 'FullBath'] = bathrooms  # ห้องน้ำ

# Prediction button
if st.button("📌 Predict Price"):
    try:
        # Predict the house price
        predicted_price = model.predict(input_data)[0]
        st.success(f"🏡 ราคาบ้านที่คาดการณ์: ${predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
