import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ตั้งค่าหน้าตาของเว็บ
st.set_page_config(page_title="AI Web App", layout="wide")

# ใช้ session_state เพื่อจัดการหน้า
if "page" not in st.session_state:
    st.session_state.page = "Machine Learning"

# ฟังก์ชันเปลี่ยนหน้า
def change_page(new_page):
    st.session_state.page = new_page

# แสดงปุ่มนำทาง (Navigation Buttons)
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🤖 Machine Learning"):
        change_page("Machine Learning")
with col2:
    if st.button("🏡 Demo Machine Learning 🇺🇸"):
        change_page("Demo Machine Learning")
with col3:
    if st.button("🧠 Neural Network"):
        change_page("Neural Network")
with col4:
    if st.button("🐱🐶 Demo Neural Network"):
        change_page("Demo Neural Network")

st.markdown("---")  # เส้นคั่นหน้า

# 🟢 หน้า Machine Learning
if st.session_state.page == "Machine Learning":
    st.title("🤖 Machine Learning: House Price Prediction")

    st.subheader("ข้อมูล Dataset ที่ใช้")
    st.write(
        """
        **Kaggle House Prices Dataset**
        - ตอนแรกได้เริ่มทำการหาข้อมูลผ่าน ChatGPT และได้คำแนะนำเกี่ยวกับเว็บไซต์ Kaggle ค่ะ
        - อันนี้ข้อมูลเกี่ยวกับราคาบ้านจากเมือง Ames, Iowa, USA ใช้สำหรับสร้างโมเดลพยากรณ์ราคาบ้าน  
        -  data source: [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)  
        """
    )
    st.subheader("ฟีเจอร์ที่ใช้ในโมเดล")
    st.write(
        """
        ฟีเจอร์หลักที่ใช้ในการพยากรณ์ราคาบ้าน :
        - **LotArea** (ขนาดที่ดิน, ตารางฟุต)  
        - **BedroomAbvGr** (จำนวนห้องนอน)  
        - **FullBath** (จำนวนห้องน้ำ)  
        - **SalePrice** (ราคาขายบ้าน)  
        """
    )
    st.subheader("ขั้นตอนการเตรียมข้อมูลใน Google Colab")
    st.write(
        " **อัปโหลด Dataset**\n"
        "```python\n"
        "from google.colab import files \n"
        "uploaded = files.upload() \n"
        "```\n"
        "ใช้ `files.upload()` เพื่ออัปโหลดไฟล์ ZIP จากเครื่อง \n"

        " **แตกไฟล์ ZIP**\n"
        "```python\n"
        "!unzip house.zip \n"
        "```\n"
        "ใช้ `!unzip house.zip` เพื่อแตกไฟล์\n"

        " **โหลดข้อมูล**\n"
        "```python\n"
        "train_data = pd.read_csv('train.csv')\n"
        "test_data = pd.read_csv('test.csv')\n"
        "```\n"
        "- โหลดไฟล์ `train.csv` และ `test.csv` เพื่อใช้งาน\n"
    )

    st.subheader("การจัดการข้อมูลที่ขาดหาย (Missing Data)")
    st.write(
        """
        - ใช้ `.isnull().sum()` เพื่อตรวจสอบข้อมูลที่ขาดหาย  
        - แก้ไขข้อมูลที่ขาดหายด้วย:
        ```python
        for col in train_data.columns:
            if train_data[col].isnull().sum() > 0:
                if train_data[col].dtype == 'object':
                    train_data[col].fillna("None", inplace=True)
                else:
                    train_data[col].fillna(train_data[col].median(), inplace=True)
        ```
        - เติมค่า `None` ให้กับข้อมูลที่เป็นข้อความ  
        - เติมค่ามัธยฐาน (Median) ให้กับข้อมูลตัวเลข  
        """
    )

    st.subheader("One-Hot Encoding และการทำให้ Train/Test มีคอลัมน์ตรงกัน")
    st.write(
        """
        - ใช้ One-Hot Encoding แปลงข้อมูลที่เป็นข้อความเป็นตัวเลข  
        - ปรับให้ `train_data` และ `test_data` มีคอลัมน์ตรงกัน  
        ```python
        train_data, test_data = train_data.align(test_data, join='left', axis=1, fill_value=0)
        ```
        """
    )

    st.subheader("การ Train โมเดล RandomForest")
    st.write(
        """
        - ใช้ `RandomForestRegressor` ในการ Train โมเดล  
        ```python
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        ```
        - ทดสอบโมเดลด้วย Mean Absolute Error (MAE)
        ```python
        y_pred = model.predict(X_valid)
        mae = mean_absolute_error(y_valid, y_pred)
        print(f'Mean Absolute Error: {mae}')
        ```
        """
    )

    st.subheader("การบันทึกโมเดลและโหลดโมเดล")
    st.write(
        """
        - ใช้ `joblib` เพื่อบันทึกโมเดล
        ```python
        joblib.dump(model, 'house_price_model.pkl')
        ```
        - โหลดโมเดลขึ้นมาใช้งานใหม่
        ```python
        model = joblib.load('house_price_model.pkl')
        ```
        """
    )

    st.subheader("ทำนายราคาบ้านตัวอย่าง")
    st.write(
        """
        - ทดสอบพยากรณ์ราคาบ้านโดยใช้ข้อมูลจำลอง  
        ```python
        sample_data = pd.DataFrame(np.zeros((1, X_train.shape[1])), columns=X_train.columns)
        sample_data.loc[0, 'GrLivArea'] = 1000
        sample_data.loc[0, 'BedroomAbvGr'] = 2
        sample_data.loc[0, 'FullBath'] = 1

        predicted_price = model.predict(sample_data)[0]
        print(f"🏡 ราคาบ้านที่คาดการณ์: ${predicted_price:,.2f}")
        ```
        """
    )

    st.success("กระบวนการสร้างโมเดลเสร็จสมบูรณ์!")

# 🟡 หน้า Demo Machine Learning
elif st.session_state.page == "Demo Machine Learning":
    try:
        model = joblib.load('house_price_model.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    st.title("🏡 House Price Prediction in us 🇺🇸 ")
    sqft = st.number_input("🏠 ขนาดพื้นที่ (ตร.ฟุต)", min_value=500, max_value=10000, value=1000)
    bedrooms = st.number_input("🛏 จำนวนห้องนอน", min_value=1, max_value=10, value=1)
    bathrooms = st.number_input("🛁 จำนวนห้องน้ำ", min_value=1, max_value=10, value=1)

    # เตรียมข้อมูลให้ตรงกับฟีเจอร์ของโมเดล
    model_features = model.feature_names_in_
    input_data = pd.DataFrame(np.zeros((1, len(model_features))), columns=model_features)
    input_data.loc[0, 'GrLivArea'] = sqft
    input_data.loc[0, 'BedroomAbvGr'] = bedrooms
    input_data.loc[0, 'FullBath'] = bathrooms

    if st.button("📌 Predict Price"):
        try:
            predicted_price = model.predict(input_data)[0]
            st.success(f"🏡 ราคาบ้านที่คาดการณ์: ${predicted_price:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
# 🔵 หน้า Neural Network
elif st.session_state.page == "Neural Network":
    st.title("🧠 Neural Network")
    st.write("ขั้นตอนแรกได้ไปหาdata set บนเว็พ kiggle และค้าหาที่ฮิดที่สุด เลยเจอ dataset cat or dogเลยคิดว่าแยกหมากับแมวก็ดูไม่ได้ง่ายขนาดนั้นเพราะมีค่อนข้างหลายสายพันธ์และค่อนข้างต่างกันมาก https://www.kaggle.com/datasets/tongpython/cat-and-dog")
    st.image("https://img5.pic.in.th/file/secure-sv1/1c2f764fab9cfb22a.png", width=600, use_container_width=False)
    st.write("และขั้นต่อมาทำการอัพ api ของ kiggle ขึ้น google colab ทำให้โหลดไฟร์zib เข้าได้ไวขึ้น และทำการแตกไฟร์ต่อ")
    st.image("https://img5.pic.in.th/file/secure-sv1/27d9df771fce3f3ad.png", width=600, use_container_width=False)
    st.write("ใช้ ImageDataGenerator จาก TensorFlow เพื่อเตรียมข้อมูลภาพสำหรับการฝึกโมเดล Machine Learning โดยเฉพาะการจำแนกภาพ (Image Classification) แบบ Binary Classification")
    st.image("https://img2.pic.in.th/pic/33d29ee08dd6fd275.png", width=600, use_container_width=False)
    st.write("สร้างโมเดล Convolutional Neural Network (CNN) สำหรับงาน Binary Classification ")
    st.write("3 ชั้น สำหรับดึง Feature จากภาพ")
    st.write("ลดขนาด Feature Map")
    st.write("2 ชั้น สำหรับการจำแนกข้อมูล")
    st.write("ใช้ Sigmoid Activation เพื่อให้ผลลัพธ์เป็นความน่าจะเป็น (0 หรือ 1)ทำให้โมเดลนี้พร้อมสำหรับการฝึกด้วยข้อมูลภาพที่เตรียมไว้")
    st.image("https://img2.pic.in.th/pic/418afd8ea65b7e17a.png", width=600, use_container_width=False)
    st.write("โค้ดนี้ใช้สำหรับ ฝึกโมเดล (Training model)")
    st.image("https://img2.pic.in.th/pic/Screenshot-from-2025-03-12-23-45-52.png", width=600, use_container_width=False)
    st.write("codeจะplotกราฟจะช่วยให้เห็นการเปลี่ยนแปลงของ Loss และ Accuracy ระหว่างการฝึก และตรวจสอบว่าโมเดล Overfitting หรือไม่")
    st.image("https://img2.pic.in.th/pic/6491366b23e31efd5.png", width=600, use_container_width=False)
    st.write("จากรูปโมเดลค่อนข้างเรียนรู้ได้ดี ไม่ overfittingจนเกินไป")
    st.image("https://img5.pic.in.th/file/secure-sv1/810e34025f882f14e.png", width=600, use_container_width=False)
    st.write("codeนี้ใช้ลองทดสอบว่าทำงานได้ไหมแยกหมาแมวเบื่องต้นได้ไหมและ savemodel ใน .h57")

# 🔴 หน้า Demo Neural Network (Cat vs Dog Classifier)
elif st.session_state.page == "Demo Neural Network":
    try:
        model = load_model('cat_dog_classifier.h5')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    st.title("🐱🐶 Cat vs Dog Classifier")
    st.write("Upload an image of a cat or dog, and the AI will classify it!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(150, 150))
        st.image(img, caption='Uploaded Image', use_column_width=True)

        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)

        if prediction[0] > 0.5:
            st.write("**Prediction:** This is a Dog! 🐶")
        else:
            st.write("**Prediction:** This is a Cat! 🐱")
