import streamlit as st
import numpy as np
import cv2
import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection System")
st.write("Hybrid AI Model (EfficientNet + SVM)")

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    feature_extractor = load_model(
        "efficientnet_feature_extractor.keras",
        compile=False
    )
    svm_model = joblib.load(
        "svm_plant_disease.pkl"
    )
    return feature_extractor, svm_model


feature_extractor, svm_model = load_models()



CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# UI
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()), dtype=np.uint8
    )
    image = cv2.imdecode(file_bytes, 1)

    st.image(
        image,
        caption="Uploaded Image",
        use_column_width=True
    )

    if st.button("🔍 Predict Disease"):
        with st.spinner("Analyzing image..."):
            img_pre = preprocess_image(image)
            features = feature_extractor.predict(img_pre, verbose=0)
            prediction = svm_model.predict(features)[0]

            st.success(
                f"🧪 Prediction: **{CLASS_NAMES[prediction]}**"
            )

