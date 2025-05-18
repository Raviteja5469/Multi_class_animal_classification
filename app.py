import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Animal Classifier",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="auto"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
        html, body {
            background-color: #f5f7fa;
        }
        .title {
            font-family: 'Segoe UI', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            color: #3f3f3f;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #5a5a5a;
        }
        .stButton>button {
            background-color: #4a90e2;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-size: 1rem;
        }
        .result-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 16px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        .confidence-bar {
            background-color: #e0e0e0;
            height: 10px;
            border-radius: 5px;
        }
        .confidence-fill {
            background-color: #4a90e2;
            height: 10px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Load model
model = load_model("multiclass_animal_model.keras")

# Class names (from your notebook)
class_names = [
    'antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat',
    'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin',
    'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish',
    'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird',
    'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito',
    'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda',
    'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer',
    'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel',
    'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra'
]

# Title
st.markdown("<div class='title'>🐾 Animal Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image of an animal and let the model tell you what it is!</div>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","webp"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

        # Preprocessing
        image = image.resize((224, 224))
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        top_index = np.argmax(prediction[0])
        top_confidence = float(np.max(prediction[0])) * 100
        top_class = class_names[top_index]

        # Get top 3 predictions
        top_3_indices = np.argsort(prediction[0])[::-1][:3]
        top_3 = [(class_names[i], float(prediction[0][i]) * 100) for i in top_3_indices]

        # Show results
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown(f"### 🧠 Prediction: **{top_class}**")
        st.markdown(f"**Confidence:** {top_confidence:.2f}%")

        for label, conf in top_3:
            st.markdown(f"**{label}** — {conf:.2f}%")
            st.markdown(
                f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {conf}%;"></div>
                </div>
                """, unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
