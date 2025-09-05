import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import os

# ------------------- App Config -------------------
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="✏️", layout="wide")

# ------------------- Custom CSS -------------------
st.markdown(
    """
    <style>
    .main {
        background-color: transparent;
    }
    h1 {
        color: #2E86C1;
        text-align: center;
    }
    .card {
        background: transparent;   /* no background */
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.25);
        margin-bottom: 20px;
    }
    .prediction {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------- Title -------------------
st.title("✏️ MNIST Digit Classifier")
st.markdown("<p style='text-align:center'>Draw or upload a digit (0–9) and let the CNN model predict it 🚀</p>", unsafe_allow_html=True)

# ------------------- Load Model -------------------
MODEL_PATH =tf.keras.models.load_model("saved_models/cnn_mnist.keras")


@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found at {MODEL_PATH}. Please train and save it first.")
    st.stop()

model = load_cnn_model()

# ------------------- Preprocessing + Prediction -------------------
def preprocess_and_predict(img_pil, model):
    original = img_pil.resize((100, 100))  # for preview

    img = img_pil.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)

    preds = model.predict(arr)[0]
    top_class = np.argmax(preds)
    confidence = preds[top_class]

    return preds, top_class, confidence, original, img

# ------------------- Layout -------------------
col1, col2, col3 = st.columns([1, 1, 1])
img_input = None

# ---- Card 1: Drawing ----
with col1:
    st.markdown("<div class='card'><h3>🎨 Draw a Digit</h3>", unsafe_allow_html=True)
    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=15,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if canvas.image_data is not None:
        img_input = Image.fromarray((canvas.image_data).astype("uint8")).convert("RGB")

# ---- Card 2: Upload ----
with col2:
    st.markdown("<div class='card'><h3>📤 Upload an Image</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        img_input = Image.open(uploaded_file)

# ---- Card 3: Prediction ----
with col3:
    st.markdown("<div class='card prediction'><h3>🔮 Prediction</h3>", unsafe_allow_html=True)

    if img_input is not None:
        preds, top_class, confidence, orig_preview, processed_preview = preprocess_and_predict(img_input, model)

        st.metric("Predicted Digit", top_class, f"{confidence:.2%}")

        # Input previews
        st.image([orig_preview, processed_preview], caption=["Original Input", "Processed (28x28)"], width=100)

        # Probability chart
        fig, ax = plt.subplots(figsize=(3, 2.5))
        bars = ax.bar(np.arange(10), preds, color="#95A5A6")
        bars[top_class].set_color("#2E86C1")  # Highlight prediction in blue
        ax.set_xticks(np.arange(10))
        ax.set_xlabel("Digit")
        ax.set_ylabel("Probability")
        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.info("✍️ Draw or upload a digit to see prediction.")

    st.markdown("</div>", unsafe_allow_html=True)
