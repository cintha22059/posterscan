# ============================================================
# 🌈 PosterScan Web App – Streamlit Patch-based Detection
# ============================================================
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# ============================================================
# ⚙️ Load Model
# ============================================================
@st.cache_resource
def load_cnn_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "MobileNetV3Large_scenario2.h5")

    if not os.path.exists(model_path):
        st.error(f"❌ File model tidak ditemukan di: {model_path}")
        st.stop()

    return load_model(model_path, compile=False)
model = load_cnn_model()
# ============================================================
# 🔧 Fungsi bantu
# ============================================================
def split_patches(img_array, num_patches_per_side=4):
    patches = []
    h, w, _ = img_array.shape
    patch_h = h // num_patches_per_side
    patch_w = w // num_patches_per_side
    for i in range(num_patches_per_side):
        for j in range(num_patches_per_side):
            y1, y2 = i * patch_h, (i + 1) * patch_h
            x1, x2 = j * patch_w, (j + 1) * patch_w
            patches.append(img_array[y1:y2, x1:x2, :])
    return np.array(patches)

def overlay_prediction(img_array, patches, preds, num_patches=4):
    plt.figure(figsize=(6, 6))
    gap = 0.05
    alpha = 0.45
    for i in range(num_patches):
        for j in range(num_patches):
            idx = i * num_patches + j
            val = preds[idx][0]
            color = (1, 0, 0, alpha) if val <= 0.5 else (0, 1, 0, alpha)
            x_pos = j + j * gap
            y_pos = i + i * gap
            ax = plt.axes([
                x_pos / (num_patches + gap * (num_patches - 1)),
                1 - (y_pos + 1) / (num_patches + gap * (num_patches - 1)),
                1 / (num_patches + gap * (num_patches - 1)),
                1 / (num_patches + gap * (num_patches - 1))
            ])
            ax.imshow(patches[idx].astype("uint8"))
            ax.imshow(np.ones_like(patches[idx]) * np.array(color[:3]), alpha=color[3])
            ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf

def get_base64_from_file(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ============================================================
# 🎨 CSS Styling
# ============================================================
page_bg = """
<style>
body {
    background: linear-gradient(90deg, #b7e1ec, #e7f3f8);
}
h1, h2, h3 {
    color: #012f41;
    text-align: center;
}
button, .stButton>button {
    background-color: #014c65;
    color: white;
    border-radius: 8px;
    font-weight: bold;
    padding: 0.5em 1.5em;
}
.patch-img {
    margin-top: 40px;   /* atur angka ini jika perlu */
}
.ai-box {
    margin-top: 200px;  
}
<style>
.ai-text {
    margin-top: 40px;   
}
.ai-label {
    margin-top: 10px;  
}
</style>

</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ============================================================
# 🏠 Halaman Awal
# ============================================================
st.markdown("<h1>PosterScan</h1>", unsafe_allow_html=True)
st.markdown("<h4>Website Deteksi Tingkat Keterlibatan Artificial Intelligence dan Manusia pada Poster Digital</h4>", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    col1, col2, col3 = st.columns([2, 1, 2]) 
    with col2: 
        if st.button("MULAI"): st.session_state.page = "deteksi" 
        st.rerun()
# ============================================================
# 📤 Halaman Deteksi
# ============================================================
elif st.session_state.page == "deteksi":
    uploaded = st.file_uploader("Upload Poster Digital", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
         st.image(uploaded, caption="Poster Digital", width=350)
        colA, colB, colC = st.columns([2, 1, 2])
        with colB:
            detect_btn = st.button("Deteksi Poster")


        if detect_btn:
            with st.spinner("Sedang menganalisis..."):
                # proses gambar
                img = image.load_img(uploaded)
                img_array = image.img_to_array(img)
                num_patches = 4
                patches = split_patches(img_array, num_patches)
                resized = np.array([tf.image.resize(p, (224, 224)) for p in patches]) / 255.0
                preds = model.predict(resized)
                ai = np.sum(preds <= 0.5)
                human = np.sum(preds > 0.5)
                total = len(preds)
                ai_percent = ai / total * 100
                human_percent = human / total * 100
                buf = overlay_prediction(img_array, patches, preds, num_patches)

            st.subheader("Results of AI Involvement Detection in Digital Posters")
            col1, col2, col3 = st.columns([1,1,1.2])
            with col1:
                st.image(uploaded, caption="Original Poster", use_container_width=True)
            with col2:
                st.markdown("<div class='patch-img'>", unsafe_allow_html=True)
                st.image(buf, caption="AI–Human Involvement Visualization", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<h1 style='text-align:center; color:red;'>{ai_percent:.0f}%</h1>", unsafe_allow_html=True)

                st.markdown("<p class='ai-text' style='text-align:center; font-size:18px;'>AI Involvement</p>", unsafe_allow_html=True)

                if 45 <= ai_percent <= 55:
                    st.markdown("<p class='ai-label'>🟡 Equally Generated by AI and Human</p>", unsafe_allow_html=True)
                elif ai_percent > 55:
                    st.markdown("<p class='ai-label'>🔴 Mostly Generated by AI</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p class='ai-label'>🟢 Mostly Generated by Human</p>", unsafe_allow_html=True)


    if st.button("⬅️ Kembali ke Halaman Awal"):
        st.session_state.page = "home"
        st.rerun()



