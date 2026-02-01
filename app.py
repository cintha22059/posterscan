# ============================================================
# 🌈 PosterScan Web App – Patch-based AI Detection (FINAL UI)
# ============================================================
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input


# ============================================================
# ⚙️ CONFIG
# ============================================================
TARGET_SIZE = (224, 224)
GRID_SIZE   = 4
THRESHOLD   = 0.5


# ============================================================
# 🧠 MODEL
# ============================================================
@st.cache_resource
def load_cnn_model():
    return models.load_model("MobileNetV3Large_scenario2.h5", compile=False)
model = load_cnn_model()


# ============================================================
# 🔧 PATCH
# ============================================================
def split_patches(img_array, num_patches):
    patches = []
    h, w, _ = img_array.shape
    ph, pw = h // num_patches, w // num_patches

    for i in range(num_patches):
        for j in range(num_patches):
            patches.append(img_array[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :])
    return np.array(patches)


def predict_patch_voting(img_array):
    img_array = preprocess_input(img_array)
    patches = split_patches(img_array, GRID_SIZE)
    resized = tf.image.resize(patches, TARGET_SIZE).numpy()

    preds = model.predict(resized, verbose=0).reshape(-1)
    binary = (preds > THRESHOLD).astype(int)
    return binary.mean(), binary, patches


# ============================================================
# 🖼️ VISUALIZATION (FULL OVERLAY ON ORIGINAL IMAGE)
# ============================================================
def overlay_on_full_image(original_img, binary_preds, num_patches):
    h, w, _ = original_img.shape
    overlay = original_img.copy().astype("float32")

    ph, pw = h // num_patches, w // num_patches

    idx = 0
    for i in range(num_patches):
        for j in range(num_patches):
            y1, y2 = i * ph, (i + 1) * ph
            x1, x2 = j * pw, (j + 1) * pw

            if binary_preds[idx]:  # AI
                color = np.array([255, 0, 0])  # 🔴 AI
                alpha = 0.4
            else:                  # Human
                color = np.array([0, 200, 0])  # 🟢 Human
                alpha = 0.35

            overlay[y1:y2, x1:x2] = (
                (1 - alpha) * overlay[y1:y2, x1:x2] + alpha * color
            )
            idx += 1

    overlay = overlay.astype("uint8")

    buf = BytesIO()
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf


def draw_ai_donut(ai_percent):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        [ai_percent, 100 - ai_percent],
        startangle=90,
        colors=["#e74c3c", "#2ecc71"],
        wedgeprops=dict(width=0.35)
    )
    ax.text(
        0, 0,
        f"{ai_percent:.0f}%",
        ha="center", va="center",
        fontsize=24, fontweight="bold"
    )
    ax.axis("equal")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close()
    buf.seek(0)
    return buf


# ============================================================
# 🎨 UI
# ============================================================
st.markdown(
    """
    <h1 style="text-align:center;">PosterScan</h1>
    <p style="text-align:center; color:gray;">
        Deteksi Tingkat Keterlibatan AI pada Poster Digital
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

uploaded = st.file_uploader("Upload Poster Digital", type=["jpg", "png", "jpeg"])

if uploaded:
    img = image.load_img(uploaded)
    img_array = image.img_to_array(img)

    # ===== PREVIEW CENTER =====
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.image(uploaded, caption="Poster Digital", width=360)

    # ===== BUTTON CENTER =====
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        detect = st.button("Deteksi Poster", use_container_width=True)

    if detect:
        with st.spinner("Menganalisis poster..."):
            ai_ratio, binary_preds, patches = predict_patch_voting(img_array)
            overlay = overlay_on_full_image(img_array, binary_preds, GRID_SIZE)

        ai_percent = ai_ratio * 100

        # ===== CENTER CONTAINER =====
        pad_l, main, pad_r = st.columns([1, 6, 1])

        with main:
            st.markdown(
                "<h3 style='text-align:center;margin-bottom:8px;'>Hasil Deteksi</h3>",
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns([1, 1, 0.8])

            with col1:
                st.image(uploaded, caption="Original", use_container_width=True)

            with col2:
                st.image(overlay, caption="Overlay AI–Human", use_container_width=True)

            with col3:
                st.image(draw_ai_donut(ai_percent), use_container_width=True)
                st.markdown(
                    f"<p style='text-align:center;font-weight:600;'>Tingkat Keterlibatan AI : {ai_percent:.0f}%</p>",
                    unsafe_allow_html=True
                )

        # ===== INTERPRETATION =====
        if ai_percent > 55:
            st.error("🔴 Dominan AI — mayoritas area terindikasi hasil generatif")
        elif ai_percent < 45:
            st.success("🟢 Dominan Human — ilustrasi mayoritas buatan manusia")
        else:
            st.info("🟡 Seimbang — kombinasi AI dan ilustrasi manusia")

        # ===== RESET BUTTON (CENTER & CONDITIONAL) =====
        _, reset_col, _ = st.columns([1, 2, 1])
        with reset_col:
            if st.button("Upload Poster Baru", use_container_width=True):
                st.experimental_rerun()
