# ============================================================
# 🌈 PosterScan Web App – Patch-based AI Detection (REVISED)
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
# 🧠 BUILD MODEL (same as Colab)
# ============================================================
def build_binary_model(backbone_ctor, img_size):
    base_model = backbone_ctor(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    return model


# ============================================================
# ⚙️ Load Weights Safely (Keras 3 friendly)
# ============================================================
@st.cache_resource
def load_cnn_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "MobileNetV3Large_scenario2.h5")

    if not os.path.exists(model_path):
        st.error(f"❌ File weights tidak ditemukan: {model_path}")
        st.stop()

    model = build_binary_model(MobileNetV3Large, 224)
    model.load_weights(model_path)

    return model


model = load_cnn_model()


# ============================================================
# 🔧 PATCHING (identik Colab)
# ============================================================
def split_patches(img_array, num_patches_per_side):
    patches = []
    h, w, _ = img_array.shape
    patch_h = h // num_patches_per_side
    patch_w = w // num_patches_per_side

    for i in range(0, h - patch_h + 1, patch_h):
        for j in range(0, w - patch_w + 1, patch_w):
            patch = img_array[i:i+patch_h, j:j+patch_w, :]
            if patch.shape[:2] == (patch_h, patch_w):
                patches.append(patch)

    return np.array(patches)


# ============================================================
# 🎯 PATCH-BASED SOFT VOTING
# ============================================================
def predict_patch_voting(img_array):
    img_array = preprocess_input(img_array)

    patches = split_patches(img_array, GRID_SIZE)

    resized = tf.image.resize(patches, TARGET_SIZE).numpy()
    preds = model.predict(resized, verbose=0).reshape(-1)

    prob_ai = preds.mean()
    return prob_ai, preds, patches


# ============================================================
# 🖼️ Overlay Visualization
# ============================================================
def overlay_prediction(patches, preds, num_patches):
    plt.figure(figsize=(6, 6))
    gap = 0.05
    alpha = 0.45

    for i in range(num_patches):
        for j in range(num_patches):
            idx = i * num_patches + j
            prob_ai = preds[idx]   # ✅ FIX

            if prob_ai > 0.5:
                label = "AI"
                color = (1, 0, 0, alpha)
                text_color = "red"
            else:
                label = "Human"
                color = (0, 1, 0, alpha)
                text_color = "green"

            x_pos = j + j * gap
            y_pos = i + i * gap

            ax = plt.axes([
                x_pos / (num_patches + gap * (num_patches - 1)),
                1 - (y_pos + 1) / (num_patches + gap * (num_patches - 1)),
                1 / (num_patches + gap * (num_patches - 1)),
                1 / (num_patches + gap * (num_patches - 1))
            ])

            ax.imshow(patches[idx].astype("uint8"))
            ax.imshow(
                np.ones_like(patches[idx]) * np.array(color[:3]),
                alpha=color[3]
            )

            ax.text(
                5, 20,
                label,
                color=text_color,
                fontsize=12,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
            )

            ax.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf




# ============================================================
# 🎨 UI
# ============================================================
st.title("PosterScan")
st.caption("Deteksi Tingkat Keterlibatan AI pada Poster Digital")

uploaded = st.file_uploader("Upload Poster Digital", type=["jpg", "jpeg", "png"])

if uploaded:
    img = image.load_img(uploaded)
    img_array = image.img_to_array(img)

    st.image(uploaded, caption="Poster Digital", width=350)

    if st.button("Deteksi Poster"):
        with st.spinner("Menganalisis poster..."):
            prob_ai, preds, patches = predict_patch_voting(img_array)
            buf = overlay_prediction(patches, preds, GRID_SIZE)

        st.subheader("Hasil Deteksi")

        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded, caption="Original")
        with col2:
            st.image(buf, caption="Patch Prediction")

        ai_percent = prob_ai * 100
        st.markdown(f"## {ai_percent:.0f}% AI Involvement")

        if 45 <= ai_percent <= 55:
            st.info("🟡 Seimbang AI & Human")
        elif ai_percent > 55:
            st.error("🔴 Dominan AI")
        else:
            st.success("🟢 Dominan Human")
