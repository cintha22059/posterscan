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
def build_binary_model(backbone_ctor, img_size, unfreeze_ratio=0.2):
    base_model = backbone_ctor(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze semua layer dulu
    for layer in base_model.layers:
        layer.trainable = False

    # Hitung jumlah layer yang akan di-unfreeze
    total_layers = len(base_model.layers)
    unfreeze_from = int(total_layers * (1 - unfreeze_ratio))

    # Unfreeze 20% layer terakhir
    for layer in base_model.layers[unfreeze_from:]:
        # optional safety: hindari BatchNorm
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    return model


@st.cache_resource
def load_cnn_model():
    model = build_binary_model(MobileNetV3Large, 224)
    model.load_weights("MobileNetV3Large_scenario2.h5")
    return model


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
# 🖼️ VISUALIZATION
# ============================================================
def overlay_prediction(patches, binary_preds, num_patches):
    plt.figure(figsize=(6, 6))

    gap = 0.02
    cell = 1 / num_patches
    size = cell - gap

    for i in range(num_patches):
        for j in range(num_patches):
            idx = i * num_patches + j
            is_ai = binary_preds[idx]

            if is_ai:
                label = "AI"
                overlay_color = np.array([255, 0, 0])    # 🔴 AI
                alpha = 0.45
                text_color = "red"
            else:
                label = "Human"
                overlay_color = np.array([0, 200, 0])    # 🟢 Human
                alpha = 0.35
                text_color = "green"

            x = j * cell + gap / 2
            y = 1 - (i + 1) * cell + gap / 2
            ax = plt.axes([x, y, size, size])

            ax.imshow(patches[idx].astype("uint8"))

            overlay = np.ones_like(patches[idx]) * overlay_color
            ax.imshow(overlay.astype("uint8"), alpha=alpha)

            ax.text(
                6, 22,
                label,
                color=text_color,
                fontsize=10,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
            )

            ax.axis("off")

    buf = BytesIO()
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
            overlay = overlay_prediction(patches, binary_preds, GRID_SIZE)

        ai_percent = ai_ratio * 100

        # ===== CENTER CONTAINER =====
        pad_l, main, pad_r = st.columns([1, 6, 1])

        with main:
            st.markdown(
                "<h3 style='text-align:center;margin-bottom:8px;'>Hasil Deteksi</h3>",
                unsafe_allow_html=True
            )


        # ===== RESULT LAYOUT =====
        with main:
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


