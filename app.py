
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Lokasi file
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MODEL_PATH = BASE_DIR / "model_hewan_efficientnet.keras"
CLASS_PATH = BASE_DIR / "class_names_efficientnet.json"

IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="Klasifikasi Jenis Hewan",
    page_icon="🐾",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

st.title("🐾 Klasifikasi Jenis Hewan")
st.write(
    "Unggah gambar hewan untuk memperoleh hasil prediksi "
    "menggunakan model CNN EfficientNet."
)

try:
    model = load_model()
    class_names = load_class_names()
except Exception as error:
    st.error(f"Gagal memuat model atau daftar kelas: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Pilih gambar hewan",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", width=400)

    if st.button("Prediksi Hewan", type="primary"):
        with st.spinner("Sedang menganalisis gambar..."):
            resized_image = image.resize(IMAGE_SIZE)
            image_array = np.asarray(resized_image, dtype=np.float32)
            input_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(input_array, verbose=0)
            scores = np.asarray(predictions)[0]

            if scores.ndim != 1 or len(scores) != len(class_names):
                st.error(
                    "Jumlah output model tidak sesuai dengan daftar kelas."
                )
                st.stop()

            # Mengurutkan tiga skor tertinggi
            top_indices = np.argsort(scores)[::-1][:3]
            best_index = int(top_indices[0])

            st.subheader("Hasil Prediksi")
            st.success(f"Jenis hewan: {class_names[best_index]}")

            # Skor ditampilkan sebagai persentase.
            # Interpretasi sebagai probabilitas bergantung pada output model.
            for rank, index in enumerate(top_indices, start=1):
                score = float(scores[index])
                st.write(
                    f"{rank}. {class_names[int(index)]}: "
                    f"{score * 100:.2f}%"
                )
                st.progress(float(np.clip(score, 0, 1)))

            st.caption(
                "Skor model bukan jaminan kebenaran prediksi. "
                "Hasil perlu dibandingkan dengan label gambar sebenarnya."
            )
