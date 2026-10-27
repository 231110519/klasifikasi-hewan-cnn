
# Klasifikasi Jenis Hewan Menggunakan CNN EfficientNet

## Deskripsi
Aplikasi berbasis web untuk mengklasifikasikan gambar hewan
menggunakan model deep learning CNN dengan arsitektur EfficientNet.

Pengguna dapat mengunggah gambar hewan dan melihat hasil prediksi
beserta tiga kelas dengan skor tertinggi.

## Fitur
- Unggah gambar hewan dalam format JPG, JPEG, dan PNG.
- Pratinjau gambar yang diunggah.
- Prediksi jenis hewan.
- Menampilkan tiga prediksi teratas beserta skornya.

## Teknologi
- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow

## Struktur Berkas
- `app.py`: kode aplikasi web.
- `requirements.txt`: daftar dependensi Python.
- `model_hewan_efficientnet.keras`: model CNN terlatih.
- `class_names_efficientnet.json`: daftar nama kelas hewan.

## Menjalankan Aplikasi

Instal dependensi:

```bash
python -m pip install -r requirements.txt
```

Jalankan aplikasi:

```bash
python -m streamlit run app.py
```

## Catatan
Model dan daftar kelas harus tersedia sesuai lokasi berkas
yang digunakan dalam kode aplikasi.
