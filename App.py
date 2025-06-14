import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Load Model
# ----------------------------
MODEL_PATH = 'model_terbaik.pkl'

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# ----------------------------
# Judul Aplikasi
# ----------------------------
st.title("Obesity Level Prediction Application")
st.markdown("Aplikasi ini memprediksi tingkat obesitas berdasarkan data yang Anda inputkan.")

# ----------------------------
# Input Pengguna
# ----------------------------
st.header("Masukkan Data Anda:")

gender = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
age = st.number_input('Usia', min_value=1, max_value=120, value=25)
height = st.number_input('Tinggi Badan (m)', min_value=1.0, max_value=2.5, value=1.70)
weight = st.number_input('Berat Badan (kg)', min_value=20, max_value=300, value=70)
family_history = st.selectbox('Riwayat Keluarga Overweight?', ['yes', 'no'])
favc = st.selectbox('Sering makan makanan tinggi kalori?', ['yes', 'no'])
fcvc = st.slider('Frekuensi makan sayur (1-3)', 1, 3, 2)
ncp = st.slider('Jumlah makan besar per hari', 1, 4, 3)
caec = st.selectbox('Ngemil?', ['no', 'Sometimes', 'Frequently', 'Always'])
smoke = st.selectbox('Merokok?', ['yes', 'no'])
scc = st.selectbox('Kontrol kalori?', ['yes', 'no'])
ch2o = st.slider('Konsumsi air (1-3)', 1, 3, 2)
faf = st.slider('Aktivitas fisik (0-3)', 0, 3, 1)
tue = st.slider('Waktu layar (0-2)', 0, 2, 1)
calc = st.selectbox('Konsumsi alkohol?', ['no', 'Sometimes', 'Frequently', 'Always'])
mtrans = st.selectbox('Transportasi utama?', [
    'Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'
])

# ----------------------------
# Buat DataFrame Input
# ----------------------------
input_df = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Height': [height],
    'Weight': [weight],
    'CALC': [calc],
    'FAVC': [favc],
    'FCVC': [fcvc],
    'NCP': [ncp],
    'SCC': [scc],
    'SMOKE': [smoke],
    'CH2O': [ch2o],
    'family_history_with_overweight': [family_history],
    'FAF': [faf],
    'TUE': [tue],
    'CAEC': [caec],
    'MTRANS': [mtrans]
})

# ----------------------------
# Prediksi
# ----------------------------
if st.button("Prediksi"):
    try:
        prediction = model.predict(input_df)
        st.success(f"Tingkat Obesitas Anda: **{prediction[0]}**")
    except Exception as e:
        st.error("Terjadi kesalahan saat melakukan prediksi.")
        st.error(str(e))
