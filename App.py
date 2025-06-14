import streamlit as st
import pandas as pd
import pickle

# === LOAD MODEL YANG SUDAH DITRAIN DAN DISIMPAN SEBELUMNYA ===
MODEL_PATH = 'model_terbaik.pkl'

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# === UI APLIKASI ===
st.title("Obesity Level Prediction App")
st.markdown("Masukkan data untuk memprediksi tingkat obesitas Anda.")

# === INPUT USER ===
age = st.number_input('Usia', 1, 120, 25)
height = st.number_input('Tinggi Badan (m)', 1.0, 2.5, 1.70)
weight = st.number_input('Berat Badan (kg)', 20, 300, 70)
fcvc = st.slider('Frekuensi makan sayur (1-3)', 1, 3, 2)
ncp = st.slider('Jumlah makan besar/hari (1-4)', 1, 4, 3)
ch2o = st.slider('Konsumsi air (1-3)', 1, 3, 2)
faf = st.slider('Aktivitas fisik (0-3)', 0, 3, 1)
tue = st.slider('Waktu layar (0-2)', 0, 2, 1)

gender = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
family_history = st.selectbox('Riwayat keluarga overweight?', ['yes', 'no'])
favc = st.selectbox('Sering makanan tinggi kalori?', ['yes', 'no'])
caec = st.selectbox('Ngemil di antara waktu makan?', ['no', 'Sometimes', 'Frequently', 'Always'])
smoke = st.selectbox('Merokok?', ['yes', 'no'])
scc = st.selectbox('Pantau kalori?', ['yes', 'no'])
calc = st.selectbox('Konsumsi alkohol?', ['no', 'Sometimes', 'Frequently', 'Always'])
mtrans = st.selectbox('Transportasi utama?', ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'])

# === BENTUK DATAFRAME DARI INPUT ===
input_df = pd.DataFrame([{
    'Age': age,
    'Height': height,
    'Weight': weight,
    'FCVC': fcvc,
    'NCP': ncp,
    'CH2O': ch2o,
    'FAF': faf,
    'TUE': tue,
    'Gender': gender,
    'family_history_with_overweight': family_history,
    'FAVC': favc,
    'CAEC': caec,
    'SMOKE': smoke,
    'SCC': scc,
    'CALC': calc,
    'MTRANS': mtrans
}])

# === PREDIKSI ===
if st.button("Prediksi"):
    try:
        prediction = model.predict(input_df)
        st.subheader("Hasil Prediksi:")
        st.success(f"Tingkat Obesitas Anda: **{prediction[0]}**")
    except Exception as e:
        st.error(f"Gagal melakukan prediksi. Pesan error: {e}")
