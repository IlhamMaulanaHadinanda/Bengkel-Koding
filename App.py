import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ----------------------------
# Load model
# ----------------------------
with open("model_terbaik.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------------------
# LabelEncoder Mapping
# ----------------------------
label_encoders = {}
label_cols = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'CAEC', 'MTRANS']

encoder_maps = {
    'Gender': ['Female', 'Male'],
    'CALC': ['no', 'Sometimes', 'Frequently', 'Always'],
    'FAVC': ['no', 'yes'],
    'SCC': ['no', 'yes'],
    'SMOKE': ['no', 'yes'],
    'family_history_with_overweight': ['no', 'yes'],
    'CAEC': ['no', 'Sometimes', 'Frequently', 'Always'],
    'MTRANS': ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking']
}

for col, classes in encoder_maps.items():
    le = LabelEncoder()
    le.fit(classes)
    label_encoders[col] = le

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Obesity Level Prediction App")
st.markdown("Aplikasi ini memprediksi tingkat obesitas berdasarkan data input Anda.")

# Input
gender = st.selectbox('Jenis Kelamin', encoder_maps['Gender'])
age = st.number_input('Usia', min_value=1, max_value=120, value=25)
height = st.number_input('Tinggi Badan (m)', min_value=1.0, max_value=2.5, value=1.70)
weight = st.number_input('Berat Badan (kg)', min_value=20, max_value=300, value=70)
family_history = st.selectbox('Riwayat keluarga overweight?', encoder_maps['family_history_with_overweight'])
favc = st.selectbox('Sering makan makanan tinggi kalori?', encoder_maps['FAVC'])
fcvc = st.slider('Frekuensi makan sayur (1-3)', 1, 3, 2)
ncp = st.slider('Jumlah makan besar per hari', 1, 4, 3)
caec = st.selectbox('Ngemil?', encoder_maps['CAEC'])
smoke = st.selectbox('Merokok?', encoder_maps['SMOKE'])
scc = st.selectbox('Kontrol kalori?', encoder_maps['SCC'])
ch2o = st.slider('Konsumsi air (1-3)', 1, 3, 2)
faf = st.slider('Aktivitas fisik (0-3)', 0, 3, 1)
tue = st.slider('Waktu layar (0-2)', 0, 2, 1)
calc = st.selectbox('Konsumsi alkohol?', encoder_maps['CALC'])
mtrans = st.selectbox('Transportasi utama?', encoder_maps['MTRANS'])

# Buat DataFrame
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
# Preprocessing Manual
# ----------------------------
for col in label_cols:
    le = label_encoders[col]
    input_df[col] = le.transform(input_df[col])

# Standarisasi numerik
numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
scaler = StandardScaler()
input_df[numerical_cols] = scaler.fit_transform(input_df[numerical_cols])

# ----------------------------
# Prediksi
# ----------------------------
if st.button("Prediksi"):
    try:
        prediction = model.predict(input_df)
        st.success(f"Tingkat Obesitas Anda: **{prediction[0]}**")
    except Exception as e:
        st.error("Gagal melakukan prediksi.")
        st.error(str(e))
