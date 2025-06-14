import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Load model, encoder, dan fitur
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

with open("feature_columns.json", "r") as f:
    expected_columns = json.load(f)

st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("🚶‍♂️ Prediksi Tingkat Obesitas Berdasarkan Data Pribadi 🍔")
st.markdown("Masukkan data pribadi Anda untuk memprediksi tingkat obesitas menggunakan model machine learning.")

# Form input
with st.form("input_form"):
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    age = st.slider("Umur", 10, 100, 25)
    height = st.number_input("Tinggi Badan (meter)", min_value=1.0, max_value=2.5, value=1.7)
    weight = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=70.0)
    family_history = st.radio("Riwayat keluarga obesitas?", ["yes", "no"])
    FAVC = st.radio("Sering konsumsi makanan tinggi kalori?", ["yes", "no"])
    FCVC = st.slider("Porsi sayur saat makan (1–3)", 1, 3, 2)
    NCP = st.slider("Jumlah makan besar per hari", 1, 4, 3)
    CAEC = st.selectbox("Camilan antar waktu?", ["no", "Sometimes", "Frequently", "Always"])
    SMOKE = st.radio("Merokok?", ["yes", "no"])
    CH2O = st.slider("Minum air putih per hari (liter)", 0.0, 3.0, 1.0)
    SCC = st.radio("Mengontrol asupan kalori?", ["yes", "no"])
    FAF = st.slider("Aktivitas fisik mingguan (jam)", 0.0, 3.0, 1.0)
    TUE = st.slider("Waktu depan layar per hari (jam)", 0.0, 4.0, 2.0)
    CALC = st.selectbox("Konsumsi alkohol?", ["no", "Sometimes", "Frequently", "Always"])
    MTRANS = st.selectbox("Transportasi utama", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
    submit = st.form_submit_button("🔎 Prediksi")

# Prediksi jika tombol ditekan
if submit:
    input_data = {
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': 1 if family_history == 'yes' else 0,
        'FAVC': 1 if FAVC == 'yes' else 0,
        'FCVC': FCVC,
        'NCP': NCP,
        'CAEC_Always': int(CAEC == 'Always'),
        'CAEC_Frequently': int(CAEC == 'Frequently'),
        'CAEC_Sometimes': int(CAEC == 'Sometimes'),
        'CAEC_no': int(CAEC == 'no'),
        'SMOKE': 1 if SMOKE == 'yes' else 0,
        'CH2O': CH2O,
        'SCC': 1 if SCC == 'yes' else 0,
        'FAF': FAF,
        'TUE': TUE,
        'CALC_Always': int(CALC == 'Always'),
        'CALC_Frequently': int(CALC == 'Frequently'),
        'CALC_Sometimes': int(CALC == 'Sometimes'),
        'CALC_no': int(CALC == 'no'),
        'MTRANS_Automobile': int(MTRANS == 'Automobile'),
        'MTRANS_Bike': int(MTRANS == 'Bike'),
        'MTRANS_Motorbike': int(MTRANS == 'Motorbike'),
        'MTRANS_Public_Transportation': int(MTRANS == 'Public_Transportation'),
        'MTRANS_Walking': int(MTRANS == 'Walking'),
        'Gender_Female': int(gender == 'Female'),
        'Gender_Male': int(gender == 'Male')
    }

    df_input = pd.DataFrame([input_data])

    # Tambahkan fitur yang belum ada
    for col in expected_columns:
        if col not in df_input.columns:
            df_input[col] = 0

    # Urutkan kolom sesuai model
    df_input = df_input[expected_columns]

    # Prediksi angka → ubah ke label
    pred_num = model.predict(df_input)[0]
    pred_label = le.inverse_transform([pred_num])[0]

    st.success(f"📊 Prediksi Tingkat Obesitas Anda: **{pred_label}**")

    # Buat 1 baris data uji dari input
test_data = {
    'Age': 21,
    'Height': 1.62,
    'Weight': 64,
    'family_history_with_overweight': 0,
    'FAVC': 0,
    'FCVC': 2,
    'NCP': 3,
    'CAEC_no': 1,
    'CAEC_Sometimes': 0,
    'CAEC_Frequently': 0,
    'CAEC_Always': 0,
    'SMOKE': 0,
    'CH2O': 2,
    'SCC': 1,
    'FAF': 0,
    'TUE': 1,
    'CALC_no': 0,
    'CALC_Sometimes': 1,
    'CALC_Frequently': 0,
    'CALC_Always': 0,
    'MTRANS_Public_Transportation': 1,
    'MTRANS_Walking': 0,
    'MTRANS_Automobile': 0,
    'MTRANS_Motorbike': 0,
    'MTRANS_Bike': 0,
    'Gender_Female': 1,
    'Gender_Male': 0
}

df_test = pd.DataFrame([test_data])
df_test = df_test[expected_columns]
result = model.predict(df_test)
print(le.inverse_transform(result)[0])

