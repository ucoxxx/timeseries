import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

# Fungsi untuk membuat dataset yang bisa digunakan oleh model RNN
def create_dataset(data, time_step=1):
    X = []
    for i in range(len(data) - time_step):
        a = data[i:(i + time_step), 0]
        X.append(a)
    return np.array(X)

# Fungsi untuk memuat model dan scaler
def load_model_and_scaler():
    model = load_model('gru_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# Fungsi untuk membuat prediksi
def predict(data, model, scaler, time_step):
    data = scaler.transform(data)
    dataX = create_dataset(data, time_step)
    dataX = dataX.reshape((dataX.shape[0], dataX.shape[1], 1))
    predictions = model.predict(dataX)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Judul aplikasi
st.title('Prediksi Time Series dengan GRU')

# Mengunggah file CSV
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data yang diunggah:")
    st.write(data.head())

    # Memilih kolom untuk prediksi
    column_name = st.selectbox("Pilih kolom untuk prediksi", data.columns)

    # Menyiapkan data untuk prediksi
    time_step = st.slider("Time Step", min_value=1, max_value=100, value=10)
    input_data = data[[column_name]].values

    # Memuat model dan scaler
    model, scaler = load_model_and_scaler()

    # Membuat prediksi
    if st.button("Prediksi"):
        predictions = predict(input_data, model, scaler, time_step)
        st.write("Prediksi:")
        st.write(predictions)

        # Plot hasil prediksi
        st.write("Plot Prediksi:")
        plt.figure(figsize=(10, 6))
        plt.plot(input_data[time_step:], label='Data Asli')
        plt.plot(predictions, label='Prediksi')
        plt.legend()
        st.pyplot(plt)
