{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyO1OnxXM/+PjDQ+38sYAz52",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ucoxxx/timeseries/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "xRNRbnHmwELr"
      },
      "outputs": [],
      "source": [
        "import streamlit as st\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from tensorflow.keras.models import load_model\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "import joblib\n",
        "\n",
        "# Fungsi untuk membuat dataset yang bisa digunakan oleh model RNN\n",
        "def create_dataset(data, time_step=1):\n",
        "    X = []\n",
        "    for i in range(len(data) - time_step):\n",
        "        a = data[i:(i + time_step), 0]\n",
        "        X.append(a)\n",
        "    return np.array(X)\n",
        "\n",
        "# Fungsi untuk memuat model dan scaler\n",
        "def load_model_and_scaler():\n",
        "    model = load_model('gru_model.h5')\n",
        "    scaler = joblib.load('scaler.pkl')\n",
        "    return model, scaler\n",
        "\n",
        "# Fungsi untuk membuat prediksi\n",
        "def predict(data, model, scaler, time_step):\n",
        "    data = scaler.transform(data)\n",
        "    dataX = create_dataset(data, time_step)\n",
        "    dataX = dataX.reshape((dataX.shape[0], dataX.shape[1], 1))\n",
        "    predictions = model.predict(dataX)\n",
        "    predictions = scaler.inverse_transform(predictions)\n",
        "    return predictions\n",
        "\n",
        "# Judul aplikasi\n",
        "st.title('Prediksi Time Series dengan GRU')\n",
        "\n",
        "# Mengunggah file CSV\n",
        "uploaded_file = st.file_uploader(\"Upload file CSV\", type=[\"csv\"])\n",
        "\n",
        "if uploaded_file is not None:\n",
        "    data = pd.read_csv(uploaded_file)\n",
        "    st.write(\"Data yang diunggah:\")\n",
        "    st.write(data.head())\n",
        "\n",
        "    # Memilih kolom untuk prediksi\n",
        "    column_name = st.selectbox(\"Pilih kolom untuk prediksi\", data.columns)\n",
        "\n",
        "    # Menyiapkan data untuk prediksi\n",
        "    time_step = st.slider(\"Time Step\", min_value=1, max_value=100, value=10)\n",
        "    input_data = data[[column_name]].values\n",
        "\n",
        "    # Memuat model dan scaler\n",
        "    model, scaler = load_model_and_scaler()\n",
        "\n",
        "    # Membuat prediksi\n",
        "    if st.button(\"Prediksi\"):\n",
        "        predictions = predict(input_data, model, scaler, time_step)\n",
        "        st.write(\"Prediksi:\")\n",
        "        st.write(predictions)\n",
        "\n",
        "        # Plot hasil prediksi\n",
        "        st.write(\"Plot Prediksi:\")\n",
        "        plt.figure(figsize=(10, 6))\n",
        "        plt.plot(input_data[time_step:], label='Data Asli')\n",
        "        plt.plot(predictions, label='Prediksi')\n",
        "        plt.legend()\n",
        "        st.pyplot(plt)\n"
      ]
    }
  ]
}