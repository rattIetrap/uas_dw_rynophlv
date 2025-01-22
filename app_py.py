import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load model dan scaler yang sudah disimpan sebelumnya
try:
    with open('model_uas.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open('scaler_uas.pkl', 'rb') as scaler_file:
        sc = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("File model atau scaler tidak ditemukan. Pastikan file sudah benar.")
    st.stop()

# Judul Aplikasi
st.title('Prediksi Biaya Asuransi Kesehatan')
st.write('Dibuat oleh: Ryno Pahlevi Al Ghiffari (NIM: 2021230039)')

# Tambahkan deskripsi singkat
st.markdown("""
    Aplikasi ini memprediksi biaya asuransi kesehatan berdasarkan beberapa faktor:
    - Umur
    - Jenis Kelamin
    - Indeks Massa Tubuh (BMI)
    - Jumlah Anak
    - Status Merokok
""")

# Input Fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Umur', 
                           min_value=18, 
                           max_value=64, 
                           value=30,
                           help="Masukkan umur Anda antara 18-64 tahun")
    
    sex = st.selectbox('Jenis Kelamin', 
                       ['Perempuan', 'Laki-laki'],
                       help="Pilih jenis kelamin Anda")
    
    bmi = st.number_input('Indeks Massa Tubuh (BMI)', 
                          min_value=15.0, 
                          max_value=50.0, 
                          value=25.0,
                          help="Masukkan Indeks Massa Tubuh (BMI) Anda")

with col2:
    children = st.number_input('Jumlah Anak', 
                                min_value=0, 
                                max_value=5, 
                                value=0,
                                help="Masukkan jumlah anak yang Anda miliki")
    
    smoker = st.selectbox('Status Merokok', 
                          ['Tidak', 'Ya'],
                          help="Pilih status merokok Anda")

# Preprocessing input
def preprocess_input(age, sex, bmi, children, smoker):
    # Konversi kategorikal ke numerik
    sex_map = {'Perempuan': 0, 'Laki-laki': 1}
    smoker_map = {'Tidak': 0, 'Ya': 1}

    # Siapkan data input
    input_data = np.array([
        age,
        sex_map[sex],
        bmi,
        children,
        smoker_map[smoker]
    ]).reshape(1, -1)

    # Gunakan scaler yang sudah disimpan
    input_data_scaled = sc.transform(input_data)

    return input_data_scaled

# Tombol Prediksi
if st.button('Prediksi Biaya Asuransi'):
    try:
        # Preprocessing
        input_scaled = preprocess_input(age, sex, bmi, children, smoker)

        # Prediksi
        prediction = model.predict(input_scaled)

        # Tampilkan hasil
        st.success(f'Estimasi Biaya Asuransi Tahunan: ${prediction[0]:,.2f}')
        
        # Tambahkan interpretasi
        if prediction[0] < 5000:
            st.info("Biaya asuransi Anda relatif rendah.")
        elif prediction[0] < 10000:
            st.info("Biaya asuransi Anda dalam kisaran menengah.")
        else:
            st.warning("Biaya asuransi Anda relatif tinggi.")
    
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam prediksi: {e}")

# Footer
st.markdown("---")
st.markdown("**Catatan:** Prediksi ini bersifat estimasi dan tidak mengikat.")
