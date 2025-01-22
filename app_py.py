import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Biaya Asuransi",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    color: #0083B8;
}
.highlight-box {
    background-color: #F0F2F6;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.stButton>button {
    color: white;
    background-color: #0083B8;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    font-size: 16px;
}
.stButton>button:hover {
    background-color: #005274;
}
</style>
""", unsafe_allow_html=True)

# Load model dan scaler
try:
    with open('model_uas.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open('scaler_uas.pkl', 'rb') as scaler_file:
        sc = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("File model atau scaler tidak ditemukan.")
    st.stop()

# Header dengan ikon
st.markdown("""
<h1 style='text-align: center; color: #0083B8;'>
    🏥 Prediktor Biaya Asuransi Kesehatan
</h1>
""", unsafe_allow_html=True)

# Informasi pembuat
st.markdown("""
<div class='highlight-box'>
<p class='big-font'>
📝 Dibuat oleh: Ryno Pahlevi Al Ghiffari (NIM: 2021230039)
</p>
<p>
Aplikasi cerdas untuk memprediksi biaya asuransi kesehatan berdasarkan faktor-faktor kesehatan personal.
</p>
</div>
""", unsafe_allow_html=True)

# Input Fields dengan desain yang lebih menarik
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Informasi Personal")
    age = st.slider('Umur', 
                    min_value=18, 
                    max_value=64, 
                    value=30,
                    help="Geser untuk memilih umur")
    
    sex = st.radio('Jenis Kelamin', 
                   ['Perempuan', 'Laki-laki'],
                   horizontal=True)
    
    bmi = st.number_input('Indeks Massa Tubuh (BMI)', 
                          min_value=15.0, 
                          max_value=50.0, 
                          value=25.0,
                          help="Masukkan BMI dengan presisi")

with col2:
    st.markdown("### 🔍 Detail Tambahan")
    children = st.number_input('Jumlah Anak', 
                                min_value=0, 
                                max_value=5, 
                                value=0,
                                help="Jumlah anak yang Anda miliki")
    
    smoker = st.radio('Status Merokok', 
                      ['Tidak', 'Ya'],
                      horizontal=True,
                      help="Pilih status merokok Anda")

# Preprocessing input (sama seperti sebelumnya)
def preprocess_input(age, sex, bmi, children, smoker):
    sex_map = {'Perempuan': 0, 'Laki-laki': 1}
    smoker_map = {'Tidak': 0, 'Ya': 1}

    input_data = np.array([
        age,
        sex_map[sex],
        bmi,
        children,
        smoker_map[smoker]
    ]).reshape(1, -1)

    input_data_scaled = sc.transform(input_data)
    return input_data_scaled

# Tombol Prediksi dengan desain
col_pred_button = st.columns(3)
with col_pred_button[1]:
    predict_button = st.button('🔮 Prediksi Biaya Asuransi')

# Proses Prediksi
if predict_button:
    try:
        input_scaled = preprocess_input(age, sex, bmi, children, smoker)
        prediction = model.predict(input_scaled)

        # Tampilan hasil dengan desain
        st.markdown("### 📈 Hasil Prediksi")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.metric(
                label="Estimasi Biaya Asuransi Tahunan",
                value=f"${prediction[0]:,.2f}"
            )
        
        with result_col2:
            if prediction[0] < 5000:
                st.success("💡 Biaya asuransi Anda relatif rendah.")
            elif prediction[0] < 10000:
                st.warning("⚠️ Biaya asuransi Anda dalam kisaran menengah.")
            else:
                st.error("🚨 Biaya asuransi Anda relatif tinggi.")
    
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam prediksi: {e}")

# Footer dengan desain
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: grey;'>
**Catatan:** Prediksi ini bersifat estimasi dan tidak mengikat. 
Konsultasikan dengan ahli untuk informasi lebih lanjut.
</p>
""", unsafe_allow_html=True)
