# tempat pemrosesan machine learning ke app.py
import streamlit as st
import numpy as np
from sklearn.preprocessing import RobustScaler
import pandas as pd

# import ml package
import joblib
import os

attribute_info = """
                Penjelasan untuk tiap-tiap kolom :
                - Age: Umur dari individu.
                - Workclass: Tipe dari pekerjaan individu. Workclass ini berisi kategori seperti:
                    - Private: Bekerja di sektor private.
                    - Self-emp-not-inc: Individu yang bekerja sendiri dan tidak tergabung dalam sebuah perusahaan.
                    - Self-emp-inc: Individu yang bekerja sendiri dan tergabung dalam sebuah perusahaan.
                    - Federal-gov: Bekerja pada pemerintahan pusat.
                    - Local-gov: Bekerja pada pemerintahan lokal.
                    - State-gov: Bekerja pada pemerintahan negara bagian.
                    - Without-pay: Tidak bekerja atau bekerja tapi tidak diupah.
                    - Never-worked: Tidak pernah bekerja.
                - Wducation: Pendidikan Terakhir.
                - Education Number: Durasi dalam menyelesaikan pendidikan.
                - Marital Status: Status Pernikahan.
                - Occupation: Bidang pekerjaan atau jabatan.
                - Relationship: Status Hubungan.
                - Race: Ras dari individu.
                - Gender: Gender dari individu.
                - Capital Gain: Jumlah keuntungan modal(financial profit).
                - Capital Loss: Jumlah kerugian modal.
                - Hours per Week: Jumlah jam kerja per-minggu.
                - Native Country: Negara asal.
                - Final Weight: Bobot pada file CPS yang dikendalikan berdasarkan perkiraan independen terhadap populasi sipil non-institusional di AS. Ini dipersiapkan setiap bulan oleh Divisi Kependudukan di Biro Sensus, dengan menggunakan 3 set kontrol.
                - Income: Level gaji dari individu. terdiri dari 2 kategori yaitu gaji yang lebih dari 50,000 dollar dan gaji yang kurang dari atau sama dengan 50,000 dollar, keduanya ditulis dengan (>50K, <=50K).
                """


def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model


def run_ml_app():
    st.subheader("Machine Learning Section")
    with st.expander("Attribute Info"):
        st.markdown(attribute_info)

    st.subheader("Input Your Data")
    with st.form("my_data"):
        age = st.number_input("Age", 17, 90)
        workclass = st.selectbox("Workclass", ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov',
                                               'Other', 'Self-emp-inc', 'Without-pay', 'Never-worked'])
        finalWeight = st.number_input(
            "Final Weight", min_value=0, step=1)
        education = st.selectbox("Education", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm',
                                               'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', '5th-6th', '10th', '1st-4th' 'Preschool' '12th'])
        educationNum = st.number_input(
            "Education Number", 1, 16)
        maritalStatus = st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse',  'Divorced', 'Married-spouse-absent',
                                                        'Separated', 'Married-AF-spouse', 'Widowed'])
        occupation = st.selectbox("Occupation", ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service',
                                                 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support',
                                                 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'])
        relationship = st.selectbox("Relationship", [
                                    'Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])
        race = st.selectbox(
            "Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
        gender = st.radio('Gender', ['Male', 'Female'])
        capitalGain = st.number_input(
            "Capital Gain", min_value=0, step=1)
        capitalLoss = st.number_input(
            "Capital Loss", min_value=0, step=1)
        hoursPerWeek = st.number_input(
            "Hours per Week", 1, 99)
        nativeCountry = st.selectbox("Native Country", ['United-States', 'Cuba', 'Jamaica', 'India', 'Other', 'Mexico', 'South', 'Puerto-Rico',
                                                        'Honduras', 'England', 'Canada', 'Germany', 'Iran', 'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia',
                                                        'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic', 'El-Salvador', 'France',
                                                        'Guatemala', 'China', 'Japan', 'Yugoslavia', 'Peru', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
                                                        'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary', 'Holand-Netherlands'])

        submitted = st.form_submit_button("Submit")

    if submitted:
        with st.expander("Your Selected Options"):
            result = {
                "Age": age,
                "Workclass": workclass,
                "Final Weight": finalWeight,
                "Education": education,
                "EducationNum": educationNum,
                "Marital Status": maritalStatus,
                "Occupation": occupation,
                "Relationship": relationship,
                "Race": race,
                "Gender": gender,
                "Capital Gain": capitalGain,
                "capital loss": capitalLoss,
                "Hours per Week": hoursPerWeek,
                "Native Country": nativeCountry
            }
            st.write(result)

        df4 = pd.read_csv(os.path.join('df4.csv'))
        data_baru = result
        df_baru = pd.DataFrame(data_baru, index=[0])
        st.write("Your Selected Options :")
        st.table(df_baru)
        df_baru.drop(columns='Capital Gain', inplace=True)
        df_baru.drop(columns='capital loss', inplace=True)
        df_baru.drop(columns='Education', inplace=True)

        # cleaning
        df_baru['Native Country'] = df_baru['Native Country'].apply(lambda x: 'USA'
                                                                    if x == 'United-States' else 'Non-USA')
        df_baru["Marital Status"] = df_baru["Marital Status"].replace(
            ['Divorced', 'Separated', 'Widowed'], "Divorced")
        df_baru["Marital Status"] = df_baru["Marital Status"].replace(
            ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'], "Married")
        df_baru['Race'] = df_baru['Race'].apply(
            lambda x: 'White' if x == 'White' else 'Non-White')

        # encoding data kategorikal
        df_baru = pd.get_dummies(df_baru)
        for kolom in df4.columns:
            if kolom not in df_baru.columns:
                df_baru[kolom] = 0

        for kolom in df_baru.columns:
            if kolom not in df4.columns:
                # karena one hot (drop_first=True)
                df_baru.drop(columns=kolom, inplace=True)

        df_baru = df_baru[df4.columns]  # match column

        # scaling
        scaler = RobustScaler()
        df4_scaled = scaler.fit_transform(df4)
        df_baru_scaled = scaler.transform(df_baru)

        # prediction section
        st.subheader('Prediction Result')
        single_array = np.array(df_baru_scaled).reshape(1, -1)

        model = load_model("model_xgb.pkl")

        prediction = model.predict(single_array)

        if prediction == 0:
            st.info("""
                Hasil Prediksi Income : <= 50K \n
                Gaji kurang dari atau sama dengan 50,000 dollar
                """)
        elif prediction == 1:
            st.info("""
                Hasil Prediksi Income : > 50K \n
                Gaji lebih dari 50,000 dollar
                """)
