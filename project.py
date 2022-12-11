import pandas as pd
import streamlit as st
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


st.write(""" # APLIKASI PREDIKSI GAGAL JANTUNG """)
st.write("Oleh: Mochammad Rizki Aji Santoso (200411100086)")
st.write("----------------------------------------------------------------------------------")

home, deskripsi, datasets, preprocessing, modeling, implementasi = st.tabs(["Beranda", "Deskripsi", "Dataset", "Preprocessing", "Modelling", "Implementation"])
with home:
    st.write("""# Gagal Jantung""") #menampilkan halaman utama
    st.image('https://p2ptm.kemkes.go.id/uploads//VHcrbkVobjRzUDN3UCs4eUJ0dVBndz09/SignsOfAHeartAttack_Hero_FINAL_860x478.png', use_column_width=False, width=600)

    st.write("Gagal jantung merupakan kondisi saat otot jantung cukup melemah. Akibat dari kondisi ini, organ ini tidak mampu lagi memompa cukup darah ke seluruh tubuh pada tekanan yang seharusnya. Meski bisa terjadi pada siapa saja, penyakit ini disebut lebih sering terjadi pada orang yang berusia lanjut. Kondisi ini tidak boleh disepelekan begitu saja dan harus segera mendapatkan penanganan medis.")
    st.write("Penanganan yang cepat akan menurunkan risiko terjadinya komplikasi berbahaya. Penanganan penyakit ini bertujuan untuk meredakan gejala dan meningkatkan kekuatan jantung.")
    

with deskripsi:
    st.subheader("""Tentang Dataset""")
    st.write("Penyakit kardiovaskular (CVDs) adalah penyebab kematian nomor 1 secara global , merenggut sekitar 17,9 juta nyawa setiap tahun , yang merupakan 31% dari semua kematian di seluruh dunia . Gagal jantung adalah kejadian umum yang disebabkan oleh CVD dan kumpulan data ini berisi 12 fitur yang dapat digunakan untuk memprediksi kematian akibat gagal jantung.")
    st.write("Sebagian besar penyakit kardiovaskular dapat dicegah dengan mengatasi faktor risiko perilaku seperti penggunaan tembakau, pola makan yang tidak sehat dan obesitas, kurangnya aktivitas fisik, dan penggunaan alkohol yang berbahaya dengan menggunakan strategi populasi luas.")
    st.write("Orang dengan penyakit kardiovaskular atau yang memiliki risiko kardiovaskular tinggi (karena adanya satu atau lebih faktor risiko seperti hipertensi, diabetes, hiperlipidemia, atau penyakit yang sudah ada) memerlukan deteksi dan penanganan dini di mana model pembelajaran mesin dapat sangat membantu.")
    st.write("""Dataset gagal jantung ini diambil dari <a href="https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data">Kaggle</a>""", unsafe_allow_html=True)
    st.subheader("""Fitur""")
    st.write(
        """
        Fitur yang terdapat pada dataset:
        - Age : Usia pasien
        - Anaemia : Masalah kesehatan yang terjadi saat jumlah sel darah merah dalam tubuh lebih rendah dibandingkan dengan jumlah normalnya, sering dikenal dengan penyakit kekurangan sel darah merah
        - Creatinine Phosphokinase: Creatine Phosphokinase (CK) adalah sejenis protein yang dikenal sebagai enzim. Protein tersebut sebagian besar ditemukan di otot rangka dan jantung, dengan jumlah yang lebih sedikit di otak. Sedangkan tes CK digunakan untuk mengukur jumlah creatine kinase dalam darah. 
        - Diabetes: Diabetes atau penyakit gula adalah penyakit kronis atau yang berlangsung jangka panjang. Penyakit ini ditandai dengan meningkatnya kadar gula darah (glukosa) hingga di atas nilai normal.
        - Ejection Fraction: Dalam pemeriksaan USG jantung, yang paling penting dinilai ialah nilai fraksi ejeksi / ejection fraction (EF). Fraksi ejeksi mencerminkan seberapa banyak darah yang terpompa dibandingkan dengan jumlah darah yang masuk ke dalam kamar jantung. Normalnya, nilai EF berkisar antara 50-70%.
        - High Blood Pressure: Tekanan darah tinggi atau disebut juga hipertensi adalah suatu kondisi ketika seseorang mempunyai tekanan darah yang terukur pada nilai 130/80 mmHg atau lebih tinggi.
        - Platelets: Trombosit (keping darah/platelet) adalah komponen darah yang berfungsi dalam pembekuan darah. Jumlahnya yang terlalu rendah dapat membuat Anda mudah memar dan mengalami perdarahan.
        - Serum Creatinine: Serum kreatinin merupakan sampah hasil metabolisme otot yang mengalir pada sirkulasi darah. Kreatinin lalu disaring ginjal untuk selanjutnya dibuang bersama urine. Serum kreatinin menjadi pertanda baik buruknya fungsi ginjal, karena organ ini yang mengatur agar kreatinin tetap berada pada kadar normalnya.
        - Serum Sodium: Kadar natrium serum adalah parameter utama yang digunakan untuk menilai tonisitas serum yang sering terganggu akibat hiperglikemia. Efek hiperglikemia terhadap penurunan konsentrasi natrium plasma telah diketahui sejak separuh abad yang lalu.
        - Sex: Jenis Kelamin 
        """
    )

with datasets:
    st.subheader("""Dataset Heart Failure Prediction""")
    df = pd.read_csv('https://raw.githubusercontent.com/rizkyluxszerr/datamining/main/gagal_jantung_datasets.csv')
    st.dataframe(df) 

with preprocessing:
    st.subheader("""Rumus Normalisasi Data""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Keterangan :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    #Mendefinisikan Varible X dan Y
    X = df.drop(columns=['smoking','time','DEATH_EVENT'])
    y = df['DEATH_EVENT'].values
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.DEATH_EVENT).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        'Positive' : [dumies[1]],
        'Negative' : [dumies[0]]
    })

    st.write(labels)

with modeling:
    st.subheader("""Metode Yang Digunakan""")
    #Split Data 
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.write("Pilih Metode yang digunakan : ")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-NN')
        destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

        #Gaussian Naive Bayes
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        probas = gaussian.predict_proba(test)
        probas = probas[:,1]
        probas = probas.round()

        gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

        #KNN
        K=10
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        if submitted :
            if naive :
                st.write('Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn :
                st.write("K-NN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
        
        grafik = st.form_submit_button("Grafik Akurasi")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)

with implementasi:
        st.subheader("Form Implementasi")
        with st.form("my_form"):
            age = st.slider('Usia pasien', 40, 95)
            anaemia = st.slider('Penurunan sel darah merah atau hemoglobin (boolean)', 0, 1)
            creatinine_phosphokinase = st.slider('Tingkat enzim CPK dalam darah (mcg/L)', 23, 7861)
            diabetes = st.slider('Jika pasien menderita diabetes (boolean)', 0, 1)
            ejection_fraction = st.slider('Persentase darah yang meninggalkan jantung pada setiap kontraksi (persentase)', 14, 80)
            high_blood_pressure = st.slider('Jika pasien memiliki hipertensi (boolean)', 0, 1)
            platelets = st.slider('Trombosit dalam darah (kiloplatelet/mL)', 25100, 850000)
            serum_creatinine = st.slider('Tingkat serum kreatinin dalam darah (mg/dL)', 0.5, 9.4)
            serum_sodium = st.slider('Tingkat natrium serum dalam darah (mEq/L)', 113, 148)
            sex = st.slider('Wanita atau pria (biner)', 0, 1)
            model = st.selectbox('Model untuk prediksi',
                    ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

            prediksi = st.form_submit_button("Submit")
            if prediksi:
                inputs = np.array([
                    age,
                    anaemia,
                    creatinine_phosphokinase,
                    diabetes,
                    ejection_fraction,
                    high_blood_pressure,
                    platelets,
                    serum_creatinine,
                    serum_sodium,
                    sex
                ])                      
            
                df_min = X.min()
                df_max = X.max()
                input_norm = ((inputs - df_min) / (df_max - df_min))
                input_norm = np.array(input_norm).reshape(1, -1)

                if model == 'Gaussian Naive Bayes':
                    mod = gaussian
                if model == 'K-NN':
                    mod = knn 
                if model == 'Decision Tree':
                    mod = dt

                input_pred = mod.predict(input_norm)

                st.subheader('Hasil Prediksi')
                st.write('Menggunakan Pemodelan :', model)

                if input_pred == 1:
                    st.error('Positive')
                else:
                    st.success('Negative')
            

