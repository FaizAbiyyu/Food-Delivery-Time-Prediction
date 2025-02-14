import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tkinter import messagebox
import tkinter as tk
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

st.title('Food Delivery Time Prediction 🛵')
tab1, tab2, tab3, tab4 = st.tabs(["Datasets", "Visualisasi", "Modeling", "Testing"])

with st.sidebar:
    st.header('Sidebar hanya bisa digunakan untuk Modeling dan Prediksi Model')
    
    model_options = ['Gradient Boosting Regressor', 'Random Forest Regressor', 'Linear Regression','SVR','Decision Tree Regressor', 'XGBoost', 'LightGBM'
                        'HPTuning Gradient Boosting Regressor', 'HPTuning Random Forest Regressor', 'HPTuning SVR' , 'HPTuning Decision Tree Regressor', 'HPTuning XGBoost','HPTuning LightGBM'
                    ]
    selected_model = st.selectbox('Pilih Model', model_options)
    
    if selected_model == 'Gradient Boosting Regressor':
        with open('GradientBoosting_Model.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selected_model == 'Random Forest Regressor':
        with open('RandomForest_Model.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selected_model == 'Linear Regression':
        with open('LinearRegression_Model.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selected_model == 'SVR':
        with open('SVR_Model.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selected_model == 'Decision Tree Regressor':
        with open('DecisionTreeRegressor_Model.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selected_model == 'XGBoost':
        with open('XGBoost_Model.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selected_model == 'LightGBM':
        with open('LightGBM_Model.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selected_model == 'HPTuning Gradient Boosting Regressor':
        with open('HT_GradientBoosting_Model.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selected_model == 'HPTuning Random Forest Regressor':
        with open('HT_RandomForest_Model.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selected_model == 'HPTuning SVR':
        with open('HT_SVR_Model.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selected_model == 'HPTuning Decision Tree Regressor':
        with open('HT_DecisionTreeRegressor_Model.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selected_model == 'HPTuning XGBoost':
        with open('HT_XGBoost_Model.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selected_model == 'HPTuning LightGBM':
        with open('HT_LightGBM_Model.pkl', 'rb') as f:
            model = pickle.load(f)
            
    st.write('Model yang dipilih:', selected_model)
    st.header('=================================')
    st.header('Bagian ini hanya bisa digunakan pada Visualisasi')
    # Menambahkan filter untuk Delivery Time
    delivery_time_range = st.slider('Pilih Rentang Waktu Pengiriman (menit)', 0, 100, (0, 100))

with tab1:    
    st.header("Datasets")
    with st.expander('Data'):
      st.write('**Raw data**')
      df = pd.read_csv('Food_Delivery_Times.csv')
      df
      
    with st.expander('Data Cleaning'):
      test = df.isnull().sum()
      test
      
      st.write("Data setelah Di Cleaning (Mengisi Kolom Object dengan Mode dan Kolom Numerik dengan Mean)")
          # Kode yang ingin dijalankan
      df['Weather'] = df['Weather'].fillna(df['Weather'].mode()[0])
      df['Traffic_Level'] = df['Traffic_Level'].fillna(df['Traffic_Level'].mode()[0])
      df['Time_of_Day'] = df['Time_of_Day'].fillna(df['Time_of_Day'].mode()[0])
        
      df['Courier_Experience_yrs'] = df['Courier_Experience_yrs'].fillna(df['Courier_Experience_yrs'].mean())
        
      test = df.isnull().sum()
      st.write(test)

    
    with st.expander('Transformasi Data'):
      # Label encoding untuk data kategorikal
      label_encoders = {}
      categorical_cols = ['Weather', 'Traffic_Level', 'Time_of_Day','Vehicle_Type']
      for col in categorical_cols:
          le = LabelEncoder()
          df[col] = le.fit_transform(df[col])
          label_encoders[col] = le
        
      # Tampilkan hasil
      st.write("Data telah di-transformasi")
      test = df.head()
      st.write(test)
      
            # Tampilkan keterangan label encoding
      st.write("Keterangan Label Encoding:")
      for col, le in label_encoders.items():
          st.write(f"{col}:")
          labels = le.classes_
          st.write(pd.DataFrame({'Nilai': range(len(labels)), 'Label': labels}))
          
with tab2:
    st.header("Data Visualization")
    
    # Filter data berdasarkan rentang waktu pengiriman dari sidebar
    df_filtered = df[(df['Delivery_Time_min'] >= delivery_time_range[0]) & (df['Delivery_Time_min'] <= delivery_time_range[1])]
    
    st.subheader("Distribusi Waktu Pengiriman")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df_filtered['Delivery_Time_min'], bins=30, kde=True, color="royalblue", ax=ax)
    ax.set_xlabel("Delivery Time (minutes)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Delivery Time")
    st.pyplot(fig)

    st.subheader("Matriks Korelasi")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_filtered.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.subheader("Jarak vs Waktu Pengiriman")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df_filtered['Distance_km'], y=df_filtered['Delivery_Time_min'], alpha=0.6, color="darkorange", ax=ax)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Delivery Time (minutes)")
    ax.set_title("Distance vs Delivery Time")
    st.pyplot(fig)

    st.subheader("Pengaruh Cuaca terhadap Waktu Pengiriman")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=df_filtered['Weather'], y=df_filtered['Delivery_Time_min'], palette="coolwarm", ax=ax) # Something worng
    ax.set_xlabel("Weather Condition")
    ax.set_ylabel("Delivery Time (minutes)")
    ax.set_title("Effect of Weather on Delivery Time")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
with tab3:
    st.header("Modeling")
    
    # Memilih fitur dan target
    features = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs', 'Weather', 'Traffic_Level', 'Time_of_Day']
    X = df[features]
    y = df['Delivery_Time_min']
    
    # Membagi data menjadi training dan testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.write('Model yang dipilih:', selected_model)
    
    # Input data uji (contoh, bisa diganti dengan input dari pengguna)
    if 'X_test' in locals() and 'y_test' in locals():
        y_pred_rl = model.predict(X_test)
        
        # Evaluasi model
        mae = mean_absolute_error(y_test, y_pred_rl)
        mse = mean_squared_error(y_test, y_pred_rl)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred_rl)
        
        st.write(f"**MAE:** {mae}")
        st.write(f"**RMSE:** {rmse}")
        st.write(f"**R-squared:** {r2}")
        
        # Plot Residuals
        st.subheader("Distribusi Residuals")
        fig, ax = plt.subplots(figsize=(10, 5))
        residuals = y_test - y_pred_rl
        sns.histplot(residuals, bins=30, kde=True, ax=ax)
        ax.axvline(x=0, color='red', linestyle='--')
        st.pyplot(fig)
        
        # Scatter Plot: Nilai Aktual vs Prediksi
        st.subheader("Nilai Aktual vs Prediksi")
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.scatterplot(x=y_test, y=y_pred_rl, ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted Values")
        st.pyplot(fig)
    else:
        st.error("Harap pastikan X_test dan y_test tersedia dalam lingkungan kerja.")

with tab4:
    st.header("Testing: Prediksi Waktu Pengiriman")
    
    # Membuat form input untuk fitur-fitur yang diperlukan
    st.subheader("Masukkan Data untuk Prediksi")
    
    with st.form("prediction_form"):
        distance_km = st.number_input("Jarak (km)", min_value=0.0, max_value=100.0, value=5.0)
        preparation_time_min = st.number_input("Waktu Persiapan (menit)", min_value=0, max_value=120, value=20)
        courier_experience_yrs = st.number_input("Pengalaman Kurir (tahun)", min_value=0, max_value=30, value=2)
        
        # Input untuk fitur kategorikal (Weather, Traffic_Level, Time_of_Day)
        weather_options = ['Clear', 'Rainy', 'Snowy', 'Foggy', 'Windy']
        weather = st.selectbox("Kondisi Cuaca", weather_options)
        
        traffic_level_options = ['Low', 'Medium', 'High']
        traffic_level = st.selectbox("Tingkat Lalu Lintas", traffic_level_options)
        
        time_of_day_options = ['Morning', 'Evening', 'Afternoon', 'Night']
        time_of_day = st.selectbox("Waktu Hari", time_of_day_options)
        
        # Tombol untuk melakukan prediksi
        submit_button = st.form_submit_button("Prediksi Waktu Pengiriman")
    
    if submit_button:
        # Mengubah input kategorikal menjadi numerik menggunakan LabelEncoder yang sudah dibuat sebelumnya
        weather_encoded = label_encoders['Weather'].transform([weather])[0]
        traffic_level_encoded = label_encoders['Traffic_Level'].transform([traffic_level])[0]
        time_of_day_encoded = label_encoders['Time_of_Day'].transform([time_of_day])[0]
        
        # Membuat array dari input yang telah diubah
        input_data = np.array([[distance_km, preparation_time_min, courier_experience_yrs, weather_encoded, traffic_level_encoded, time_of_day_encoded]])
        
        # Melakukan prediksi menggunakan model yang telah dipilih
        predicted_delivery_time = model.predict(input_data)
        
        # Menampilkan hasil prediksi
        st.subheader("Hasil Prediksi")
        st.write(f"Prediksi Waktu Pengiriman Kurang Lebih: **{predicted_delivery_time[0]:.2f} menit**")
        
        # # Menampilkan keterangan label encoding untuk memudahkan interpretasi
        # st.write("Keterangan Label Encoding:")
        # st.write(f"**Cuaca:** {weather} = {weather_encoded}")
        # st.write(f"**Tingkat Lalu Lintas:** {traffic_level} = {traffic_level_encoded}")
        # st.write(f"**Waktu Hari:** {time_of_day} = {time_of_day_encoded}")
