from nbformat import write
import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler

st.title("Machine Learning Website")
st.header("1. Upload dataset")
uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    df = "data/" + uploaded_file.name
    with open(df, "wb") as f:
        f.write(bytes_data) 

    st.header("2. Display dataset")
    dataframe = pd.read_csv(df)
    st.write(dataframe)
    x = dataframe.iloc[:,0:-1]
    y = dataframe.iloc[:,-1]
    #  Choose input features
    st.header("3. Choose input features")
    for i in x.columns:
        agree = st.checkbox(i)
        if agree == False:
            x = x.drop(i, 1)
    st.write(x)
    for i in x.columns:
        if x[i].dtypes == object:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            x = np.array(ct.fit_transform(x))
    # Output features
    st.header("3. Output features")
    st.write(y)
    st.header("Our model is LinearRegressions")
    # Train_test_spilt
    # st.header("4. Choose hyper parameters")
    # 
    st.header("4. K-Fold or train/test split:")
    k_f_arr = pd.DataFrame(columns = ['MAE', 'MSE'])
    radio_choose = st.radio("", ('K-Fold','Train/test split'))
    if radio_choose=='K-Fold':
        k = st.slider("Select number of k:",2 ,10 ,4)
        # scaler
        scaler = MaxAbsScaler().fit(x)
        x = scaler.transform(x)
        # 
        k_folds = KFold(n_splits=k, shuffle=True, random_state=100)
        reg_linear_kf = LinearRegression()
        score_linear_mae = cross_val_score(reg_linear_kf, x, y, scoring='neg_mean_absolute_error', cv=k_folds)
        score_linear_mse = cross_val_score(reg_linear_kf, x, y, scoring='neg_mean_squared_error', cv=k_folds)
        k_f_arr = pd.DataFrame(columns = ['MAE', 'MSE'])
        for i in range(k):
            k_f_arr = k_f_arr.append({'MAE' : abs(score_linear_mae[i]), 'MSE' : abs(score_linear_mse[i])}, ignore_index = True)
    else: 
        train_per = st.slider(
        'Select a range of training dataset',
        0, 100, 80)
        st.write('The training dataset is ', train_per,'%')
        st.write('Therefore, the test dataset is ', 100 - train_per,'%')
        x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=(100-train_per)/100 ,random_state=0)
        model = LinearRegression()
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test)
    # 
    if st.button("RUN"):
        if x.shape[1] == 0:
            st.write("Please select input feature")
        else:
            if radio_choose=='K-Fold':
                st.write(k_f_arr)
                labels = []
                for i in range(k):
                    labels.append(str(i+1) + '-Fold')
                x_axis = np.arange(len(labels))
                fig, ax = plt.subplots(figsize=(20, 20))
                plt.bar(x_axis - 0.25, k_f_arr['MAE'], width=0.5, color='red', label='MAE')
                plt.bar(x_axis + 0.25, k_f_arr['MSE'], width=0.5, color='blue', label='MSE')
                plt.xticks(x_axis, labels)  
            else:
                mse = mean_squared_error(y_test, y_predict)
                mae = mean_absolute_error(y_test, y_predict)
                k_f_arr = k_f_arr.append({'MAE' : mae, 'MSE' : mse}, ignore_index = True)
                st.write("y test:", y_test)
                st.write("y predict:", y_predict)
                st.write(k_f_arr)
                fig, ax = plt.subplots(figsize=(20, 20))
                plt.bar(0.25, k_f_arr['MAE'], width=0.5, color='red', label='MAE')
                plt.bar(0.75, k_f_arr['MSE'], width=0.5, color='blue', label='MSE') 
            plt.title('Compare MAE and MSE', fontsize=30)
            plt.xlabel('Linear Regression', fontsize=15)
            plt.ylabel('Metrics', fontsize=15)
            plt.yscale('log')
            fig.tight_layout()
            st.pyplot(fig) 

    


    