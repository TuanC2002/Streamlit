import streamlit as st
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.decomposition import PCA

st.title("Machine Learning Website")
st.header("1. Upload dataset")
uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    df = "./data/" + uploaded_file.name
    with open(df, "wb") as f:
        f.write(bytes_data) 

    st.header("2. Display dataset")
    dataframe = pd.read_csv(df)
    st.write(dataframe)
    x = dataframe.iloc[:,1:]
    y = dataframe.iloc[:,0]
    st.header("3. Choose input features")
    for i in x.columns:
        agree = st.checkbox(i)
        if agree == False:
            x = x.drop(i, 1)
    st.write(x)
    st.header("3. Output features")
    st.write(y)
    st.header("Our model is LogisticRegression")
    radio_choose = st.radio("", ('K-Fold','Train/test split','PCA'))
    if radio_choose=='K-Fold':
        k = st.slider("Select number of k:",2 ,10 ,4)
        # scaler
        scaler = MaxAbsScaler().fit(x)
        x = scaler.transform(x)
        # 
        k_folds = KFold(n_splits=k, shuffle=True, random_state=100)
        reg_logistic_kf = LogisticRegression()
        f1_logistic = cross_val_score(reg_logistic_kf, x, y, scoring='f1', cv=k_folds)
        log_loss_logistic = cross_val_score(reg_logistic_kf, x, y, scoring='neg_log_loss', cv=k_folds)
        k_f_arr = pd.DataFrame(columns = ['F1', 'Log_Loss'])
        for i in range(k):
            k_f_arr = k_f_arr.append({'F1' : abs(f1_logistic[i]), 'Log_Loss' : abs(log_loss_logistic[i])}, ignore_index = True)
    elif radio_choose=='Train/test split': 
        train_per = st.slider(
        'Select a range of training dataset',
        0, 100, 80)
        st.write('The training dataset is ', train_per,'%')
        st.write('Therefore, the test dataset is ', 100 - train_per,'%')
        x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=(100-train_per)/100 ,random_state=0)
        model = LogisticRegression()
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test)
    else:
        # split data train test
        train_per = st.slider(
        'Select a range of training dataset',
        0, 100, 80)
        st.write('The training dataset is ', train_per,'%')
        st.write('Therefore, the test dataset is ', 100 - train_per,'%')
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(100-train_per)/100, random_state=0)
        # pca
        number = st.number_input('Insert a number',min_value = 1,max_value = 13, step=1)
        st.write('The number of components is', number)
        pca = PCA(n_components=number)
        x_train_pca = pca.fit_transform(x_train)
        x_test_pca = pca.transform(x_test)
        #train 
        model_lr = LogisticRegression(random_state = 0)
        model_lr = model_lr.fit(x_train_pca, y_train)
        y_predict = model_lr.predict(x_test_pca)

    if st.button("RUN"):
        if x.shape[1] == 0:
            st.write("Please select input feature")
        else:
            if radio_choose=='K-Fold':
                x_axis = np.arange(k)
                ave = []
                for i in range(k):
                    ave.append(statistics.mean(k_f_arr['F1']))
                fig1, ax1 = plt.subplots()
                plt.bar(x_axis, k_f_arr['F1'], width = 0.5, color='red', label='F1 score')
                plt.ylim(0,1)
                plt.plot(ave, color='blue',label='Mean log loss')
                plt.title('F1 score', fontsize=30)
                plt.xlabel('K-fold', fontsize=15)
                plt.ylabel('f1-score', fontsize=15)
                plt.legend()
                st.pyplot(fig1)
                ave2 =[]
                for i in range(k):
                    ave2.append(statistics.mean(k_f_arr['Log_Loss']))
                fig2, ax2 = plt.subplots()
                plt.bar(x_axis, k_f_arr['Log_Loss'], width = 0.5, color='red', label='Log loss')
                plt.ylim(0,1)
                plt.plot(ave2, color='blue',label='Mean log loss')
                plt.title('Log loss', fontsize=30)
                plt.xlabel('K-fold', fontsize=15)
                plt.ylabel('log loss', fontsize=15)
                plt.legend()
                st.pyplot(fig2)
            elif radio_choose =='Train/test split':
                cm = confusion_matrix(y_test,y_predict)
                st.write("Accuracy score = ", accuracy_score(y_test,y_predict))
                disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
                disp.plot()
                st.pyplot(plt)
            else:
                st.write("Accuracy score = ",accuracy_score(y_test,y_predict))
                #confusion matrix
                st.write("Confusion_matrix ")
                cm = confusion_matrix(y_test, y_predict)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_lr.classes_)
                disp.plot()
                st.pyplot(plt)

