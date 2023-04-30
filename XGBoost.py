import streamlit as st
import statistics
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from xgboost import XGBClassifier as xgb

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
    x = dataframe.iloc[:,1:-1]
    y = dataframe.iloc[:,-1]
    st.header("3. Choose input features")
    for i in x.columns:
        agree = st.checkbox(i)
        if agree == False:
            x = x.drop(i, 1)
    st.write(x)
    st.header("3. Output features")
    st.write(y)
    st.header("Training")
    train_per = st.slider(
        'Select a range of training dataset',
        0, 100, 80)
    st.write('The training dataset is ', train_per,'%')
    st.write('Therefore, the test dataset is ', 100 - train_per,'%')
    x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=(100-train_per)/100 ,random_state=0)
    radio_choose = st.radio("", ('Logistic Regression','svm','XGBoost','DecisionTree'))
    
    if radio_choose=='Logistic Regression':
        model = LogisticRegression()
        model.fit(x_train,y_train)
    elif radio_choose=='svm': 
        model = SVC(kernel='poly')
        model.fit(x_train,y_train)
    elif radio_choose=='DecisionTree':
        model = DecisionTreeClassifier()
        model.fit(x_train,y_train)
    else:
        model = xgb()
        model.fit(x_train,y_train)
    if st.button("RUN"):
        if x.shape[1] == 0:
            st.write("Please select input feature")
        else:
            st.write("Model: ",radio_choose )
            y_predict = model.predict(x_test)
            st.write("Accuracy score = ",accuracy_score(y_test,y_predict))
            #confusion matrix
            st.write("Confusion_matrix ")
            cm = confusion_matrix(y_test, y_predict)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot()
            st.pyplot(plt)
    
    st.header("Compare different models")
    options = st.multiselect(
    'Models will be used to compared:',
    ['Logistic Regression', 'Decision Tree', 'SVM','XGBoost'],
    ['Logistic Regression', 'Decision Tree', 'SVM','XGBoost'])
    if st.button("Compare"):
        st.write('Models selected:', options)
        df_acc = pd.DataFrame(columns = ['Models', 'Accuracy'])
        for i in options:
            if i == 'Logistic Regression':
                lr = LogisticRegression().fit(x_train, y_train)
                df_acc = df_acc.append({'Models' : 'LR', 'Accuracy' : lr.score(x_test, y_test)}, ignore_index = True)
            elif i == 'Decision Tree':
                dt = DecisionTreeClassifier().fit(x_train, y_train)
                df_acc = df_acc.append({'Models' : 'DT', 'Accuracy' : dt.score(x_test, y_test)}, ignore_index = True)
            elif i == 'SVM':
                svm =  SVC(kernel='poly').fit(x_train, y_train)
                df_acc = df_acc.append({'Models' : 'SVM poly', 'Accuracy' : svm.score(x_test, y_test)}, ignore_index = True)
            elif i == 'XGBoost':
                xg = xgb()
                sv = xg.fit(x_train, y_train)
                df_acc = df_acc.append({'Models' : 'XGBoost', 'Accuracy' : sv.score(x_test, y_test)}, ignore_index = True)
        st.write(df_acc)
        New_Colors = ['green', 'blue', 'purple', 'brown', 'teal', 'red']
        fig, ax = plt.subplots()
        plt.bar(df_acc['Models'], df_acc['Accuracy'], color=New_Colors)
        plt.title('Compare accuracy of different models', fontsize=14)
        plt.xlabel('Models', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.grid(True)
        st.pyplot(fig)

