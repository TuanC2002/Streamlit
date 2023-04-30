import streamlit as st
import statistics
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from xgboost import XGBClassifier as xgb
from sklearn.model_selection import GridSearchCV as gscv
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
    x = dataframe.iloc[:,0:-1]
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
    st.header("SMV model: default parameter")
    if st.button('Run with default parameter'):
        model = SVC()
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test) 
        st.write("Accuracy score:", accuracy_score(y_test, y_predict))
        st.write("Confusion_matrix ")
        cm = confusion_matrix(y_test, y_predict)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot()
        st.pyplot(plt)


    st.header("SMV model: use GridSearhCV to search besst parameter")
    st.write("Enter paramter, separated by ',' ")
    C_para = st.text_input('C paramter')
    # 
    st.write('kernel=`rbf`')
    # 
    gamma_para = st.text_input('gamma: float')
    option = st.multiselect(
        'gamma: auto or float',
        ['auto','scale'],
        ['auto','scale'])
    # st.write(gamma_para)
    # #######
    
    # st.write(param_grid)
    if st.button('Run'):
        C_para = [float(x) for x in C_para.strip().split(',')]
        if gamma_para =='':
            gamma_para=[]
        else:
            gamma_para = [float(x) for x in gamma_para.strip().split(',')]
        for i in option:
            gamma_para.append(str(i))
        param_grid = {'C': C_para,  
            'gamma': gamma_para, 
            'kernel': ['rbf']}
        st.write(param_grid)
        grid =gscv(SVC(), param_grid, refit = True,verbose = 3)
        grid.fit(x_train,y_train) 
        st.write("Best paragrameter for svm model")
        st.write(grid.best_params_)
        model = grid.best_estimator_.fit(x_train,y_train)
        y_predict = model.predict(x_test) 
        st.write("Accuracy score:", accuracy_score(y_test, y_predict))
        st.write("Confusion_matrix ")
        cm = confusion_matrix(y_test, y_predict)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot()
        st.pyplot(plt)