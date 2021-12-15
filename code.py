import pandas as pd
import streamlit as st
from sklearn import datasets

import tensorflow as tf
from keras.models import load_model


# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout


def user_input_features(xCol):
    data = {}
    xCol['VR_DESPESA_CONTRATADA'] = xCol['VR_DESPESA_CONTRATADA'].astype(float, errors = 'raise')
    data[col] = st.sidebar.slider(col, xCol['VR_DESPESA_CONTRATADA'].min(), xCol['VR_DESPESA_CONTRATADA'].max(), float(xCol['VR_DESPESA_CONTRATADA'].mean()))  

    # for col in xCol.columns.to_list():
    #     xCol[col] = xCol[col].astype(float, errors = 'raise')
    #     data[col] = st.sidebar.slider(col, xCol[col].min(), xCol[col].max(), float(xCol[col].mean()))  
    return pd.DataFrame(data, index=[0])


def model_ia(x,y):
    model = Sequential()
    model.add(Dense(8, input_dim=9, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.load('pesos.hdf5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def load_data():
    dataset = pd.read_csv('dataset.csv', encoding= 'unicode_escape')

    x = dataset.iloc[:,:-1]
    y = dataset.iloc[:,-1]
    return  (x,y)
     


def main():

    X_casas, y_casas = load_data()

    st.write("# Predição de Eleição #")
    st.write('---')
    
    model = model_ia(X_casas,y_casas)
    
    st.sidebar.header('Escolha de parametros para Predição')






    # df = user_input_features(X_casas)













    # st.header('Parametros especificados')
    # st.write(df)
    # st.write('---')

    # prediction = model.predict(df)

    # st.header('Previsão de diabetes')
    # st.write(prediction)
    # st.write('---')

if __name__=='__main__':
    main()