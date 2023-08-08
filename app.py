import streamlit as st
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title = 'Customer Churn Prediction',
                  initial_sidebar_state = "expanded",
                  menu_items = {
                      'About' : 'Milestone 1 Customer Churn Predicton '
                  })

image = Image.open('indihome.jpg')

# load model
class columnDropperTransformer():
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)

    def fit(self, X, y=None):
        return self

pickles = open('preprocessings.pkl', 'rb')
preprocessing = pickle.load(pickles)
saved_model=load_model('Model.h5')

def predict(inputs):
    df = pd.DataFrame(inputs, index=[0])
    df = preprocessing.transform(df)
    y_pred = saved_model.predict(df)
    y_pred = np.where(y_pred < 0.5, 0, 1).squeeze()
    print(y_pred)
    return y_pred.item()

columns = ['SeniorCitizen', 'Partner', 'tenure', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
'DeviceProtection', 'TechSupport', 'Contract', 'MonthlyCharges', 'TotalCharges']
label = ['Not Churn', 'Churn']

st.title("Customer Churn Prediction")
st.image(image)


SeniorCitizen = st.selectbox("Senior Citizen", ['Yes', 'No'])
Partner = st.selectbox("Marriage Status", ['Married', 'Not Married'])
tenure = st.slider("Tenure Length", min_value=0.0, max_value=72.0, value=24.0, step=1.0, help='Tenure Length Default 24 Months')
MultipleLines = st.selectbox("Multiple Lines", ['Yes', 'No'])
InternetService = st.selectbox("Which internet service do you use?", ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox("Do you have online security?", ['No', 'Yes', 'No internet service'])
OnlineBackup = st.selectbox("Do you have online backup?", ['No', 'Yes', 'No internet service'])
DeviceProtection = st.selectbox("Do you have device protection?", ['No', 'Yes', 'No internet service'])
TechSupport = st.selectbox("Do you have Tech Support?", ['No', 'Yes', 'No internet service'])
Contract = st.selectbox("Which contract do you use?", ['Month-to-month', 'One year', 'Two year'])
MonthlyCharges = st.number_input("Monthly Charges", min_value=19.0, max_value=119.0, value=75.0, step=0.1, help='Customers Monthly Charges Default is $75')
TotalCharges = st.number_input("Total Charges", min_value=19.0, max_value=8685.0, value=500.0, step=0.1, help='Customers Total Charges Default is $500')

#inference
new_data = [SeniorCitizen, Partner, tenure, 
MultipleLines, InternetService, OnlineSecurity, OnlineBackup, 
DeviceProtection, TechSupport, 
Contract, MonthlyCharges, TotalCharges]
new_data = pd.DataFrame([new_data], columns = columns)
new_data = preprocessing.transform(new_data).tolist()
res = saved_model.predict(new_data)

res = 0 if res < 0.5 else 1

press = st.button('Predict')
if press:
   st.title(label[res])
