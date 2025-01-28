import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf


model=tf.keras.models.load_model('ANNmodel.h5')


with open ('label_ecoder_gender.pkl','rb') as file:
    label_ecoder_gender=pickle.load(file)

with open ('onehot_encode_geo.pkl','rb') as file:
    onehot_geo=pickle.load(file)
    
with open ('scaler.pkl','rb') as file:
    scaler=pickle.load(file)
    
st.title("Customers Churn")

geography=st.selectbox('Geography',onehot_geo.categories_[0])
gender=st.selectbox('Gender',label_ecoder_gender.classes_)
age=st.slider('Age',18,90)
creditscore=st.number_input("Credit Score")
tenure=st.slider('Tenure',0,10)
balance=st.number_input('Balance')
num_of_prod=st.slider('Number of Products',0,4)
has_cr_card=st.selectbox('HasCrCard',[0,1])
is_active_member=st.selectbox('Member',[0,1])
estimated_salary=st.number_input("salary")


input_data=pd.DataFrame({
    'CreditScore':[creditscore],
    'Geography':[geography],
    'Gender':[label_ecoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_prod],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary],
    
})
geo_encode=onehot_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encode,columns=onehot_geo.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
input_data=input_data.drop(columns=['Geography'],axis=1)

input_data_scaled=scaler.transform(input_data)

pred=model.predict(input_data_scaled)
pred_prob=pred[0][0]

if pred_prob>0.5:
    st.text('the customer is likely to churn')
else:
    st.text("the customer is not about to churn")
