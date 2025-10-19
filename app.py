import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model

# --- Load Model and Preprocessors ---
model = load_model('model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)


# --- Streamlit UI ---
st.title("💡 Customer Churn Prediction App")

st.markdown("### Please fill in the customer details below:")

col1, col2 = st.columns(2)
with col1:
    Geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    Gender = st.selectbox('👩‍💼 Gender', label_encoder_gender.classes_)
    Age = st.slider('🎂 Age', 18, 92, 30)
    Tenure = st.slider('📅 Tenure (Years with Bank)', 0, 10, 3)
    Number_of_Products = st.slider('🛍️ Number of Products', 1, 4, 1)

with col2:
    CreditScore = st.number_input('💳 Credit Score', 300, 850, 650)
    Balance = st.number_input('🏦 Account Balance', min_value=0.0, step=100.0)
    EstimatedSalary = st.number_input('💰 Estimated Salary', min_value=0.0, step=100.0)
    HasCrCard = st.selectbox('💳 Has Credit Card?', [0, 1])
    IsActiveMember = st.selectbox('✅ Is Active Member?', [0, 1])


# --- Encode Categorical Features ---
gender_encoded = label_encoder_gender.transform([Gender])[0]
geo_encoded = onehot_encoder_geo.transform([[Geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# --- Create Input DataFrame with Correct Feature Names ---
input_data = pd.DataFrame({
    'CreditScore': [CreditScore],
    'Gender': [gender_encoded],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [Number_of_Products],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
})

# Combine OneHotEncoded geography columns
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)], axis=1)

# --- Scale Input Data ---
input_data_scaled = scaler.transform(input_data)

# --- Predict ---
prediction = model.predict(input_data_scaled)[0][0]
churn_prob = float(prediction)

# --- Display Results ---
st.subheader("🔮 Prediction Result:")
st.write(f"**Churn Probability:** {churn_prob:.2f}")

if churn_prob > 0.5:
    st.write("⚠️ The customer is **likely to churn.**")
else:
    st.write("✅ The customer is **unlikely to churn.**")

st.caption("Model prediction powered by ANN (Artificial Neural Network).")
