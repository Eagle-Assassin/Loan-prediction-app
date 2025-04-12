# import pickle
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

model_columns = joblib.load('model_columns.pkl')
model = joblib.load('feature_model.pkl')
print(model_columns)
print(type(model)) 
# with open('Loan-prediction-app/feature_model.pkl', 'rb') as file:
#     model = pickle.load(file)
# with open('Loan-prediction-app/model_columns.pkl', 'rb') as file:
#     model_columns = pickle.load(file)
# model_columns = joblib.load('model_columns.pkl')
# model = joblib.load('feature_model.pkl')
print(type(model)) 
def run():
    st.title('Predict Approval of your loan')
   
    col1, col2, col3 = st.columns(3)
    #First row
    with col1:
        no_of_dependents = st.number_input('Number of dependents', value=0)
    with col2:
        # edu_dropdown = ('Not Graduate', 'Graduate')
        # edu_options = list(range(len(edu_dropdown)))
        education = st.selectbox('Education', options=['Not Graduate', 'Graduate'])
    with col3:       
        # self_employed_dropdown = ('No', 'Yes')
        # self_employed_options= list(range(len(self_employed_dropdown)))
        self_employed = st.selectbox('Self Employed', options=['No', 'Yes'])
    #Second row
    with col1:
        income_annum = st.number_input("Applicant's Annual Income", value=0)
    with col2:
        loan_amount = st.number_input("Loan Amount", value=0)   
    with col3:
        loan_term = st.number_input('Tenure(In Years)', value=0)
    #Third row
    with col1:
        cibil_score =  st.number_input('Cibili score', value=0)
    with col2:
        residential_assets_value = st.number_input('Residential Assets Value', value=0)
    with col3:
        commercial_assets_value = st.number_input('Commercial Assests Value', value=0)
    #$th row
    with col1:
        luxury_assets_value = st.number_input('Luxury Assets Value', value=0)
    with col2:
        bank_asset_value = st.number_input('Bank Assets Value', value=0)

    print('income_annum', income_annum)
    print('no_of_dependents', no_of_dependents)
    print('education', education)
    print('self_employed', self_employed)
    print('loan_amount', loan_amount)
    print('loan_term', loan_term)
    print('cibil_score', cibil_score)
    print('residential_assets_value', residential_assets_value)
    print('commercial_assets_value', commercial_assets_value)
    print('luxury_assets_value', luxury_assets_value)
    print('bank_asset_value', bank_asset_value)

    if st.button('Submit'):
        user_input = {
    'no_of_dependents': no_of_dependents, 
    'education': education,
    'self_employed':self_employed, 
    'income_annum':income_annum,
    'loan_amount':loan_amount, 
    'loan_term':loan_term, 
    'cibil_score':cibil_score, 
    'residential_assets_value':residential_assets_value,
    'commercial_assets_value':commercial_assets_value, 
    'luxury_assets_value':luxury_assets_value, 
    'bank_asset_value':bank_asset_value,
        }
        input_df = pd.DataFrame(user_input,index=range(0,1))
        input_encoded = pd.get_dummies(input_df)

        # Reindex to match training columns (fill missing columns with 0)
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
        print(model)
        prediction=model.predict(input_encoded)

        print(model.predict_proba(input_encoded.values.reshape(1, -1)))

        if hasattr(model, 'predict'):
            pass
        prediction = model.predict(input_encoded)
        probabilities = model.predict_proba(input_encoded)
        print("Prediction:", prediction)
        print("Probabilities:", probabilities)
    # Display the prediction and probabilities
        st.success(f"Prediction: {'Approved' if prediction[0] == 1 else 'Rejected'}")
        st.write(f"Approval Probability: {probabilities[0][1]:.2f}")
        st.write(f"Rejection Probability: {probabilities[0][0]:.2f}")
        # else:
        #  st.error("The loaded object is not a valid model.")



run()