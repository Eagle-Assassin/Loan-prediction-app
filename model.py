import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


model_columns = joblib.load('model_columns.pkl')
model = joblib.load('feature_model.pkl')


user_input = {

    'no_of_dependents': np.int64(5), 
    'education': ['Graduate'],
    'self_employed':'Yes', 
    'income_annum':np.int64(1000000),
    'loan_amount':np.int64(2300000), 
    'loan_term':np.int64(12), 
    'cibil_score':np.int64(317), 
    'residential_assets_value':np.int64(2800000),
    'commercial_assets_value':np.int64(500000), 
    'luxury_assets_value':np.int64(3300000), 
    'bank_asset_value':np.int64(800000),

}


input_df = pd.DataFrame(user_input,index=range(0,1))
input_encoded = pd.get_dummies(input_df)

# Reindex to match training columns (fill missing columns with 0)
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

prediction=model.predict(input_encoded)

print(model.predict_proba(input_encoded.values.reshape(1, -1)))


print(prediction)