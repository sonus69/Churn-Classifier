import joblib
import pandas as pd

# Muat model
model = joblib.load('logreg_model.joblib')

# Data input
input_data = {
    'tenure': [72],
    'InternetService': ['Fiber optic'],
    'OnlineSecurity': ['Yes'],
    'TechSupport': ['Yes'],
    'Contract': ['Two year'],
    'PaymentMethod': ['Credit card (automatic)']
}
input_df = pd.DataFrame(input_data)

# Prediksi
prediction = model.predict(input_df)
print({"prediction": int(prediction[0])})
