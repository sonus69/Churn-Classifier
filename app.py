from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Memuat model regresi logistik pipeline

model = joblib.load('logreg_model.joblib')

# Tentukan data input untuk model
class CustomerData(BaseModel):
    tenure: int
    InternetService: str
    OnlineSecurity: str
    TechSupport: str
    Contract: str
    PaymentMethod: str

# Membuat FastAPI app
app = FastAPI()

# Tentukan endpoint prediksi
@app.post("/predict")
def predict(data: CustomerData):
    # Ubah input data ke dictionary kemudian ke DataFrame
    input_data = {
        'tenure': [data.tenure],
        'InternetService': [data.InternetService],
        'OnlineSecurity': [data.OnlineSecurity],
        'TechSupport': [data.TechSupport],
        'Contract': [data.Contract],
        'PaymentMethod': [data.PaymentMethod]
    }
   
    import pandas as pd
    input_df = pd.DataFrame(input_data)
   
    # Buat predikasi
    prediction = model.predict(input_df)
   
    # Return prediksinya
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)