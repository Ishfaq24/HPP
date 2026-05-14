import joblib
import pandas as pd

model = joblib.load('../models/model.pkl')

sample_data = {
     'City': ['Bangalore'],
    'Locality_Tier': ['Premium'],
    'BHK': [5],
    'Bathrooms': [5],
    'Super_Area_sqft': [3500],
    'Carpet_Area_sqft': [3000],
    'Floor_No': [1],
    'Total_Floors': [2],
    'Property_Age_years': [2],
    'Parking': [4],
    'Furnishing': ['Furnished'],
    'Lift': [0],
    'Gated_Society': [1],
    'Distance_to_Metro_km': [3],
    'Distance_to_CityCenter_km': [12],
    'Nearby_School_km': [1],
    'Nearby_Hospital_km': [2],
    'Crime_Rate_Index': [8],
    'Price_per_sqft_INR': [16000]
}

input_df = pd.DataFrame(sample_data)

prediction = model.predict(input_df)

print(f"Predicted House Price: ₹{prediction[0]:,.2f}")
