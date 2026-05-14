from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('../models/model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    data = {
        'City': [request.form['city']],
        'Locality_Tier': [request.form['locality']],
        'BHK': [int(request.form['bhk'])],
        'Bathrooms': [int(request.form['bathrooms'])],
        'Super_Area_sqft': [float(request.form['super_area'])],
        'Carpet_Area_sqft': [float(request.form['carpet_area'])],
        'Floor_No': [int(request.form['floor_no'])],
        'Total_Floors': [int(request.form['total_floors'])],
        'Property_Age_years': [int(request.form['property_age'])],
        'Parking': [int(request.form['parking'])],
        'Furnishing': [request.form['furnishing']],
        'Lift': [int(request.form['lift'])],
        'Gated_Society': [int(request.form['gated'])],
        'Distance_to_Metro_km': [float(request.form['metro'])],
        'Distance_to_CityCenter_km': [float(request.form['city_center'])],
        'Nearby_School_km': [float(request.form['school'])],
        'Nearby_Hospital_km': [float(request.form['hospital'])],
        'Crime_Rate_Index': [float(request.form['crime'])],
        'Price_per_sqft_INR': [float(request.form['price_sqft'])]
    }

    df = pd.DataFrame(data)

    prediction = model.predict(df)

    result = f"Predicted House Price: ₹{prediction[0]:,.2f}"

    return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
    