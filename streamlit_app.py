pip install streamlit

import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

from lightgbm import LGBMRegressor

def load_model():
    try:
        with open("../models/regressor.pkl", "rb") as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        st.error("Model file not found. Please make sure the model file is available.")
        return None

data = load_model()

if data is not None:
    print("Keys in the loaded dictionary:", data.keys())

    # Check if the key is present before accessing it
    if "normalization" in data:
        norm = data["normalization"]
        model = data["model"]
        # ... rest of the code ...
    else:
        st.error("Key 'normalization' not found in the loaded dictionary.")

    le_manufacturer = LabelEncoder()
    le_engine = LabelEncoder()
    le_transmission = LabelEncoder()

def main():
    # build the main app
    st.title("USED CAR PRICE PREDICTION APP")

    st.write("""### We need some information to predict the price""")

    manufacturers = (
        "Abarth", "Alfa-Romero", "Audi", "BMW", "Bentley", "Chevrolet", "Chrysler", "Citroen", "DS",
        "Dacia", "Fiat", "Ford", "Honda", "Hyundai", "Infiniti", "Isuzu", "Jaguar", "Jeep", "Kia",
        "Land-Rover", "Lexus", "MG", "Maserati", "Mazda", "Mercedes-Benz", "Mini", "Mitsubishi", "Nissan",
        "Peugeot", "Porsche", "Renault", "Seat", "Skoda", "Smart", "Subaru", "Suzuki", "Toyota", "Vauxhall",
        "Volkswagen", "Volvo",
    )

    engines = ("Diesel", "Electric", "Hybrid", "Petrol", "Plug_in_hybrid")

    transmissions = ("Automatic", "Manual", "Semiautomatic")

    manufacturer = st.selectbox("Manufacturer", manufacturers)
    age = st.slider("Age of Car", 1, 50, 1)
    mileage = st.number_input("Mileage", min_value=0, max_value=999999, value=0)
    engine = st.selectbox("Engine", engines)
    transmission = st.selectbox("Transmission", transmissions)

    ok = st.button("Predict Price")

    if ok:
        X = np.array([[manufacturer, age, mileage, engine, transmission]])
        X[:, 0] = le_manufacturer.fit_transform(X[:, 0])
        X[:, 3] = le_engine.fit_transform(X[:, 3])
        X[:, 4] = le_transmission.fit_transform(X[:, 4])
        scaled_X = norm.transform(X)

        try:
            price = model.predict(scaled_X)
            # convert the price from log_price to actual price
            actual_price = np.exp(price) + 1
            actual_price = round(actual_price[0])  # round to the nearest Pounds
            st.subheader(f"The estimated cost of the car is {actual_price:,} pounds")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    main()
