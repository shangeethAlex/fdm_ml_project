import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline



# Function to make predictions
def predict(data):
    pred_df = data.get_data_as_frame()
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    return results[0]

# Streamlit app
st.set_page_config(page_title="CO2 Emissions Prediction", layout="centered")

st.title("CO2 Emissions Prediction")

# Input fields for categorical data
make = st.selectbox("Make", [
    'Select Make', 'ACURA', 'ALFA ROMEO', 'ASTON MARTIN', 'AUDI', 'BENTLEY', 'BMW', 
    'BUGATTI', 'BUICK', 'CADILLAC', 'CHEVROLET', 'CHRYSLER', 'DODGE', 'FIAT', 
    'FORD', 'GENESIS', 'GMC', 'HONDA', 'HYUNDAI', 'INFINITI', 'JAGUAR', 'JEEP', 
    'KIA', 'LAMBORGHINI', 'LAMBORGHINI Aventador Coupe', 'LAND ROVER', 
    'LAND ROVER Range Rover SVAutobiography Dynamic', 'LEXUS', 'LEXUS UX 250h AWD', 
    'LINCOLN', 'MASERATI', 'MAZDA', 'MERCEDES-BENZ', 'MINI', 'MITSUBISHI', 
    'NISSAN', 'PORSCHE', 'RAM', 'ROLLS-ROYCE', 'SCION', 'SMART', 'SRT', 'SUBARU', 
    'TOYOTA', 'VOLKSWAGEN', 'VOLVO'
])

model = st.selectbox("Model", [
    'Select Model', 'ILX', 'ILX HYBRID', 'MDX 4WD', 'V90 CC T5 AWD', 'XC40 T5 AWD', 'XC40 T4 AWD','F-150'
])  # Add more models as needed

vehicle_class = st.selectbox("Vehicle Class", [
    'Select Vehicle Class', 'COMPACT', 'SUV - SMALL', 'MID-SIZE', 'TWO-SEATER', 'MINICOMPACT',
    'SUBCOMPACT', 'FULL-SIZE', 'STATION WAGON - SMALL', 'SUV - STANDARD', 'VAN - CARGO',
    'VAN - PASSENGER', 'PICKUP TRUCK - STANDARD', 'SPECIAL PURPOSE VEHICLE', 
    'PICKUP TRUCK - SMALL', 'MINIVAN', 'STATION WAGON - MID-SIZE'
])

transmission = st.selectbox("Transmission", [
    'Select Transmission', 'AS5', 'M6', 'AV7', 'AS6', 'AM7', 'AM8', 'AS9', 'AM9', 
    'AS10', 'AM6', 'A8', 'A6', 'M7', 'AV8', 'AS8', 'AS7', 'A7', 'A9', 'AV', 
    'A10', 'A4', 'M5', 'A5', 'AV6', 'AV10', 'AS4', 'AM5'
])

fuel_type = st.selectbox("Fuel Type", [
    'Select Fuel Type', 'Z', 'D', 'X', 'E'
])

# Input fields for numerical data
engine_size = st.number_input("Engine Size (L)", min_value=0.0, step=0.1)
cylinders = st.number_input("Cylinders", min_value=1, step=1)
fuel_consumption_city = st.number_input("Fuel Consumption City (L/100 km)", min_value=0.0, step=0.1)
fuel_consumption_hwy = st.number_input("Fuel Consumption Hwy (L/100 km)", min_value=0.0, step=0.1)
fuel_consumption_comb = st.number_input("Fuel Consumption Comb (L/100 km)", min_value=0.0, step=0.1)
fuel_consumption_comb_mpg = st.number_input("Fuel Consumption Comb (mpg)", min_value=0, step=1)

# Predict button
if st.button("Predict CO2 Emissions"):
    if (make != 'Select Make' and model != 'Select Model' and 
        vehicle_class != 'Select Vehicle Class' and transmission != 'Select Transmission' and 
        fuel_type != 'Select Fuel Type'):
        
        data = CustomData(
            Make=make,
            Model=model,
            Vehicle_Class=vehicle_class,
            Transmission=transmission,
            Fuel_Type=fuel_type,
            Engine_Size=engine_size,
            Cylinders=cylinders,
            Fuel_Consumption_City=fuel_consumption_city,
            Fuel_Consumption_Hwy=fuel_consumption_hwy,
            Fuel_Consumption_Comb=fuel_consumption_comb,
            Fuel_Consumption_Comb_mpg=fuel_consumption_comb_mpg
        )
        prediction = predict(data)
        st.success(f"The predicted CO2 Emissions (g/km) is: {int(round(prediction))}")  # Round and convert to int
    else:
        st.error("Please fill in all the required fields.")
