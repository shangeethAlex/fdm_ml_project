import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):  # to do predictions
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/proprocessor.pkl'
            
            # Load the model and preprocessor
            logging.info("Loading model and preprocessor...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Log input features before processing
            logging.info(f"Input features before scaling: {features}")

            # Scale the data
            data_scaled = preprocessor.transform(features)
            logging.info(f"Scaled features: {data_scaled}")

            # Make predictions
            preds = model.predict(data_scaled)
            logging.info(f"Predictions: {preds}")

            return preds

        except Exception as e:
            logging.error(f"An error occurred during prediction: {e}")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 Make: str,
                 Model: str,
                 Vehicle_Class: str,
                 Engine_Size: float,
                 Cylinders: int,
                 Transmission: str,
                 Fuel_Type: str,
                 Fuel_Consumption_City: float,
                 Fuel_Consumption_Hwy: float,
                 Fuel_Consumption_Comb: float,
                 Fuel_Consumption_Comb_mpg: int):
        
        self.Make = Make
        self.Model = Model
        self.Vehicle_Class = Vehicle_Class
        self.Engine_Size = Engine_Size
        self.Cylinders = Cylinders
        self.Transmission = Transmission
        self.Fuel_Type = Fuel_Type
        self.Fuel_Consumption_City = Fuel_Consumption_City
        self.Fuel_Consumption_Hwy = Fuel_Consumption_Hwy
        self.Fuel_Consumption_Comb = Fuel_Consumption_Comb
        self.Fuel_Consumption_Comb_mpg = Fuel_Consumption_Comb_mpg

    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {
                "Make": [self.Make],
                "Model": [self.Model],
                "Vehicle Class": [self.Vehicle_Class],  # Fix column name
                "Engine Size(L)": [self.Engine_Size],  # Fix column name
                "Cylinders": [self.Cylinders],
                "Transmission": [self.Transmission],
                "Fuel Type": [self.Fuel_Type],  # Fix column name
                "Fuel Consumption City (L/100 km)": [self.Fuel_Consumption_City],  # Fix column name
                "Fuel Consumption Hwy (L/100 km)": [self.Fuel_Consumption_Hwy],  # Fix column name
                "Fuel Consumption Comb (L/100 km)": [self.Fuel_Consumption_Comb],  # Fix column name
                "Fuel Consumption Comb (mpg)": [self.Fuel_Consumption_Comb_mpg]  # Fix column name
            }
        
            logging.info(f"Creating DataFrame from input: {custom_data_input_dict}")
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            logging.error(f"An error occurred while creating DataFrame: {e}")
            raise CustomException(e, sys)
