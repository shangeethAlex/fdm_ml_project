import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,load_object

class PredictPipeline:
    def __init__(self):
        pass


    def predict(self,features): #to do predictions
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/proprocessor.pkl'
            model  = load_object(file_path=model_path)
            preprocessor  = load_object(file_path=preprocessor_path)

            #scale the data
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        

      


class CustomData:
    def __init__(self,
                 Make:str,
                 Model:str,
                 Vehicle_Class:str,
                 Engine_Size:float,
                 Cylinders:float,
                 Transmission:str,
                 Fuel_Type:str,
                 Fuel_Consumption_City:float,
                 Fuel_Consumption_Hwy:float,
                 Fuel_Consumption_Comb:float,
                 Fuel_Consumption_Comb_mpg:float
                 ):
        
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
                "Make":[self.Make],
                "Model":[self.Model],
                "Vehicle_Class":[self.Vehicle_Class],
                "Engine_Size":[self.Engine_Size],
                "Cylinders":[self.Cylinders],
                "Transmission":[self.Transmission],
                "Fuel_Type":[self.Fuel_Type],
                "Fuel_Consumption_City":[self.Fuel_Consumption_City],
                "Fuel_Consumption_Hwy":[self.Fuel_Consumption_Hwy],
                "Fuel_Consumption_Comb":[self.Fuel_Consumption_Comb],
                "Fuel_Consumption_Comb_mpg":[self.Fuel_Consumption_Comb_mpg]
            }

            return pd.DataFrame(custom_data_input_dict)
        except:
            pass
    
        