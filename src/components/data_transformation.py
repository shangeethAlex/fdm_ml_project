import sys
from dataclasses import dataclass
import scipy
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ['Engine Size(L)','Cylinders','Fuel Consumption City (L/100 km)','Fuel Consumption Hwy (L/100 km)','Fuel Consumption Comb (L/100 km)','Fuel Consumption Comb (mpg)']
            categorical_columns = ['Make', 'Model','Vehicle Class','Transmission','Fuel Type']

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="CO2 Emissions(g/km)"
            numerical_columns = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)', 'Fuel Consumption Comb (mpg)']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            print("Shape of input_feature_train_arr:", input_feature_train_arr.shape)
            print("Type of input_feature_train_arr:", type(input_feature_train_arr))
            print("Data type of input_feature_train_arr:", input_feature_train_arr.dtype)
        
            # Convert input_feature arrays to dense if they're sparse
            if scipy.sparse.issparse(input_feature_train_arr):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if scipy.sparse.issparse(input_feature_test_arr):
                input_feature_test_arr = input_feature_test_arr.toarray()

            print("Shape of input_feature_train_arr after conversion:", input_feature_train_arr.shape)
            print("Data type of input_feature_train_arr after conversion:", input_feature_train_arr.dtype)

            print("Shape of target_feature_train_df:", target_feature_train_df.shape)

            # Ensure target_feature arrays are 2D
            target_feature_train_arr = np.array(target_feature_train_df).reshape(-1, 1)
            target_feature_test_arr = np.array(target_feature_test_df).reshape(-1, 1)

            print("Shape of target_feature_train_arr after reshape:", target_feature_train_arr.shape)

            # Combine features and target
            train_arr = np.column_stack((input_feature_train_arr, target_feature_train_arr))
            test_arr = np.column_stack((input_feature_test_arr, target_feature_test_arr))

            print("Shape of train_arr:", train_arr.shape)
            print("Shape of test_arr:", test_arr.shape)

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)

            
            
            
            
            
            
