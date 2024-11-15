# Vehicle CO2 Emission Prediction Model

## Project Overview

This project is focused on predicting the CO2 emissions of vehicles based on various input features like engine size, fuel type, weight, and driving conditions. We applied a variety of machine learning algorithms including RNN, Linear Regression, Decision Trees, SVM, and Random Forest to create a model capable of providing accurate emission predictions. The model is deployed through **Streamlit**, providing an interactive web-based interface for users to input vehicle information and receive CO2 emission predictions in real-time.

## Technologies Used

- **Machine Learning**: 
  - **RNN** (Recurrent Neural Networks)
  - **Linear Regression**
  - **Decision Trees**
  - **Support Vector Machine (SVM)**
  - **Random Forest**
  
- **Deployment**:
  - **Streamlit** for creating an interactive, user-friendly web interface.

- **Data Handling**:
  - **Pandas** and **NumPy** for data manipulation and analysis.

- **Visualization**:
  - **Matplotlib** and **Seaborn** for visualizing trends and model performance.

## Features

- **Real-Time Emission Prediction**: Users can input vehicle specifications like engine type, fuel type, and weight, and receive an estimated CO2 emission prediction.
- **Interactive UI**: Built using **Streamlit**, providing an easy-to-use interface.
- **Model Comparison**: The project compares the performance of different machine learning models, showcasing the best fit for emission prediction.

## Data Preprocessing

The dataset was preprocessed to improve the quality of the data and ensure optimal model performance. The following preprocessing steps were applied:

1. **Handling Missing Values**: 
   - We identified and handled missing values in the dataset by using imputation techniques. For numerical features, missing values were filled with the **mean** (for continuous variables) or the **mode** (for categorical variables).
   - Any rows with critical missing data (e.g., missing fuel type or CO2 emission values) were removed to ensure data integrity.

2. **Dealing with Misspelled Words**:
   - We identified potential misspelled entries in categorical columns such as fuel type (e.g., "regualr" instead of "regular") and corrected them using string matching techniques.
   - A dictionary of common misspellings was used to ensure consistency across the dataset.

3. **Case Normalization**:
   - To standardize the dataset, all categorical variables (such as fuel type and transmission) were converted to lowercase. This helped to eliminate inconsistencies caused by case variations (e.g., "Gasoline" and "gasoline" were treated as the same).

4. **Handling Outliers**:
   - We detected and handled outliers in numerical features like CO2 emissions, engine size, and fuel consumption. Outliers were either capped or removed based on statistical analysis (such as using the IQR method).
   
5. **Feature Encoding**:
   - Categorical variables (e.g., fuel type, transmission) were encoded using **Label Encoding** for machine learning algorithms that require numerical input.
   - We also performed one-hot encoding for certain categorical features where appropriate (e.g., **fuel type**).

6. **Scaling Numerical Features**:
   - Numerical features like weight, engine size, and fuel consumption were scaled using **StandardScaler** to ensure all features contributed equally to the model performance.

7. **Feature Engineering**:
   - Additional features such as **engine power-to-weight ratio** and **fuel efficiency** were created to improve model predictions.

## Challenges Faced

- **Data Preprocessing**: Handling missing or inconsistent data was a challenge, as some vehicle specifications were incomplete in the dataset. We used feature engineering and data imputation techniques to handle this.
- **Model Performance**: Achieving high accuracy across different models required extensive hyperparameter tuning. Some algorithms (like Decision Trees and Random Forest) performed better than others, while fine-tuning the SVM model presented some initial difficulties.
- **Model Interpretability**: Understanding how different features influenced the emission prediction was a challenge, but we used tools like feature importance and SHAP values to gain insights.

## Future Improvements

- **Enhanced User Input**: Adding more detailed vehicle-specific features like driving behavior (city vs highway) and maintenance history to improve prediction accuracy.
- **Model Optimization**: Testing additional machine learning models like XGBoost and deep learning techniques to further improve prediction accuracy.
- **Real-time Data Integration**: Implementing a system that pulls real-time vehicle data to make dynamic predictions based on current conditions.

## CONTENT

This dataset captures the details of how CO2 emissions by a vehicle can vary with the different features. The dataset has been taken from Canada Government official open data website. This is a compiled version. It contains data over a period of 7 years.

There are total 7385 rows and 12 columns. There are a few abbreviations that have been used to describe the features. I am listing them out here. The same can be found in the Data Description sheet.

### Model

- **4WD/4X4** = Four-wheel drive
- **AWD** = All-wheel drive
- **FFV** = Flexible-fuel vehicle
- **SWB** = Short wheelbase
- **LWB** = Long wheelbase
- **EWB** = Extended wheelbase

### Transmission

- **A** = Automatic
- **AM** = Automated manual
- **AS** = Automatic with select shift
- **AV** = Continuously variable
- **M** = Manual
- **3 - 10** = Number of gears

### Fuel type

- **X** = Regular gasoline
- **Z** = Premium gasoline
- **D** = Diesel
- **E** = Ethanol (E85)
- **N** = Natural gas

### Fuel Consumption

City and highway fuel consumption ratings are shown in litres per 100 kilometres (L/100 km) - the combined rating (55% city, 45% hwy) is shown in L/100 km and in miles per gallon (mpg).

### CO2 Emissions

The tailpipe emissions of carbon dioxide (in grams per kilometre) for combined city and highway driving.

## Conclusion

This project was a great learning experience that combined machine learning, data preprocessing, and model deployment. It also allowed us to contribute to the growing field of environmental sustainability by using data science to predict and understand CO2 emissions from vehicles.

## ACKNOWLEDGEMENTS

The data has been taken and compiled from the below Canada Government official link:  
[https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64#wb-auto-6](https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64#wb-auto-6)
