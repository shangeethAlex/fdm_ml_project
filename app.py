from flask import Flask,request,render_template
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
                    
            Make=request.form.get('Make'),
            Model=request.form.get('Model'),
            Vehicle_Class=request.form.get('Vehicle_Class'),
            Engine_Size=request.form.get('Engine_Size'),
            Cylinders=request.form.get('Cylinders'),
            Transmission=float(request.form.get('Transmission')),
            Fuel_Type=float(request.form.get('Fuel_Type')),
            Fuel_Consumption_City=request.form.get('Fuel_Consumption_City'),
            Fuel_Consumption_Hwy=request.form.get('Fuel_Consumption_Hwy'),
            Fuel_Consumption_Comb=request.form.get('Fuel_Consumption_Comb'),
            Fuel_Consumption_Comb_mpg=request.form.get('Fuel_Consumption_Comb_mpg'),
            )
        
        
        
        
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    
    

if __name__=="__main__":
    # app.run(host="0.0.0.0",port=8080)        
    app.run(host='0.0.0.0', port=8080)        
