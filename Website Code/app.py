import csv
import io
from flask import Flask, request, render_template, redirect, url_for
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from werkzeug.utils import secure_filename
from helper import manufacturing_test, naval_test, plot_predictions_manufacturing, plot_predictions_naval, preprocess_sensor_data_hydraulic, hydraulic_test, plot_predictions, bearing_test, plot_predictions_bearing

# Create a Flask app instance
app = Flask(__name__)

# Set upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define a route for the root URL


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/hydraulic', methods=['GET', 'POST'])
def hydraulic():
    if request.method == 'POST':
        # print(request.files)
        if 'folder[]' not in request.files:
            return redirect(request.url)
        folder = request.files.getlist('folder[]')
        data = {}
        # print(folder)
        for file in folder:
            print(file.filename[:-4])
            decoded_file = file.read().decode("utf-8")
            # print(decoded_file[0:20])
            # Use StringIO to make the string behave like a file for pandas
            file_like_object = io.StringIO(decoded_file)
            
            # Read the decoded file contents into a DataFrame
            raw_data = pd.read_csv(file_like_object, header=None)
            data[file.filename[:-4]] = preprocess_sensor_data_hydraulic(raw_data)
        # print(data)
        # print(folder)
        features = []
        for sensor_name, sensor_data in data.items():
            stats = pd.DataFrame({
                f'{sensor_name}_mean': sensor_data.mean(axis=1),
                f'{sensor_name}_std': sensor_data.std(axis=1),
                f'{sensor_name}_max': sensor_data.max(axis=1),
                f'{sensor_name}_min': sensor_data.min(axis=1),
                f'{sensor_name}_range': sensor_data.max(axis=1) - sensor_data.min(axis=1),
                f'{sensor_name}_rms': np.sqrt((sensor_data ** 2).mean(axis=1))
            })
            features.append(stats)

        X = pd.concat(features, axis=1)
        column_order = [
            'CE_mean', 'CE_std', 'CE_max', 'CE_min', 'CE_range', 'CE_rms',
            'CP_mean', 'CP_std', 'CP_max', 'CP_min', 'CP_range', 'CP_rms',
            'SE_mean', 'SE_std', 'SE_max', 'SE_min', 'SE_range', 'SE_rms',
            'PS6_mean', 'PS6_std', 'PS6_max', 'PS6_min', 'PS6_range', 'PS6_rms',
            'PS2_mean', 'PS2_std', 'PS2_max', 'PS2_min', 'PS2_range', 'PS2_rms',
            'FS1_mean', 'FS1_std', 'FS1_max', 'FS1_min', 'FS1_range', 'FS1_rms',
            'FS2_mean', 'FS2_std', 'FS2_max', 'FS2_min', 'FS2_range', 'FS2_rms',
            'EPS1_mean', 'EPS1_std', 'EPS1_max', 'EPS1_min', 'EPS1_range',
            'EPS1_rms', 'TS4_mean', 'TS4_std', 'TS4_max', 'TS4_min', 'TS4_range',
            'TS4_rms', 'TS2_mean', 'TS2_std', 'TS2_max', 'TS2_min', 'TS2_range',
            'TS2_rms'
        ]

        # Reorder the DataFrame columns
        X = X[column_order]
        # if folder and allowed_file(folder.filename):
        #     # Save the uploaded folder
        #     folder_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(folder.filename))
        #     folder.save(folder_path)
            
        # Preprocess and predict
        print(X.columns)
        predictions = hydraulic_test(X)
        print(predictions)
        # Generate graphs from predictions
        img_base64 = plot_predictions(predictions)
        current_state = "Failure" if predictions[-1][4]==1 else "Stable"
        
        # Return the HTML page with predictions and graph
        return render_template('/hydraulic.html', predictions=predictions, img_base64=img_base64, current_state=current_state)
    
    return render_template('/hydraulic.html', predictions=[['','','','','']], img_base64="", current_state="")

@app.route('/bearing', methods=['GET', 'POST'])
def bearing():
    if request.method == 'POST':
        # print(request.files)
        if 'inp_file' not in request.files:
            return redirect(request.url)
        file = request.files.get('inp_file')
        decoded_file = file.read().decode("utf-8")
        file_like_object = io.StringIO(decoded_file)
        data = pd.read_csv(file_like_object)
        features = ['max', 'min', 'mean', 'sd', 'rms', 'skewness', 'kurtosis', 'crest', 'form']
        X = data[features]
        # reader = csv.reader(decoded_file)
        # header = next(reader)  
        # print(header)
        
        print(X.columns)
        predictions, labelled_predictions = bearing_test(X)
        print(predictions)
        # # Generate graphs from predictions
        img_base64_1, img_base64_2, class_counts = plot_predictions_bearing(labelled_predictions)
        print(class_counts)
        # Return the HTML page with predictions and graph
        return render_template('/ball_bearing.html', labelled_predictions=labelled_predictions, img_base64_1=img_base64_1, img_base64_2=img_base64_2, class_counts=class_counts)
    
    return render_template('/ball_bearing.html', labelled_predictions=['','','','',''], img_base64_1="", img_base64_2="", class_counts=None)

@app.route('/manufacturing', methods=['GET', 'POST'])
def manufacturing():
    if request.method == 'POST':
        # print(request.files)
        if 'inp_file' not in request.files:
            return redirect(request.url)
        file = request.files.get('inp_file')
        decoded_file = file.read().decode("utf-8")
        file_like_object = io.StringIO(decoded_file)
        
        data = pd.read_csv(file_like_object)
        # data = pd.read_csv(f"predictive_maintenance.csv")
        features = ['UDI', 'Type', 'Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
        df = data[features]

        label_encoder = f"models/Manufacturing/label_encoder.pkl"
        with open(label_encoder, 'rb') as file:
            loaded_label_encoder = pickle.load(file)

        df['Type'] = loaded_label_encoder.transform(df['Type'])
        X = df
        
        print(X.columns)
        predictions = manufacturing_test(X)
        print(predictions)
        img_base64 = plot_predictions_manufacturing(predictions)
        current_state = "Failure" if predictions[-1]==1 else "No Failure"
        return render_template('/manufacturing.html', predictions=predictions, img_base64=img_base64, current_state=current_state )
    
    return render_template('/manufacturing.html', predictions=['','','','',''], img_base64="", current_state="")

@app.route('/naval', methods=['GET', 'POST'])
def naval():
    if request.method == 'POST':
        # print(request.files)
        if 'inp_file' not in request.files:
            return redirect(request.url)
        file = request.files.get('inp_file')
        decoded_file = file.read().decode("utf-8")
        file_like_object = io.StringIO(decoded_file)
        
        data = pd.read_csv(file_like_object)
        features = ['Lever position','Ship speed', 'Gas Turbine shaft torque', 'Gas Turbine rate of revolutions', 'Gas Generator rate of revolutions', 'Starboard Propeller Torque', 'Port Propeller Torque',\
             'HP Turbine exit temperature', 'GT Compressor inlet air temperature', 'GT Compressor outlet air temperature', 'HP Turbine exit pressure', 'GT Compressor inlet air pressure', 'GT Compressor outlet air pressure',\
             'Gas Turbine exhaust gas pressure', 'Turbine Injecton Control', 'Fuel flow']
        data.columns = features
        data = data.drop(['GT Compressor inlet air temperature','GT Compressor inlet air pressure'],axis=1)
        X = data
        
        print(X.columns)
        predictions = naval_test(X)
        print(predictions)
        img_base64 = plot_predictions_naval(predictions)
        # current_state = "Failure" if predictions[-1]==1 else "No Failure"
        current_state = predictions[-1]
        return render_template('/naval.html', predictions=predictions, img_base64=img_base64, current_state=current_state )
    
    return render_template('/naval.html', predictions=['','','','',''], img_base64="", current_state=["", ""])

@app.route('/')
def index():
    return render_template('home.html')

if __name__ == '__main__':
    # Run the Flask app on localhost:5000
    app.run(debug=True)
