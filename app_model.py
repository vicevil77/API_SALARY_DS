from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import subprocess
import logging


#EJEMPLO PARA LA PRUEBA:
#ejemplo en clase :  /api/v1/predict?work_year=5&experience_level=Senior&employment_type=Full-time&job_title=Software%20Engineer&salary_currency=USD&salary_in_usd=50000&employee_residence=USA&remote_ratio=0.5&company_location=USA&company_size=100-500
# REENTRENO: /api/v1/retrain/

#AL FIN FUNCIONA WEBHOOK

root_path= "/home/vicevil/DESPLIEGUE_API_SALARIOS/"

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/', methods=['GET'])
def hello(): # Ligado al endopoint "/" o sea el home, con el método GET
    return "<style>" \
           "body { background-image: url('/static/images/fondo.jpg'); }" \
           "</style>" \
           "<h1><b>Enhorabuena, ha sido un placer teneros como alumnos, he aprendido mucho y espero que os sirva, más en lo humano que en lo técnico, pero de primeras en lo técnico. Y es la primera vez que lo escribo.<b><br></h1>" \
           "<br>" \
           "<h1>Puedes elegir entre las siguientes variables, poniendo en la barra de direccion las variables(COL) en el orden establecido separadas de la forma siguiente:</h1>" \
           "<h4>eliges /api/v1/predict o /api/v1/retrain + ?COL=VAL#...COL=VAL#...COL=VAL</h4>" \
            "<br>" \
            "<h2>Por ejemplo:  /api/v1/predict?work_year=5&experience_level=Senior&employment_type=Full-time&job_title=Software%20Engineer&salary_currency=USD&salary_in_usd=50000&employee_residence=USA&remote_ratio=0.5&company_location=USA&company_size=100-500</h2>" \
           "<br>" \
           "<h3>COL: work_years</h2>" \
           "<p>VAL: [2020,2021, 2022]</p>" \
           "<h3>COL:experience_level</h2>" \
           "<p>VAL: ['MI', 'SE', 'EN' ,'EX']</p>" \
           "<h3>COL: employment_type</h2>" \
           "<p>VAL: ['FT', 'CT' ,'PT' ,'FL']</p>" \
           "<h3>COL: job_title</h2>" \
           "<p>VAL: ['Data Scientist', 'Machine Learning Scientist', 'Big Data Engineer', 'Product Data Analyst', 'Machine Learning Engineer', 'Data Analyst', 'Lead Data Scientist', 'Business Data Analyst', 'Lead Data Engineer', 'Lead Data Analyst', 'Data Engineer', 'Data Science Consultant', 'BI Data Analyst', 'Director of Data Science', 'Research Scientist', 'Machine Learning Manager', 'Data Engineering Manager', 'Machine Learning Infrastructure Engineer', 'ML Engineer', 'AI Scientist', 'Computer Vision Engineer', 'Principal Data Scientist', 'Data Science Manager', 'Head of Data', '3D Computer Vision Researcher', 'Data Analytics Engineer', 'Applied Data Scientist', 'Marketing Data Analyst', 'Cloud Data Engineer', 'Financial Data Analyst', 'Computer Vision Software Engineer', 'Director of Data Engineering', 'Data Science Engineer', 'Principal Data Engineer', 'Machine Learning Developer', 'Applied Machine Learning Scientist', 'Data Analytics Manager', 'Head of Data Science', 'Data Specialist', 'Data Architect', 'Finance Data Analyst', 'Principal Data Analyst', 'Big Data Architect', 'Staff Data Scientist', 'Analytics Engineer', 'ETL Developer', 'Head of Machine Learning', 'NLP Engineer']</p>" \
           "<h3>COL: salary_currency</h2>" \
           "<p>VAL: ['EUR' ,'USD' ,'GBP', 'HUF' ,'INR' ,'JPY', 'CNY', 'MXN', 'CAD' ,'DKK' ,'PLN' ,'SGD' , 'CLP' ,'BRL' ,'TRY', 'AUD' ,'CHF']</p>" \
           "<h3>COL: salary_in_usd</h2>" \
           "<p>VAL: [79833, 260000, 109024, 20000, 150000, 72000, 190000, 35735, 135000, 125000, 51321, 40481, 39916, 87000, 85000, 8000, 41689, 114047, 5707, 56000, 43331, 6072, 47899, 98000, 115000, 325000, 42000, 33511, 100000, 117104, 59303, 70000, 68428, 450000, 46759, 74130, 103000, 250000, 10000, 138000, 45760, 50180, 106000, 112872, 15966, 76958, 188000, 105000, 70139, 91000, 45896, 54742, 60000, 148261, 38776, 118000, 120000, 138350, 110000, 130800, 21669, 412000, 45618, 62726, 49268, 190200, 91237, 42197, 82528, 235000, 53192, 5409, 270000, 80000, 79197, 140000, 54238, 47282, 153667, 28476, 59102, 170000, 88654, 76833, 19609, 276000, 29751, 89294, 12000, 95746, 75000, 36259, 62000, 73000, 51519, 187442, 30428, 94564, 113476, 103160, 45391, 225000, 50000, 40189, 90000, 200000, 110037, 10354, 151000, 9466, 40570, 49646, 38400, 24000, 63711, 77364, 220000, 240000, 82500, 82744, 62649, 153000, 160000, 168000, 75774, 13400, 144000, 127221, 119059, 423000, 230000, 28369, 63831, 130026, 165000, 55000, 60757, 174000, 2]</p>" \
           "<h3>COL: employee_residence</h2>" \
           "<p>VAL: ['DE' ,'JP' ,'GB' ,'HN' ,'US', 'HU', 'NZ', 'FR', 'IN', 'PK', 'PL' ,'PT' ,'CN' ,'GR', 'AE', 'NL', 'MX', 'CA' ,'AT', 'NG', 'PH' ,'ES' ,'DK' ,'RU' ,'IT' ,'HR' ,'BG' ,'SG','BR' ,'IQ', 'VN' ,'BE' ,'UA', 'MT' ,'CL' ,'RO' ,'IR' ,'CO' ,'MD', 'KE', 'SI', 'HK','TR', 'RS', 'PR', 'LU', 'JE', 'CZ', 'AR', 'DZ', 'TN', 'MY', 'EE', 'AU', 'BO', 'IE','CH']</p>" \
           "<h3>COL: remote_ratio</h2>" \
           "<p>VAL: [ 0 ,50 ,100]</p>" \
           "<h3>COL: company_location</h2>" \
           "<p>VAL: ['DE', 'JP', 'GB', 'HN', 'US', 'HU', 'NZ', 'FR', 'IN', 'PK', 'CN', 'GR', 'AE', 'NL', 'MX', 'CA', 'AT', 'NG', 'ES', 'PT', 'DK', 'IT', 'HR', 'LU', 'PL', 'SG', 'RO', 'IQ', 'BR', 'BE', 'UA', 'IL', 'RU', 'MT', 'CL', 'IR', 'CO', 'MD', 'KE', 'SI', 'CH', 'VN', 'AS', 'TR', 'CZ', 'DZ', 'EE', 'MY', 'AU', 'IE']</p>" \
           "<h3>COL: company_size</h2>" \
           "<p>VAL: ['L' ,'S' ,'M']</p>"

with open(r'/home/vicevil/DESPLIEGUE_API_SALARIOS/data/salary_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

@app.route('/api/v1/predict', methods=['GET'])
def predict():
    # obtener los datos de la solicitud GET 
    data = {
        "work_year": request.args.get('work_year'),
        "experience_level": request.args.get('experience_level'),
        "employment_type": request.args.get('employment_type'),
        "job_title": request.args.get('job_title'),
        "salary_currency": request.args.get('salary_currency'),
        "salary_in_usd": request.args.get('salary_in_usd'),
        "employee_residence": request.args.get('employee_residence'),
        "remote_ratio": request.args.get('remote_ratio'),
        "company_location": request.args.get('company_location'),
        "company_size": request.args.get('company_size')
    }
    # convertir los datos a DataFrame
    data_df = pd.DataFrame(data, index=[0])
    # preprocesar los datos
    preprocessed_data = pipeline['preprocessor'].transform(data_df)
    # hacer la predicción
    prediction = pipeline['model'].predict(preprocessed_data)
    return {'prediction': prediction[0]}

@app.route('/api/v1/retrain', methods=['GET'])
def retrain():
    # leer los datos del archivo CSV
    data = pd.read_csv(r'/home/vicevil/DESPLIEGUE_API_SALARIOS/data/ds_salaries.csv')
    # dividir los datos en características y variable objetivo
    X = data.drop('salary', axis=1)
    y = data['salary']
    # preprocesar las características del pipieline
    X_preprocessed = pipeline['preprocessor'].transform(X)
    # dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)
    # reentrenar el modelo
    pipeline['model'].fit(X_train, y_train)
    # calcular el error cuadrático medio y el error porcentual absoluto medio
    rmse = np.sqrt(mean_squared_error(y_test, pipeline['model'].predict(X_test)))
    mape = mean_absolute_percentage_error(y_test, pipeline['model'].predict(X_test))
    # guardar el modelo reentrenado
    with open(r'/home/vicevil/DESPLIEGUE_API_SALARIOS/data/retrain/retrained_model.pkl', 'wb') as file:
            pickle.dump(pipeline['model'], file)
    return {'message': 'Model retrained successfully', 'RMSE': rmse, 'MAPE': mape}
@app.route('/webhook_2024', methods=['POST'])
def webhook():
    # Ruta al repositorio donde se realizará el pull
    path_repo = "/home/vicevil/DESPLIEGUE_API_SALARIOS"
    servidor_web = '/var/www/vicevil_pythonanywhere_com_wsgi.py'

    # Comprueba si la solicitud POST contiene datos JSON
    if request.is_json:
        payload = request.json
        # Verifica si la carga útil (payload) contiene información sobre el repositorio
        if 'repository' in payload:
            # Extrae el nombre del repositorio y la URL de clonación
            repo_name = payload['repository']['name']
            clone_url = payload['repository']['clone_url']
            
            # Cambia al directorio del repositorio
            try:
                os.chdir(path_repo)
            except FileNotFoundError:
                return jsonify({'message': 'El directorio del repositorio no existe'}), 404

            # Realiza un git pull en el repositorio
            try:
                subprocess.run(['git', 'pull'], check=True)
                subprocess.run(['touch', servidor_web], check=True) # Trick to automatically reload PythonAnywhere WebServer
                return jsonify({'message': f'Se realizó un git pull en el repositorio {repo_name}'}), 200
            except subprocess.CalledProcessError:
                return jsonify({'message': f'Error al realizar git pull en el repositorio {repo_name}'}), 500
        else:
            return jsonify({'message': 'No se encontró información sobre el repositorio en la carga útil (payload)'}), 400
    else:
        return jsonify({'message': 'La solicitud no contiene datos JSON'}), 400
    
    
if __name__ == '__main__':
    app.run()