# Importación de librerías
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from werkzeug.utils import cached_property
import pandas as pd
import numpy as np
import joblib
import traceback

# Función para quitar tildes de las cadenas
# def quitartildes(column):
#     a, b = 'áéíóúüñÁÉÍÓÚÜàèìòù', 'aeiouunAEIOUUaeiou'
#     trans = str.maketrans(a, b)
#     column = column.str.strip().str.upper().str.translate(trans)
#     return column

# Crear la aplicación Flask con el nombre "api_grupo5"
app = Flask("api_grupo5")

# Definir la API Flask con Flask-Restx
api = Api(
    app,
    version='1.0',
    title='Car Price Prediction API',
    description='Predict the price of a car based on its features'
)

# Definir los parámetros de entrada
price_predict_model = api.model('PricePredictModel', {
    'Year': fields.Integer(required=True, description='Year of the car'),
    'Mileage': fields.Integer(required=True, description='Mileage of the car'),
    'State': fields.String(required=True, description='State of the car'),
    'Make': fields.String(required=True, description='Make of the car'),
    'Model': fields.String(required=True, description='Model of the car')
})

# Definir la ruta para la API
@api.route('/predict')
class PricePrediction(Resource):
    @api.expect(price_predict_model)
    def post(self):
        try:
            # Obtener los datos de entrada
            data = request.json
            # Cargar el modelo
            model = joblib.load('stacking_regressor.pkl')

                        
            # Cargar el pipeline
            pipeline_loaded = joblib.load('pipeline_model.pkl')

            # Aplicar preprocesamiento
            current_year = datetime.now().year
            # data['State'] = quitartildes(data['State'])
            # data['Model'] = quitartildes(data['Model'])
            # data['Make'] = quitartildes(data['Make'])
            data['Car_Age'] = current_year - data['Year']
            data['Mileage_Year'] = data['Year'] / data['Mileage']
            data['Brand_Model'] = data['Make'] + '_' + data['Model']
            # Definir el rango de millaje
            mileage = data['Mileage']
            mileage_range = 'Bajo' if mileage <= 25000 else \
                        'Medio' if mileage <= 50000 else \
                        'Alto' if mileage <= 75000 else \
                        'Muy Alto'
            data['mileage_range'] = mileage_range

            # Eliminar columnas no necesarias
            data.pop('Mileage')
            data.pop('Make')

            # Crear DataFrame a partir de los datos
            X_input = pd.DataFrame([data])

            # Aplicar el pipeline cargado a los datos
            X_preprocessed = pipeline_loaded.transform(X_input)

            # Realizar la predicción
            prediction = model.predict(X_preprocessed)

            # Devolver la predicción
            return {'prediction': prediction.tolist()}

        except Exception as e:
            traceback.print_exc()
            return {'error': str(e)}, 500

# Ejecutar la aplicación en el puerto 5000
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
    
