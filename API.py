# Importación de librerías
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from werkzeug.utils import cached_property
from flask_restx import Api, Resource, fields
import joblib
import traceback

# Cargar el modelo
model = joblib.load('modelo.pkl')

# Función para quitar tildes de las cadenas
def quitartildes(column):
    a, b = 'áéíóúüñÁÉÍÓÚÜàèìòù', 'aeiouunAEIOUUaeiou'
    trans = str.maketrans(a, b)
    column = column.str.strip().str.upper().str.translate(trans)
    return column

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
            X_input = pd.DataFrame([data])

            # Aplicar preprocesamiento
            current_year = datetime.now().year
            X_input['State'] = quitartildes(X_input['State'])
            X_input['Model'] = quitartildes(X_input['Model'])
            X_input['Make'] = quitartildes(X_input['Make'])
            X_input['Car_Age'] = current_year - X_input['Year']
            X_input['Mileage_Year'] = X_input['Year'] / X_input['Mileage']
            X_input['Brand_Model'] = X_input['Make'] + '_' + X_input['Model']

            q1 = X_input['Mileage'].quantile(0.25)
            q2 = X_input['Mileage'].quantile(0.5)
            q3 = X_input['Mileage'].quantile(0.75)

            rango_bajo = X_input['Mileage'] <= q1
            rango_medio = (X_input['Mileage'] > q1) & (X_input['Mileage'] <= q2)
            rango_alto = (X_input['Mileage'] > q2) & (X_input['Mileage'] <= q3)
            rango_muy_alto = X_input['Mileage'] > q3

            X_input['mileage_range'] = ''
            X_input.loc[rango_bajo, 'mileage_range'] = 'Bajo'
            X_input.loc[rango_medio, 'mileage_range'] = 'Medio'
            X_input.loc[rango_alto, 'mileage_range'] = 'Alto'
            X_input.loc[rango_muy_alto, 'mileage_range'] = 'Muy Alto'

            X_input = X_input.drop(["Mileage","Make"], axis=1)

            # Definir el preprocesamiento
            numeric_features = X_input.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X_input.select_dtypes(include=['object']).columns

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)])

            pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

            # Aplicar el pipeline a los datos
            X_input_preprocessed = pipeline.fit_transform(X_input)

            # Realizar la predicción
            prediction = model.predict(X_input_preprocessed)

            # Devolver la predicción
            return {'prediction': prediction.tolist()}

        except Exception as e:
            traceback.print_exc()
            return {'error': 'Internal Server Error'}, 500

# Ejecutar la aplicación en el puerto 5000
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
