from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from pycaret.classification import load_model as load_classification_model, predict_model as predict_classification
from pycaret.regression import load_model as load_regression_model, predict_model as predict_regression

app = Flask(__name__)

# Load models
mushroom_model = load_classification_model('mushroom_classifier')
price_model = load_regression_model('price_pipeline')
anomaly_model = joblib.load('final_iforest_model.pkl')

# Load encoders
div_name_encoder = joblib.load('div_name_encoder.pkl')
merchant_encoder = joblib.load('merchant_encoder.pkl')
cat_desc_encoder = joblib.load('cat_desc_encoder.pkl')

# Define columns for different models
mushroom_cols = [
    'odor', 'gill-size', 'gill-color', 'cap-color', 'bruises', 
    'spore-print-color', 'gill-spacing', 'ring-type', 'stalk-surface-above-ring'
]

price_cols = [
    'flat_type', 'storey_range', 'floor_area_sqm', 
    'flat_model', 'cbd_dist', 'min_dist_mrt', 'lease_left'
]

anomaly_cols = ['FISCAL_YR', 'FISCAL_MTH', 'DIV_NAME', 'MERCHANT', 'CAT_DESC', 'AMT', 'Year', 'Month', 'DayOfWeek', 'FiscalQuarter']

# Home route
@app.route('/')
def home():
    return render_template("index.html")  # Default to the mushroom prediction page

# Predict mushroom class
@app.route('/predict_mushroom', methods=['GET', 'POST'])
def predict_mushroom():
    try:
        int_features = [x for x in request.form.values()]
        final = np.array(int_features)
        data_unseen = pd.DataFrame([final], columns=mushroom_cols)
        
        prediction = predict_classification(mushroom_model, data=data_unseen, round=0)
        predicted_class = prediction['Label'][0] if 'Label' in prediction.columns else prediction['prediction_label'][0]
        
        return render_template('index.html', pred1='Predicted Mushroom Class is {}'.format(predicted_class))
    except Exception as e:
        return render_template('index.html', pred1=f'Error in prediction: {str(e)}')

# Predict house price
@app.route('/predict_price', methods=['GET', 'POST'])
def predict_price():
    try:
        int_features = [x for x in request.form.values()]
        data_unseen = pd.DataFrame([int_features], columns=price_cols)
        
        for col in price_cols:
            if col not in data_unseen.columns:
                data_unseen[col] = np.nan

        prediction = predict_regression(price_model, data=data_unseen)
        prediction_value = prediction['prediction_label'][0]
        formatted_value = "${:,.2f}".format(prediction_value)
        
        return render_template('home.html', pred=f'Predicted Resale price will be {formatted_value}')
    except Exception as e:
        return render_template('home.html', pred=f'')

# API Endpoints for both models
@app.route('/mushroom_predict_api', methods=['POST'])
def mushroom_predict_api():
    try:
        data = request.get_json(force=True)
        data_unseen = pd.DataFrame([data])
        
        prediction = predict_classification(mushroom_model, data=data_unseen)
        output = prediction['Label'][0] if 'Label' in prediction.columns else prediction['prediction_label'][0]
    except Exception as e:
        output = f'Error in prediction: {str(e)}'
    
    return jsonify(output)

@app.route('/price_predict_api', methods=['GET', 'POST'])
def price_predict_api():
    try:
        data = request.get_json(force=True)
        data_unseen = pd.DataFrame([data])
        
        for col in price_cols:
            if col not in data_unseen.columns:
                data_unseen[col] = np.nan
        
        prediction = predict_regression(price_model, data=data_unseen)
        output = prediction['prediction_label'][0]
    except Exception as e:
        output = f''
    
    return jsonify(output)

# Predict transaction anomaly
@app.route('/predict_anomaly', methods=['POST', 'GET'])
def predict_anomaly():
    try:
        # Extract form data
        int_features = [request.form['FISCAL_YR'],
                        request.form['FISCAL_MTH'],
                        request.form['DIV_NAME'],
                        request.form['MERCHANT'],
                        request.form['CAT_DESC'],
                        float(request.form['AMT']),
                        int(request.form['Year']),
                        int(request.form['Month']),
                        int(request.form['DayOfWeek']),
                        int(request.form['FiscalQuarter'])]

        # Encode categorical inputs
        div_name_encoded = encode_input(int_features[2], div_name_encoder, "Division name")
        merchant_encoded = encode_input(int_features[3], merchant_encoder, "Merchant")
        cat_desc_encoded = encode_input(int_features[4], cat_desc_encoder, "Category description")

        # Create DataFrame for the input data
        data_unseen = pd.DataFrame([[int_features[0], int_features[1], div_name_encoded, merchant_encoded, cat_desc_encoded, 
                                      int_features[5], int_features[6], int_features[7], int_features[8], int_features[9]]], 
                                    columns=anomaly_cols)

        # Make a prediction
        prediction = anomaly_model.predict(data_unseen)

        # Convert prediction to human-readable format
        result = "This is a normal transaction." if prediction[0] == 1 else "This transaction is anomalous!"

        return render_template('index1.html', pred2=result)

    except Exception as e:
        return render_template('index1.html', pred2=f"")

# Encode input with handling for unseen categories
def encode_input(input_value, encoder, input_name):
    try:
        return encoder.transform([input_value])[0]
    except ValueError:
        return -1  # Assign an "Unknown" category label or a default encoding


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)