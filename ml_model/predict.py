"""
Customer Churn Prediction - Prediction Service
================================================
This script handles predictions for the API.
"""

import pandas as pd
import numpy as np
import pickle
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def load_model(model_name='random_forest'):
    """Load a trained model by name."""
    model_file = os.path.join(MODELS_DIR, f'{model_name}.pkl')
    if os.path.exists(model_file):
        return pickle.load(open(model_file, 'rb'))
    return None

def load_scaler():
    """Load the scaler."""
    scaler_file = os.path.join(MODELS_DIR, 'scaler.pkl')
    if os.path.exists(scaler_file):
        return pickle.load(open(scaler_file, 'rb'))
    return None

def load_encoders():
    """Load label encoders."""
    encoder_file = os.path.join(MODELS_DIR, 'label_encoders.pkl')
    if os.path.exists(encoder_file):
        return pickle.load(open(encoder_file, 'rb'))
    return None

def predict_churn(customer_data, model_name='random_forest'):
    """
    Predict churn for a customer.
    
    Args:
        customer_data: dict with keys: Age, Gender, Tenure, Usage Frequency, 
                       Support Calls, Payment Delay, Subscription Type, 
                       Contract Length, Total Spend, Last Interaction
        model_name: name of the model to use
    
    Returns:
        dict with prediction and probability
    """
    model = load_model(model_name)
    scaler = load_scaler()
    encoders = load_encoders()
    
    if model is None or scaler is None:
        return {'error': 'Model not found'}
    
    # Create dataframe
    df = pd.DataFrame([customer_data])
    
    # Encode categorical variables
    categorical_cols = ['Gender', 'Subscription Type', 'Contract Length']
    for col in categorical_cols:
        if col in df.columns and col in encoders:
            df[col] = encoders[col].transform(df[col])
    
    # Ensure correct column order
    feature_names = pickle.load(open(os.path.join(MODELS_DIR, 'feature_names.pkl'), 'rb'))
    df = df[feature_names]
    
    # Scale
    X_scaled = scaler.transform(df)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0] if hasattr(model, 'predict_proba') else [0, 1] if prediction else [1, 0]
    
    return {
        'prediction': int(prediction),
        'churn_probability': float(round(probability[1] * 100, 2)),
        'retention_probability': float(round(probability[0] * 100, 2)),
        'risk_level': 'High' if probability[1] > 0.7 else 'Medium' if probability[1] > 0.4 else 'Low',
        'model_used': model_name
    }

if __name__ == '__main__':
    # Test prediction
    test_customer = {
        'Age': 35,
        'Gender': 'Male',
        'Tenure': 12,
        'Usage Frequency': 10,
        'Support Calls': 5,
        'Payment Delay': 15,
        'Subscription Type': 'Basic',
        'Contract Length': 'Monthly',
        'Total Spend': 500,
        'Last Interaction': 7
    }
    
    result = predict_churn(test_customer)
    print(json.dumps(result, indent=2))
