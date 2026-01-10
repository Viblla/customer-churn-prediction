"""
Customer Churn Prediction - Flask Dashboard
============================================
A beautiful, interactive web dashboard for ML model visualization
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import os
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE_DIR, 'ml_model')
RESULTS_DIR = os.path.join(ML_DIR, 'results')
MODELS_DIR = os.path.join(ML_DIR, 'models')

def load_dashboard_data():
    """Load the dashboard data from JSON file"""
    try:
        with open(os.path.join(RESULTS_DIR, 'dashboard_data.json'), 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def load_model(model_name):
    """Load a specific model"""
    model_file = os.path.join(MODELS_DIR, f'{model_name}.pkl')
    if os.path.exists(model_file):
        return pickle.load(open(model_file, 'rb'))
    return None

def load_preprocessors():
    """Load scaler and encoders"""
    scaler = pickle.load(open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb'))
    encoders = pickle.load(open(os.path.join(MODELS_DIR, 'label_encoders.pkl'), 'rb'))
    features = pickle.load(open(os.path.join(MODELS_DIR, 'feature_names.pkl'), 'rb'))
    return scaler, encoders, features

# =====================================================
# WEB ROUTES
# =====================================================

@app.route('/')
def index():
    """Main dashboard page"""
    data = load_dashboard_data()
    if data is None:
        return render_template('error.html', message="Dashboard data not found. Please run the ML training first.")
    return render_template('dashboard.html', data=data)

@app.route('/predict')
def predict_page():
    """Prediction page"""
    data = load_dashboard_data()
    models = list(data['model_results'].keys()) if data else []
    return render_template('predict.html', models=models, data=data)

# =====================================================
# API ROUTES
# =====================================================

@app.route('/api/dashboard')
def api_dashboard():
    """Get all dashboard data"""
    data = load_dashboard_data()
    if data:
        return jsonify({'success': True, 'data': data})
    return jsonify({'success': False, 'message': 'Data not found'}), 404

@app.route('/api/models')
def api_models():
    """Get model results"""
    data = load_dashboard_data()
    if data:
        return jsonify({
            'success': True,
            'data': {
                'model_results': data['model_results'],
                'model_ranking': data['model_ranking'],
                'best_model': data['best_model']
            }
        })
    return jsonify({'success': False}), 404

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Predict churn for a customer"""
    try:
        customer_data = request.json
        model_name = request.args.get('model', 'logistic_regression')
        
        # Load model and preprocessors
        model = load_model(model_name)
        if model is None:
            return jsonify({'success': False, 'message': f'Model {model_name} not found'}), 404
        
        scaler, encoders, feature_names = load_preprocessors()
        
        # Create DataFrame
        df = pd.DataFrame([customer_data])
        
        # Encode categorical columns
        categorical_cols = ['Gender', 'Subscription Type', 'Contract Length']
        for col in categorical_cols:
            if col in df.columns and col in encoders:
                try:
                    df[col] = encoders[col].transform(df[col].astype(str))
                except ValueError:
                    df[col] = 0
        
        # Ensure correct column order
        df = df[feature_names]
        
        # Scale features
        X_scaled = scaler.transform(df)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        
        # Get probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_scaled)[0]
            churn_prob = float(proba[1]) * 100
            stay_prob = float(proba[0]) * 100
        else:
            churn_prob = float(prediction) * 100
            stay_prob = (1 - float(prediction)) * 100
        
        # Risk level
        if churn_prob >= 70:
            risk_level = 'High'
            risk_color = '#ef4444'
        elif churn_prob >= 40:
            risk_level = 'Medium'
            risk_color = '#f59e0b'
        else:
            risk_level = 'Low'
            risk_color = '#10b981'
        
        # Recommendations
        recommendations = get_recommendations(customer_data, churn_prob)
        
        return jsonify({
            'success': True,
            'data': {
                'prediction': int(prediction),
                'churn_probability': round(churn_prob, 2),
                'stay_probability': round(stay_prob, 2),
                'risk_level': risk_level,
                'risk_color': risk_color,
                'model_used': model_name,
                'recommendations': recommendations
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

def get_recommendations(customer_data, churn_prob):
    """Generate retention recommendations"""
    recs = []
    
    if customer_data.get('Support Calls', 0) > 4:
        recs.append({'icon': '🔧', 'text': 'High support calls - Improve product experience'})
    
    if customer_data.get('Payment Delay', 0) > 15:
        recs.append({'icon': '💳', 'text': 'Payment delays - Offer flexible payment plans'})
    
    if customer_data.get('Tenure', 0) < 6:
        recs.append({'icon': '🎁', 'text': 'New customer - Provide onboarding incentives'})
    
    if customer_data.get('Usage Frequency', 0) < 5:
        recs.append({'icon': '📊', 'text': 'Low usage - Send engagement campaigns'})
    
    if customer_data.get('Contract Length') == 'Monthly':
        recs.append({'icon': '📝', 'text': 'Monthly contract - Offer annual discount'})
    
    if churn_prob >= 70:
        recs.append({'icon': '🚨', 'text': 'HIGH RISK - Immediate intervention required'})
        recs.append({'icon': '📞', 'text': 'Schedule personal outreach call'})
    elif churn_prob >= 40:
        recs.append({'icon': '⚠️', 'text': 'Medium risk - Proactive engagement recommended'})
    
    if not recs:
        recs.append({'icon': '✅', 'text': 'Customer appears satisfied - Maintain current service'})
    
    return recs

if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     Customer Churn Prediction Dashboard                    ║
    ║     Running on http://localhost:5000                       ║
    ╠════════════════════════════════════════════════════════════╣
    ║  Pages:                                                    ║
    ║    /          - Main Dashboard                             ║
    ║    /predict   - Churn Prediction Tool                      ║
    ║                                                            ║
    ║  API Endpoints:                                            ║
    ║    GET  /api/dashboard  - All dashboard data               ║
    ║    GET  /api/models     - Model results                    ║
    ║    POST /api/predict    - Predict customer churn           ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    app.run(debug=True, host='0.0.0.0', port=5000)
