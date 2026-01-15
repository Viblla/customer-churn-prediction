/**
 * Customer Churn Prediction - Backend API Server
 * ================================================
 * Express.js server providing API endpoints for the dashboard
 */

const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { PythonShell } = require('python-shell');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Paths
const ML_RESULTS_PATH = path.join(__dirname, '..', 'ml_model', 'results', 'dashboard_data.json');
const ML_MODEL_PATH = path.join(__dirname, '..', 'ml_model');

// Helper: Read dashboard data
function getDashboardData() {
    try {
        if (fs.existsSync(ML_RESULTS_PATH)) {
            const data = fs.readFileSync(ML_RESULTS_PATH, 'utf8');
            return JSON.parse(data);
        }
        return null;
    } catch (error) {
        console.error('Error reading dashboard data:', error);
        return null;
    }
}

// =====================================================
// API ROUTES
// =====================================================

// Health check
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', message: 'Server is running' });
});

// Get all dashboard data
app.get('/api/dashboard', (req, res) => {
    const data = getDashboardData();
    if (data) {
        res.json({ success: true, data });
    } else {
        res.status(404).json({ 
            success: false, 
            message: 'Dashboard data not found. Please run the ML training first.' 
        });
    }
});

// Get data statistics
app.get('/api/stats', (req, res) => {
    const data = getDashboardData();
    if (data) {
        res.json({ 
            success: true, 
            data: {
                dataStats: data.data_stats,
                featureStats: data.feature_stats
            }
        });
    } else {
        res.status(404).json({ success: false, message: 'Data not found' });
    }
});

// Get model results
app.get('/api/models', (req, res) => {
    const data = getDashboardData();
    if (data) {
        res.json({ 
            success: true, 
            data: {
                modelResults: data.model_results,
                modelRanking: data.model_ranking,
                bestModel: data.best_model
            }
        });
    } else {
        res.status(404).json({ success: false, message: 'Model data not found' });
    }
});

// Get ROC curves
app.get('/api/roc-curves', (req, res) => {
    const data = getDashboardData();
    if (data) {
        res.json({ 
            success: true, 
            data: data.roc_curves
        });
    } else {
        res.status(404).json({ success: false, message: 'ROC data not found' });
    }
});

// Get confusion matrices
app.get('/api/confusion-matrices', (req, res) => {
    const data = getDashboardData();
    if (data) {
        res.json({ 
            success: true, 
            data: data.confusion_matrices
        });
    } else {
        res.status(404).json({ success: false, message: 'Confusion matrix data not found' });
    }
});

// Get feature importance
app.get('/api/feature-importance', (req, res) => {
    const data = getDashboardData();
    if (data) {
        res.json({ 
            success: true, 
            data: data.feature_importance
        });
    } else {
        res.status(404).json({ success: false, message: 'Feature importance data not found' });
    }
});

// Get customer segments
app.get('/api/segments', (req, res) => {
    const data = getDashboardData();
    if (data) {
        res.json({ 
            success: true, 
            data: data.customer_segments
        });
    } else {
        res.status(404).json({ success: false, message: 'Segment data not found' });
    }
});

// Get churn factors
app.get('/api/churn-factors', (req, res) => {
    const data = getDashboardData();
    if (data) {
        res.json({ 
            success: true, 
            data: data.churn_factors
        });
    } else {
        res.status(404).json({ success: false, message: 'Churn factors data not found' });
    }
});

// Get age distribution
app.get('/api/age-distribution', (req, res) => {
    const data = getDashboardData();
    if (data) {
        res.json({ 
            success: true, 
            data: data.age_distribution
        });
    } else {
        res.status(404).json({ success: false, message: 'Age distribution data not found' });
    }
});

// Predict churn for a customer
app.post('/api/predict', async (req, res) => {
    const customerData = req.body;
    const modelName = req.query.model || 'random_forest';
    
    // Validate required fields
    const requiredFields = [
        'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls',
        'Payment Delay', 'Subscription Type', 'Contract Length', 'Total Spend', 'Last Interaction'
    ];
    
    for (const field of requiredFields) {
        if (customerData[field] === undefined) {
            return res.status(400).json({ 
                success: false, 
                message: `Missing required field: ${field}` 
            });
        }
    }
    
    try {
        // Create a Python script call for prediction
        const options = {
            mode: 'json',
            pythonPath: 'python',
            scriptPath: ML_MODEL_PATH,
            args: [JSON.stringify(customerData), modelName]
        };
        
        // Create inline prediction script
        const predictScript = `
import sys
import json
sys.path.insert(0, r'${ML_MODEL_PATH}')
from predict import predict_churn

customer_data = json.loads(sys.argv[1])
model_name = sys.argv[2].replace(' ', '_').lower()
result = predict_churn(customer_data, model_name)
print(json.dumps(result))
`;
        
        const { spawn } = require('child_process');
        const python = spawn('python', ['-c', predictScript, JSON.stringify(customerData), modelName]);
        
        let output = '';
        let errorOutput = '';
        
        python.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        python.stderr.on('data', (data) => {
            errorOutput += data.toString();
        });
        
        python.on('close', (code) => {
            if (code === 0) {
                try {
                    const result = JSON.parse(output.trim());
                    res.json({ success: true, data: result });
                } catch (e) {
                    res.status(500).json({ success: false, message: 'Failed to parse prediction result' });
                }
            } else {
                res.status(500).json({ success: false, message: errorOutput || 'Prediction failed' });
            }
        });
        
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
});

// Get list of available models
app.get('/api/available-models', (req, res) => {
    const data = getDashboardData();
    if (data) {
        const models = Object.keys(data.model_results).map(name => ({
            name,
            accuracy: data.model_results[name].accuracy,
            displayName: name
        }));
        res.json({ success: true, data: models });
    } else {
        res.status(404).json({ success: false, message: 'Model data not found' });
    }
});

// Serve static files from React build in production
if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.join(__dirname, '..', 'frontend', 'build')));
    
    app.get('*', (req, res) => {
        res.sendFile(path.join(__dirname, '..', 'frontend', 'build', 'index.html'));
    });
}

// Start server
const server = app.listen(PORT, () => {
    console.log(`
╔════════════════════════════════════════════════════════════╗
║     Customer Churn Prediction API Server                   ║
║     Running on http://localhost:${PORT}                        ║
╠════════════════════════════════════════════════════════════╣
║  Endpoints:                                                ║
║    GET  /api/health            - Health check              ║
║    GET  /api/dashboard         - All dashboard data        ║
║    GET  /api/stats             - Data statistics           ║
║    GET  /api/models            - Model results & ranking   ║
║    GET  /api/roc-curves        - ROC curve data            ║
║    GET  /api/confusion-matrices - Confusion matrices       ║
║    GET  /api/feature-importance - Feature importance       ║
║    GET  /api/segments          - Customer segments         ║
║    GET  /api/churn-factors     - Churn factors analysis    ║
║    POST /api/predict           - Predict customer churn    ║
╚════════════════════════════════════════════════════════════╝
    `);
});

// Error handling
server.on('error', (err) => {
    console.error('Server error:', err);
    process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
    process.exit(1);
});
