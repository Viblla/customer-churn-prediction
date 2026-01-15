# Customer Churn Prediction Dashboard

A comprehensive Machine Learning project that predicts customer churn using multiple ML algorithms, with a beautiful Node.js/React dashboard for visualization and real-time predictions.

## 🚀 Features

### Machine Learning
- **9+ ML Algorithms**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM, KNN, Naive Bayes, Neural Network
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Feature Importance Analysis**
- **Customer Segmentation Insights**

### Dashboard
- **Interactive Visualizations**: Charts, graphs, and tables
- **Model Comparison**: Side-by-side performance metrics
- **ROC Curves**: Visual model evaluation
- **Confusion Matrices**: Detailed prediction analysis
- **Live Predictions**: Predict churn for new customers
- **Customer Insights**: Segmentation by subscription, contract, age, gender
- **Responsive Design**: Works seamlessly across all devices

## 📁 Project Structure

```
Customer_Churn_Prediction/
├── Dataset/
│   ├── customer_churn_dataset-training-master.csv
│   └── customer_churn_dataset-testing-master.csv
├── ml_model/
│   ├── train_model.py          # ML training pipeline
│   ├── predict.py              # Prediction service
│   ├── requirements.txt        # Python dependencies
│   ├── models/                 # Saved ML models
│   └── results/                # Generated metrics JSON
├── backend/
│   ├── server.js               # Express API server
│   └── package.json
├── frontend/
│   ├── src/
│   │   ├── App.js              # Main React component
│   │   ├── index.js
│   │   └── index.css           # Styling
│   ├── public/
│   └── package.json
└── README.md
```

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Step 1: Install Python Dependencies

```bash
cd ml_model
pip install -r requirements.txt
```

### Step 2: Train ML Models

```bash
cd ml_model
python train_model.py
```

This will:
- Train 9+ ML models
- Generate performance metrics
- Save models to `ml_model/models/`
- Save dashboard data to `ml_model/results/dashboard_data.json`

### Step 3: Start Backend Server

```bash
cd backend
npm install
npm start
```

The API server will run on `http://localhost:5000`

### Step 4: Start Frontend Dashboard

```bash
cd frontend
npm install
npm start
```

The dashboard will open at `http://localhost:3000`

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/dashboard` | GET | All dashboard data |
| `/api/stats` | GET | Dataset statistics |
| `/api/models` | GET | Model results & ranking |
| `/api/roc-curves` | GET | ROC curve data |
| `/api/confusion-matrices` | GET | Confusion matrices |
| `/api/feature-importance` | GET | Feature importance |
| `/api/segments` | GET | Customer segments |
| `/api/churn-factors` | GET | Churn risk factors |
| `/api/predict` | POST | Predict customer churn |

## 📈 Dashboard Features

### Overview Tab
- Dataset statistics cards
- Model accuracy comparison chart
- Churn distribution pie chart
- Feature importance ranking
- Age group churn analysis

### Model Comparison Tab
- Full model ranking table
- Multi-metric bar charts
- ROC curves overlay
- Confusion matrices for top models

### Insights Tab
- Churn by subscription type
- Churn by contract length
- Churn risk factors analysis
- Gender-based segmentation

### Predict Churn Tab
- Interactive prediction form
- Model selection dropdown
- Real-time churn prediction
- Risk level classification

## 🛠️ Tech Stack

- **ML Pipeline**: Python, Scikit-learn, XGBoost, LightGBM
- **Backend**: Node.js, Express.js
- **Frontend**: React, Recharts
- **Styling**: Custom CSS

## 📝 Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| CustomerID | Numeric | Unique identifier |
| Age | Numeric | Customer age |
| Gender | Categorical | Male/Female |
| Tenure | Numeric | Months with company |
| Usage Frequency | Numeric | Service usage frequency |
| Support Calls | Numeric | Number of support calls |
| Payment Delay | Numeric | Payment delay in days |
| Subscription Type | Categorical | Basic/Standard/Premium |
| Contract Length | Categorical | Monthly/Quarterly/Annual |
| Total Spend | Numeric | Total amount spent |
| Last Interaction | Numeric | Days since last interaction |
| Churn | Target | 0 = Stayed, 1 = Churned |

## 🏆 Model Performance

Models are automatically ranked by accuracy. Typical results include:
- Best models usually achieve 85-90%+ accuracy
- Feature importance helps identify key churn predictors
- ROC-AUC scores indicate model discrimination ability

## 📄 License

This project is for educational purposes.

---

Built with ❤️ for GIKI Data Science Project
