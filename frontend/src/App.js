import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  LineChart, Line, PieChart, Pie, Cell, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar, AreaChart, Area
} from 'recharts';

// Icons
const Icons = {
  dashboard: '📊',
  models: '🤖',
  predict: '🔮',
  insights: '💡',
  users: '👥',
  chart: '📈',
  warning: '⚠️',
  check: '✅',
  close: '❌',
  trophy: '🏆',
  target: '🎯',
  brain: '🧠',
  lightning: '⚡',
  fire: '🔥'
};

// Color palette
const COLORS = {
  primary: '#667eea',
  secondary: '#764ba2',
  success: '#48bb78',
  warning: '#ed8936',
  danger: '#f56565',
  info: '#4299e1',
  chart: ['#667eea', '#764ba2', '#48bb78', '#ed8936', '#f56565', '#4299e1', '#9f7aea', '#38b2ac', '#fc8181', '#f6ad55']
};

const API_BASE = 'http://localhost:5000/api';

function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [selectedModel, setSelectedModel] = useState('random_forest');
  
  // Prediction form state
  const [formData, setFormData] = useState({
    Age: 35,
    Gender: 'Male',
    Tenure: 12,
    'Usage Frequency': 15,
    'Support Calls': 3,
    'Payment Delay': 10,
    'Subscription Type': 'Standard',
    'Contract Length': 'Monthly',
    'Total Spend': 500,
    'Last Interaction': 7
  });

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_BASE}/dashboard`);
      if (response.data.success) {
        setDashboardData(response.data.data);
      } else {
        setError(response.data.message);
      }
    } catch (err) {
      setError('Failed to load dashboard data. Make sure the ML training has been completed and the backend server is running.');
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    setPredicting(true);
    setPredictionResult(null);
    try {
      const response = await axios.post(`${API_BASE}/predict?model=${selectedModel}`, formData);
      if (response.data.success) {
        setPredictionResult(response.data.data);
      }
    } catch (err) {
      console.error('Prediction failed:', err);
    } finally {
      setPredicting(false);
    }
  };

  const handleFormChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction'].includes(field)
        ? parseInt(value) || 0
        : value
    }));
  };

  if (loading) {
    return (
      <div className="app">
        <div className="loading">
          <div className="loading-spinner"></div>
          <p className="loading-text">Loading dashboard data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="app">
        <div className="header">
          <h1>{Icons.dashboard} Customer Churn Prediction</h1>
          <p>Machine Learning Analytics Dashboard</p>
        </div>
        <div className="card">
          <div className="error">
            <div className="error-icon">{Icons.warning}</div>
            <h2 className="error-title">Data Not Available</h2>
            <p className="error-message">{error}</p>
            <button className="retry-button" onClick={fetchDashboardData}>
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  const { data_stats, model_ranking, best_model, feature_importance, customer_segments, churn_factors, roc_curves, confusion_matrices, age_distribution } = dashboardData;

  // Prepare chart data
  const modelComparisonData = model_ranking.map(m => ({
    name: m.name.length > 12 ? m.name.substring(0, 12) + '...' : m.name,
    fullName: m.name,
    Accuracy: m.accuracy,
    Precision: m.precision,
    Recall: m.recall,
    'F1 Score': m.f1_score,
    'ROC AUC': m.roc_auc
  }));

  const churnDistributionData = [
    { name: 'Churned', value: data_stats.churn_rate_train, color: COLORS.danger },
    { name: 'Retained', value: 100 - data_stats.churn_rate_train, color: COLORS.success }
  ];

  const subscriptionChurnData = Object.entries(customer_segments.by_subscription).map(([name, data]) => ({
    name,
    'Churn Rate': data.churn_rate,
    'Avg Spend': data.avg_spend,
    Count: data.count
  }));

  const contractChurnData = Object.entries(customer_segments.by_contract).map(([name, data]) => ({
    name,
    'Churn Rate': data.churn_rate,
    'Avg Tenure': data.avg_tenure,
    Count: data.count
  }));

  const radarData = model_ranking.slice(0, 5).map(m => ({
    model: m.name,
    Accuracy: m.accuracy,
    Precision: m.precision,
    Recall: m.recall,
    F1: m.f1_score,
    AUC: m.roc_auc
  }));

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <h1>{Icons.dashboard} Customer Churn Prediction</h1>
        <p>Machine Learning Analytics Dashboard - Powered by {model_ranking.length} ML Models</p>
        <div className="header-stats">
          <div className="header-stat">
            <div className="header-stat-value">{data_stats.total_rows.toLocaleString()}</div>
            <div className="header-stat-label">Total Customers</div>
          </div>
          <div className="header-stat">
            <div className="header-stat-value">{data_stats.churn_rate_train}%</div>
            <div className="header-stat-label">Churn Rate</div>
          </div>
          <div className="header-stat">
            <div className="header-stat-value">{best_model.accuracy}%</div>
            <div className="header-stat-label">Best Accuracy</div>
          </div>
          <div className="header-stat">
            <div className="header-stat-value">{best_model.name}</div>
            <div className="header-stat-label">Best Model</div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="nav-tabs">
        <button className={`nav-tab ${activeTab === 'overview' ? 'active' : ''}`} onClick={() => setActiveTab('overview')}>
          {Icons.chart} Overview
        </button>
        <button className={`nav-tab ${activeTab === 'models' ? 'active' : ''}`} onClick={() => setActiveTab('models')}>
          {Icons.brain} Model Comparison
        </button>
        <button className={`nav-tab ${activeTab === 'insights' ? 'active' : ''}`} onClick={() => setActiveTab('insights')}>
          {Icons.insights} Insights
        </button>
        <button className={`nav-tab ${activeTab === 'predict' ? 'active' : ''}`} onClick={() => setActiveTab('predict')}>
          {Icons.predict} Predict Churn
        </button>
      </nav>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <>
          {/* Stat Cards */}
          <div className="stat-cards">
            <div className="stat-card purple">
              <div className="stat-card-icon">{Icons.users}</div>
              <div className="stat-card-value">{data_stats.training_rows.toLocaleString()}</div>
              <div className="stat-card-label">Training Samples</div>
            </div>
            <div className="stat-card green">
              <div className="stat-card-icon">{Icons.target}</div>
              <div className="stat-card-value">{data_stats.testing_rows.toLocaleString()}</div>
              <div className="stat-card-label">Testing Samples</div>
            </div>
            <div className="stat-card blue">
              <div className="stat-card-icon">{Icons.brain}</div>
              <div className="stat-card-value">{model_ranking.length}</div>
              <div className="stat-card-label">ML Models Trained</div>
            </div>
            <div className="stat-card orange">
              <div className="stat-card-icon">{Icons.lightning}</div>
              <div className="stat-card-value">{data_stats.features}</div>
              <div className="stat-card-label">Features Used</div>
            </div>
          </div>

          <div className="dashboard-grid two-col">
            {/* Model Performance Overview */}
            <div className="card">
              <div className="card-header">
                <div className="card-title">
                  <span className="card-title-icon purple">{Icons.trophy}</span>
                  Model Accuracy Comparison
                </div>
              </div>
              <div className="chart-container large">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={modelComparisonData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis type="number" domain={[0, 100]} />
                    <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 12 }} />
                    <Tooltip 
                      contentStyle={{ borderRadius: '10px', border: 'none', boxShadow: '0 4px 15px rgba(0,0,0,0.1)' }}
                      formatter={(value) => [`${value}%`, 'Accuracy']}
                    />
                    <Bar dataKey="Accuracy" fill="url(#colorGradient)" radius={[0, 5, 5, 0]} />
                    <defs>
                      <linearGradient id="colorGradient" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stopColor={COLORS.primary} />
                        <stop offset="100%" stopColor={COLORS.secondary} />
                      </linearGradient>
                    </defs>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Churn Distribution */}
            <div className="card">
              <div className="card-header">
                <div className="card-title">
                  <span className="card-title-icon orange">{Icons.chart}</span>
                  Churn Distribution
                </div>
              </div>
              <div className="chart-container large">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={churnDistributionData}
                      cx="50%"
                      cy="50%"
                      innerRadius={80}
                      outerRadius={120}
                      paddingAngle={5}
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
                    >
                      {churnDistributionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => `${value.toFixed(1)}%`} />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="dashboard-grid two-col">
            {/* Feature Importance */}
            <div className="card">
              <div className="card-header">
                <div className="card-title">
                  <span className="card-title-icon green">{Icons.fire}</span>
                  Feature Importance (Top 10)
                </div>
              </div>
              <div style={{ padding: '10px 0' }}>
                {feature_importance.slice(0, 10).map((feat, index) => (
                  <div key={feat.feature} className="feature-bar">
                    <span className="feature-name">{feat.feature}</span>
                    <div className="feature-bar-container">
                      <div 
                        className="feature-bar-fill" 
                        style={{ width: `${(feat.importance / feature_importance[0].importance) * 100}%` }}
                      ></div>
                    </div>
                    <span className="feature-value">{feat.importance.toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Age Distribution */}
            <div className="card">
              <div className="card-header">
                <div className="card-title">
                  <span className="card-title-icon blue">{Icons.users}</span>
                  Churn by Age Group
                </div>
              </div>
              <div className="chart-container large">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={age_distribution}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="Age_Group" />
                    <YAxis />
                    <Tooltip contentStyle={{ borderRadius: '10px', border: 'none', boxShadow: '0 4px 15px rgba(0,0,0,0.1)' }} />
                    <Area type="monotone" dataKey="churn_rate" name="Churn Rate %" stroke={COLORS.primary} fill="url(#areaGradient)" />
                    <defs>
                      <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={COLORS.primary} stopOpacity={0.8} />
                        <stop offset="100%" stopColor={COLORS.primary} stopOpacity={0.1} />
                      </linearGradient>
                    </defs>
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Models Tab */}
      {activeTab === 'models' && (
        <>
          {/* Model Ranking Table */}
          <div className="card" style={{ marginBottom: '25px' }}>
            <div className="card-header">
              <div className="card-title">
                <span className="card-title-icon purple">{Icons.trophy}</span>
                Model Performance Ranking
              </div>
            </div>
            <table className="ranking-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Model</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1 Score</th>
                  <th>ROC AUC</th>
                </tr>
              </thead>
              <tbody>
                {model_ranking.map((model, index) => (
                  <tr key={model.name}>
                    <td>
                      <div className={`rank ${index === 0 ? 'gold' : index === 1 ? 'silver' : index === 2 ? 'bronze' : 'normal'}`}>
                        {model.rank}
                      </div>
                    </td>
                    <td><strong>{model.name}</strong></td>
                    <td><span className={`metric-badge ${model.accuracy >= 85 ? 'high' : model.accuracy >= 75 ? 'medium' : 'low'}`}>{model.accuracy}%</span></td>
                    <td>{model.precision}%</td>
                    <td>{model.recall}%</td>
                    <td>{model.f1_score}%</td>
                    <td>{model.roc_auc}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="dashboard-grid two-col">
            {/* Multi-Metric Comparison */}
            <div className="card">
              <div className="card-header">
                <div className="card-title">
                  <span className="card-title-icon blue">{Icons.chart}</span>
                  Multi-Metric Comparison
                </div>
              </div>
              <div className="chart-container large">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={modelComparisonData.slice(0, 6)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 100]} />
                    <Tooltip contentStyle={{ borderRadius: '10px', border: 'none', boxShadow: '0 4px 15px rgba(0,0,0,0.1)' }} />
                    <Legend />
                    <Bar dataKey="Accuracy" fill={COLORS.chart[0]} radius={[3, 3, 0, 0]} />
                    <Bar dataKey="Precision" fill={COLORS.chart[1]} radius={[3, 3, 0, 0]} />
                    <Bar dataKey="Recall" fill={COLORS.chart[2]} radius={[3, 3, 0, 0]} />
                    <Bar dataKey="F1 Score" fill={COLORS.chart[3]} radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* ROC Curves */}
            <div className="card">
              <div className="card-header">
                <div className="card-title">
                  <span className="card-title-icon green">{Icons.target}</span>
                  ROC Curves (Top 5 Models)
                </div>
              </div>
              <div className="chart-container large">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="fpr" type="number" domain={[0, 1]} label={{ value: 'False Positive Rate', position: 'bottom' }} />
                    <YAxis domain={[0, 1]} label={{ value: 'True Positive Rate', angle: -90, position: 'left' }} />
                    <Tooltip contentStyle={{ borderRadius: '10px', border: 'none', boxShadow: '0 4px 15px rgba(0,0,0,0.1)' }} />
                    <Legend />
                    {Object.entries(roc_curves).slice(0, 5).map(([name, data], index) => (
                      <Line
                        key={name}
                        data={data.fpr.map((fpr, i) => ({ fpr, tpr: data.tpr[i] }))}
                        type="monotone"
                        dataKey="tpr"
                        name={`${name} (AUC: ${data.auc}%)`}
                        stroke={COLORS.chart[index]}
                        strokeWidth={2}
                        dot={false}
                      />
                    ))}
                    <Line
                      data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]}
                      type="monotone"
                      dataKey="tpr"
                      name="Random Baseline"
                      stroke="#a0aec0"
                      strokeDasharray="5 5"
                      strokeWidth={1}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="dashboard-grid three-col">
            {/* Confusion Matrices */}
            {Object.entries(confusion_matrices).slice(0, 3).map(([name, cm]) => (
              <div key={name} className="card">
                <div className="card-header">
                  <div className="card-title">{name}</div>
                </div>
                <div className="confusion-matrix">
                  <div className="cm-cell tn">
                    <span className="cm-cell-value">{cm.tn.toLocaleString()}</span>
                    <span className="cm-cell-label">True Negative</span>
                  </div>
                  <div className="cm-cell fp">
                    <span className="cm-cell-value">{cm.fp.toLocaleString()}</span>
                    <span className="cm-cell-label">False Positive</span>
                  </div>
                  <div className="cm-cell fn">
                    <span className="cm-cell-value">{cm.fn.toLocaleString()}</span>
                    <span className="cm-cell-label">False Negative</span>
                  </div>
                  <div className="cm-cell tp">
                    <span className="cm-cell-value">{cm.tp.toLocaleString()}</span>
                    <span className="cm-cell-label">True Positive</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* Insights Tab */}
      {activeTab === 'insights' && (
        <>
          <div className="dashboard-grid two-col">
            {/* Churn by Subscription Type */}
            <div className="card">
              <div className="card-header">
                <div className="card-title">
                  <span className="card-title-icon purple">{Icons.chart}</span>
                  Churn by Subscription Type
                </div>
              </div>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={subscriptionChurnData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip contentStyle={{ borderRadius: '10px', border: 'none', boxShadow: '0 4px 15px rgba(0,0,0,0.1)' }} />
                    <Legend />
                    <Bar dataKey="Churn Rate" fill={COLORS.danger} radius={[5, 5, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="segment-grid" style={{ marginTop: '20px' }}>
                {subscriptionChurnData.map(seg => (
                  <div key={seg.name} className="segment-card">
                    <div className="segment-title">{seg.name}</div>
                    <div className={`segment-churn ${seg['Churn Rate'] > 50 ? 'high' : seg['Churn Rate'] > 30 ? 'medium' : 'low'}`}>
                      {seg['Churn Rate']}%
                    </div>
                    <div className="segment-count">{seg.Count.toLocaleString()} customers</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Churn by Contract Length */}
            <div className="card">
              <div className="card-header">
                <div className="card-title">
                  <span className="card-title-icon green">{Icons.chart}</span>
                  Churn by Contract Length
                </div>
              </div>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={contractChurnData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip contentStyle={{ borderRadius: '10px', border: 'none', boxShadow: '0 4px 15px rgba(0,0,0,0.1)' }} />
                    <Legend />
                    <Bar dataKey="Churn Rate" fill={COLORS.warning} radius={[5, 5, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="segment-grid" style={{ marginTop: '20px' }}>
                {contractChurnData.map(seg => (
                  <div key={seg.name} className="segment-card">
                    <div className="segment-title">{seg.name}</div>
                    <div className={`segment-churn ${seg['Churn Rate'] > 50 ? 'high' : seg['Churn Rate'] > 30 ? 'medium' : 'low'}`}>
                      {seg['Churn Rate']}%
                    </div>
                    <div className="segment-count">{seg.Count.toLocaleString()} customers</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Churn Risk Factors */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <span className="card-title-icon orange">{Icons.warning}</span>
                Churn Risk Factors Analysis
              </div>
            </div>
            <table className="ranking-table">
              <thead>
                <tr>
                  <th>Factor</th>
                  <th>Churned Avg</th>
                  <th>Retained Avg</th>
                  <th>Difference</th>
                  <th>Impact</th>
                </tr>
              </thead>
              <tbody>
                {churn_factors.map(factor => (
                  <tr key={factor.feature}>
                    <td><strong>{factor.feature}</strong></td>
                    <td>{factor.churned_avg}</td>
                    <td>{factor.retained_avg}</td>
                    <td>
                      <span style={{ color: factor.difference_pct > 0 ? COLORS.danger : COLORS.success }}>
                        {factor.difference_pct > 0 ? '+' : ''}{factor.difference_pct}%
                      </span>
                    </td>
                    <td>
                      <span className={`metric-badge ${Math.abs(factor.difference_pct) > 20 ? 'high' : Math.abs(factor.difference_pct) > 10 ? 'medium' : 'low'}`}>
                        {Math.abs(factor.difference_pct) > 20 ? 'High' : Math.abs(factor.difference_pct) > 10 ? 'Medium' : 'Low'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Gender Analysis */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <span className="card-title-icon blue">{Icons.users}</span>
                Churn by Gender
              </div>
            </div>
            <div className="segment-grid" style={{ maxWidth: '600px' }}>
              {Object.entries(customer_segments.by_gender).map(([gender, data]) => (
                <div key={gender} className="segment-card">
                  <div className="segment-title">{gender}</div>
                  <div className={`segment-churn ${data.churn_rate > 50 ? 'high' : data.churn_rate > 30 ? 'medium' : 'low'}`}>
                    {data.churn_rate}%
                  </div>
                  <div className="segment-count">{data.count.toLocaleString()} customers</div>
                  <div className="segment-count">Avg Spend: ${data.avg_spend}</div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {/* Predict Tab */}
      {activeTab === 'predict' && (
        <div className="dashboard-grid two-col">
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <span className="card-title-icon purple">{Icons.predict}</span>
                Predict Customer Churn
              </div>
            </div>
            
            <div className="form-group" style={{ marginBottom: '20px' }}>
              <label>Select Model</label>
              <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                {model_ranking.map(m => (
                  <option key={m.name} value={m.name.replace(' ', '_').toLowerCase()}>
                    {m.name} (Accuracy: {m.accuracy}%)
                  </option>
                ))}
              </select>
            </div>

            <div className="prediction-form">
              <div className="form-group">
                <label>Age</label>
                <input type="number" value={formData.Age} onChange={(e) => handleFormChange('Age', e.target.value)} min="18" max="100" />
              </div>
              <div className="form-group">
                <label>Gender</label>
                <select value={formData.Gender} onChange={(e) => handleFormChange('Gender', e.target.value)}>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>
              <div className="form-group">
                <label>Tenure (months)</label>
                <input type="number" value={formData.Tenure} onChange={(e) => handleFormChange('Tenure', e.target.value)} min="0" max="72" />
              </div>
              <div className="form-group">
                <label>Usage Frequency</label>
                <input type="number" value={formData['Usage Frequency']} onChange={(e) => handleFormChange('Usage Frequency', e.target.value)} min="0" max="30" />
              </div>
              <div className="form-group">
                <label>Support Calls</label>
                <input type="number" value={formData['Support Calls']} onChange={(e) => handleFormChange('Support Calls', e.target.value)} min="0" max="10" />
              </div>
              <div className="form-group">
                <label>Payment Delay (days)</label>
                <input type="number" value={formData['Payment Delay']} onChange={(e) => handleFormChange('Payment Delay', e.target.value)} min="0" max="30" />
              </div>
              <div className="form-group">
                <label>Subscription Type</label>
                <select value={formData['Subscription Type']} onChange={(e) => handleFormChange('Subscription Type', e.target.value)}>
                  <option value="Basic">Basic</option>
                  <option value="Standard">Standard</option>
                  <option value="Premium">Premium</option>
                </select>
              </div>
              <div className="form-group">
                <label>Contract Length</label>
                <select value={formData['Contract Length']} onChange={(e) => handleFormChange('Contract Length', e.target.value)}>
                  <option value="Monthly">Monthly</option>
                  <option value="Quarterly">Quarterly</option>
                  <option value="Annual">Annual</option>
                </select>
              </div>
              <div className="form-group">
                <label>Total Spend ($)</label>
                <input type="number" value={formData['Total Spend']} onChange={(e) => handleFormChange('Total Spend', e.target.value)} min="0" max="2000" />
              </div>
              <div className="form-group">
                <label>Last Interaction (days)</label>
                <input type="number" value={formData['Last Interaction']} onChange={(e) => handleFormChange('Last Interaction', e.target.value)} min="0" max="30" />
              </div>
              <button className="predict-button" onClick={handlePredict} disabled={predicting}>
                {predicting ? 'Predicting...' : `${Icons.predict} Predict Churn`}
              </button>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <span className="card-title-icon green">{Icons.target}</span>
                Prediction Result
              </div>
            </div>
            
            {predictionResult ? (
              <div className={`prediction-result ${predictionResult.prediction === 1 ? 'churn' : 'no-churn'}`}>
                <div className="prediction-icon">
                  {predictionResult.prediction === 1 ? Icons.warning : Icons.check}
                </div>
                <div className="prediction-text">
                  {predictionResult.prediction === 1 ? 'Customer Likely to Churn' : 'Customer Likely to Stay'}
                </div>
                <div className="prediction-probability">
                  Churn Probability: <strong>{predictionResult.churn_probability}%</strong>
                </div>
                <div className="prediction-probability">
                  Retention Probability: <strong>{predictionResult.retention_probability}%</strong>
                </div>
                <div className={`risk-level ${predictionResult.risk_level.toLowerCase()}`}>
                  {predictionResult.risk_level} Risk
                </div>
                <p style={{ marginTop: '20px', fontSize: '0.9rem', color: '#718096' }}>
                  Model used: {predictionResult.model_used}
                </p>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '60px 20px', color: '#718096' }}>
                <div style={{ fontSize: '4rem', marginBottom: '20px' }}>{Icons.predict}</div>
                <p>Fill in customer details and click "Predict Churn" to see the prediction result.</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="footer">
        <p>Customer Churn Prediction Dashboard | Built with React, Node.js & Machine Learning</p>
        <p>Data: {data_stats.total_rows.toLocaleString()} customers | Models: {model_ranking.length} | Generated: {new Date(dashboardData.generated_at).toLocaleString()}</p>
      </footer>
    </div>
  );
}

export default App;
