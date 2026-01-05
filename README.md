# OHLCV Stock Analysis - Predictive Trading System

![OHLCV Stock Analysis](./imgs/ohlcv_analysis.png)

## Description of the Problem

### Background
Stock market prediction is one of the most challenging problems in financial analytics. Traders and investors need reliable tools to forecast price movements and identify profitable trading opportunities. Traditional technical analysis relies on subjective interpretation of charts and indicators, which can be inconsistent and error-prone.

### Problem Statement
Quantitative traders and financial institutions need automated, data-driven systems to:

- **Predict Price Direction**: Classify whether the stock price will move up or down in the next trading period
- **Forecast Price Targets**: Predict the exact closing price for informed position sizing and profit-taking
- **Reduce Human Bias**: Replace subjective analysis with objective machine learning predictions
- **Optimize Trading Strategy**: Enable backtesting and live trading with consistent, reproducible signals
- **Scale Analysis**: Process hundreds of stocks simultaneously with real-time predictions

### Solution Approach

This project implements a dual-model machine learning pipeline for stock price prediction:

1. **Price Direction Classification**: Binary classification using XGBoost to predict up/down movement
2. **Price Target Regression**: Regression model using XGBoost to forecast future closing prices
3. **Technical Feature Engineering**: Extract OHLCV-derived features including volatility, momentum, and trend indicators
4. **Production Deployment**: Flask API + Kubernetes for scalable, enterprise-grade deployment

The solution processes OHLCV (Open-High-Low-Close-Volume) data and engineered technical indicators to make probabilistic predictions on stock price movements within a configurable time horizon.

### Business Impact
- **For Traders**: Data-driven entry/exit signals with quantified confidence scores
- **For Hedge Funds**: Automated screening and position management across large portfolios
- **For FinTech Platforms**: White-label prediction API for retail and institutional clients
- **For Risk Management**: Probabilistic forecasts for scenario planning and hedging

## Instructions on How to Run the Project

### Prerequisites

#### System Requirements
- Python 3.10 or higher
- Docker (for containerization)
- Kubernetes & minikube (for orchestration)
- Jupyter Notebook
- 8GB RAM minimum (for XGBoost training)
- 5GB free disk space

### Local Development Setup

#### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv OHLCV_venv
source OHLCV_venv/bin/activate  # On Windows: OHLCV_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Data Setup
Training data should be placed in the `data/` folder. The project expects OHLCV format CSV files with columns: `open`, `high`, `low`, `close`, `volume`, and timestamp.

#### 3. Run the Complete Pipeline

**Step 1: Data Analysis and Model Training**
```bash
# Open Jupyter notebook for EDA and feature engineering
jupyter notebook notebook.ipynb
```

**Step 2: Train Models**
```bash
# Train and save both XGBoost models
python train.py
```

This generates:
- `models/model_xgb_class.bin` - Classification model (direction prediction)
- `models/model_xgb_reg.bin` - Regression model (price target prediction)

**Step 3: Start the Prediction Service Locally**

**Build the Docker image**

From the project root:
```bash
docker build -t ohlcv-predictor -f deployment/flask/Dockerfile .
```

**Run the Docker container**
```bash
docker run -it --rm -p 9696:9696 ohlcv-predictor
```

The Flask API will be available at: `http://localhost:9696`

**Test the service**

Navigate to `deployment/flask/` and run:
```bash
jupyter notebook predict_test.ipynb
```

**Step 4: Kubernetes Deployment**

Deploy to minikube from the project root:
```bash
./deployment/kubernetes/deploy.sh
```

This deploys the service with:
- Health checks and auto-restart
- Resource limits (CPU: 250m-500m, Memory: 512Mi-1Gi)
- Service exposure on port 9696

[For detailed Kubernetes setup, see `./deployment/kubernetes/README.md`]

### Project Structure
```
OHLCV-stock-analysis/
├── data/
│   └── README.md                                    # Data format documentation
├── deployment/
│   ├── flask/
│   │   ├── Dockerfile                              # Docker container config
│   │   ├── Pipfile                                 # Pipenv dependencies
│   │   ├── Pipfile.lock                            # Locked dependencies
│   │   ├── predict_test.ipynb                      # API testing notebook
│   │   ├── predict.py                              # Flask web service
│   │   └── README.md                               # Flask deployment docs
│   └── kubernetes/
│       ├── deploy.sh                               # Deployment automation script
│       ├── deployment.yaml                         # K8s manifests
│       └── README.md                               # K8s deployment guide
├── imgs/
│   └── README.md                                   # Image assets
├── models/
│   ├── model_xgb_class.bin                         # Classification model
│   ├── model_xgb_reg.bin                           # Regression model
│   └── README.md                                   # Model documentation
├── notebook.ipynb                                  # Complete analysis & training
├── train.py                                        # Model training pipeline
├── requirements.txt                                # Python dependencies
└── README.md                                       # This file
```

## API Usage

### Health Check
```bash
curl -X GET http://localhost:9696/health
```

### Price Prediction
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "open": 100.0,
    "high": 105.0,
    "low": 99.5,
    "close": 104.0,
    "volume": 1500.0,
    "return_1": 0.02,
    "range": 5.5,
    "body": 4.0,
    "volatility_5": 1.5,
    "volume_change": 0.05,
    "trend_slope_5": 0.8
  }'
```

**Response:**
```json
{
  "classification_label": 1,
  "classification_probability": 0.8234,
  "regression_prediction": 104.52
}
```

- `classification_label`: 1 = price up, 0 = price down
- `classification_probability`: Confidence score (0-1)
- `regression_prediction`: Forecasted closing price
