import pickle
import xgboost as xgb
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask('ohlcv_prediction')

# ======================
# LOAD MODELS
# ======================

dv_class = None
class_model = None
dv_reg = None
reg_model = None

def load_models():
    global dv_class, class_model, dv_reg, reg_model
    try:
        with open('models/model_xgb_class.bin', 'rb') as f:
            dv_class, class_model = pickle.load(f)
        with open('models/model_xgb_reg.bin', 'rb') as f:
            dv_reg, reg_model = pickle.load(f)
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

load_models()

# ======================
# REQUIRED FEATURES
# ======================

REQUIRED_FEATURES = [
    'open', 'high', 'low', 'close', 'volume',
    'return_1', 'range', 'body', 'volatility_5',
    'volume_change', 'trend_slope_5'
]

# ======================
# HEALTH CHECK
# ======================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'ohlcv-prediction',
        'timestamp': str(datetime.now())
    })

# ======================
# PREDICT ENDPOINT
# ======================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate input
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        missing_features = [f for f in REQUIRED_FEATURES if f not in data]
        if missing_features:
            return jsonify({
                'error': 'Missing required features',
                'missing_features': missing_features
            }), 400

        # Validate feature values
        try:
            X = np.array([[float(data[f]) for f in REQUIRED_FEATURES]])
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid feature value: {e}'}), 400

        # Check for NaN/inf values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            return jsonify({'error': 'Features contain NaN or infinity values'}), 400

        # Classification - pass numpy array directly, not DMatrix
        class_proba = float(class_model.predict_proba(X)[0][1])
        class_label = int(class_proba >= 0.5)

        result = {
            'classification_probability': round(class_proba, 4),
            'classification_label': class_label
        }

        # Regression (always)
        reg_pred = float(reg_model.predict(X)[0])
        result['regression_prediction'] = round(reg_pred, 2)

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# ======================
# RUN APP
# ======================

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=9696)