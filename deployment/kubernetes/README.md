# Kubernetes Local Deployment

![Minikube and Kubernetes](../../imgs/minikube_and_kubernetes.png)

This project includes a complete Kubernetes deployment setup for the OHLCV Stock Analysis Prediction Service using minikube for local development and testing.

# Record of Deployment on Kubernetes

![Kubernetes Deploy](../../imgs/KubernetesDeploy.gif)

## 📋 Prerequisites

- [Docker](https://www.docker.com/) installed and running
- [minikube](https://minikube.sigs.k8s.io/docs/start/) installed
- [kubectl](https://kubernetes.io/docs/tasks/tools/) installed

## 🚀 Deployment

### 1. Deploy to Local Kubernetes Cluster

```bash
# From the project root directory
./deployment/kubernetes/deploy.sh
```

The deployment script will:
- Start minikube
- Build the Docker image in minikube's environment
- Deploy the service to Kubernetes
- Wait for the pod to be ready
- Display service information
- Set up port forwarding to localhost:9696

### 2. Access the Service

```bash
# Get the service URL
minikube service ohlcv-prediction-service -n ohlcv-prediction --url

# Or open in browser directly
minikube service ohlcv-prediction-service -n ohlcv-prediction
```

### 3. View in Kubernetes Dashboard

```bash
# Open Kubernetes dashboard
minikube dashboard
```

**Important:** In the dashboard, select **"ohlcv-prediction"** from the namespace dropdown to view your deployment.

## 🧪 Testing the Service

### Health Check

```bash
curl -X GET http://localhost:9696/health
```

**Response:**
```json
{
  "service": "ohlcv-prediction",
  "status": "healthy",
  "timestamp": "2026-01-05 16:14:56.529579"
}
```

### Price Direction & Target Prediction

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

**Response Fields:**
- `classification_label`: 1 = price up, 0 = price down
- `classification_probability`: Confidence score (0-1)
- `regression_prediction`: Forecasted closing price

### Test via Jupyter Notebook

```bash
# Navigate to deployment directory
cd deployment/flask

# Open testing notebook
jupyter notebook predict_test.ipynb
```

## 📁 Deployment Structure

```
deployment/kubernetes/
├── deploy.sh              # Automated deployment script
├── deployment.yaml        # Kubernetes manifests (namespace, configmap, deployment, service)
└── README.md              # This file
```

## 🛠️ Useful Commands

```bash
# Check deployment status
kubectl get all -n ohlcv-prediction

# View pod logs
kubectl logs -l app=ohlcv-prediction -n ohlcv-prediction

# View detailed pod information
kubectl describe pod -l app=ohlcv-prediction -n ohlcv-prediction

# Delete deployment
kubectl delete -f deployment/kubernetes/deployment.yaml

# Stop minikube
minikube stop

# Start minikube again
minikube start

# Kill port forwarding
pkill -f "kubectl port-forward.*9696"
```

## 🔧 Service Configuration

- **Image:** `ohlcv-predictor:latest` (built locally in minikube)
- **Port:** Service runs on port 80, forwards to container port 9696
- **NodePort:** 30081 for external access
- **Namespace:** `ohlcv-prediction`
- **Health Checks:** Readiness and liveness probes on `/health`
- **Readiness Probe:** 15s initial delay, 10s interval, 5s timeout
- **Liveness Probe:** 30s initial delay, 30s interval, 10s timeout
- **Resources:**
  - CPU: 250m request, 500m limit
  - Memory: 512Mi request, 1Gi limit

## 📊 Architecture

The deployment creates:

1. **Namespace:** `ohlcv-prediction` for resource isolation and organization
2. **ConfigMap:** Environment variables (FLASK_ENV=production, MODEL_PATH=/app/models)
3. **Deployment:** Single replica of the Flask application with auto-restart
4. **Service:** NodePort service for external access to port 9696
5. **Pod:** Running container with embedded XGBoost models

### Kubernetes Resource Flow

```
┌─────────────────────────────────────┐
│   Kubernetes Cluster (minikube)     │
├─────────────────────────────────────┤
│  Namespace: ohlcv-prediction        │
│                                     │
│  ┌─ Deployment ─────────────────┐   │
│  │ ohlcv-prediction-deployment  │   │
│  │                              │   │
│  │ ┌─ Pod ──────────────────┐   │   │
│  │ │ ohlcv-prediction       │   │   │
│  │ │ - Flask App (9696)     │   │   │
│  │ │ - XGBoost Models       │   │   │
│  │ │ - Health Checks        │   │   │
│  │ └────────────────────────┘   │   │
│  └───────────────┬──────────────┘   │
│                  │                  │
│  ┌───────────────┴──────────────┐   │
│  │ Service: NodePort (30081)   │    │
│  │ Port Forwarding (9696)      │    │
│  └─────────────────────────────┘    │
└───────┬─────────────────────────────┘
        │
        └─→ localhost:9696
```

## The Flask Application

The Flask application serves two endpoints:

- `GET /health` - Health check endpoint for Kubernetes probes
- `POST /predict` - ML prediction endpoint (classification + regression)

### Model Loading

Models are embedded in the Docker image at build time:
- `models/model_xgb_class.bin` - Classification model (price direction)
- `models/model_xgb_reg.bin` - Regression model (price target)

Models load automatically when the pod starts, confirmed by:
```
INFO:predict:Models loaded successfully
```

## 🔄 Deployment Workflow

1. **Build Phase**
   ```bash
   docker build -t ohlcv-predictor:latest -f deployment/flask/Dockerfile .
   ```

2. **Deploy Phase**
   ```bash
   kubectl apply -f deployment/kubernetes/deployment.yaml
   ```

3. **Ready Phase**
   - Readiness probe checks `/health` every 10s
   - Pod ready when health check passes
   - Service starts forwarding traffic

4. **Port Forward Phase**
   ```bash
   kubectl port-forward service/ohlcv-prediction-service 9696:80
   ```

5. **Access Phase**
   - Service available at `http://localhost:9696`

## 🚨 Troubleshooting

### Service not accessible

```bash
# Check if pod is running
kubectl get pods -n ohlcv-prediction

# View pod logs
kubectl logs -l app=ohlcv-prediction -n ohlcv-prediction

# Check service status
kubectl get svc -n ohlcv-prediction

# Verify port forwarding
netstat -an | grep 9696
```

### Port already in use

```bash
# Kill existing port forward
pkill -f "kubectl port-forward.*9696"

# Or use a different local port
kubectl port-forward service/ohlcv-prediction-service 8080:80 -n ohlcv-prediction
```

### Pod crashes or restarts

```bash
# Check pod events
kubectl describe pod -l app=ohlcv-prediction -n ohlcv-prediction

# View complete logs
kubectl logs -l app=ohlcv-prediction -n ohlcv-prediction --all-containers=true
```

### Models not loading

Ensure model files exist in `models/` directory before building:
```bash
ls -la models/model_xgb_*.bin
```

## 📈 Monitoring

```bash
# Watch pod status
kubectl get pods -n ohlcv-prediction --watch

# View resource usage (requires metrics-server)
kubectl top pod -n ohlcv-prediction

# Monitor deployment
kubectl rollout status deployment/ohlcv-prediction-deployment -n ohlcv-prediction
```

## 🧹 Cleanup

```bash
# Delete entire namespace and all resources
kubectl delete namespace ohlcv-prediction

# Or delete specific resources
kubectl delete -f deployment/kubernetes/deployment.yaml

# Stop minikube
minikube stop

# Delete minikube cluster (irreversible)
minikube delete
```

---

This provides a complete local Kubernetes environment for developing and testing the OHLCV stock analysis prediction service.