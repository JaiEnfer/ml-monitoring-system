![CI](https://github.com/JaiEnfer/ml-monitoring-system/actions/workflows/ci.yml/badge.svg)
![Docker](https://img.shields.io/badge/Docker-Build%20Ready-blue)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)
![MLOps](https://img.shields.io/badge/MLOps-Drift%20Monitoring-orange)
![GHCR](https://img.shields.io/badge/GHCR-Docker%20Image-success)
![Lifecycle](https://img.shields.io/badge/ML%20Lifecycle-Train%20→%20Monitor%20→%20Retrain-purple)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-brightgreen)




# ML Monitoring System (MLOps Core)

A **production-style Machine Learning system** that demonstrates the **full post-deployment lifecycle** of an ML model:  
**serving, logging, monitoring, data drift detection, alerting, retraining, CI/CD, and containerization**.


---

## 🚀 Key Features

- **FastAPI-based ML inference service**
- **Versioned model artifacts**
- **Persistent prediction logging (SQLite)**
- **Data drift detection using Evidently**
- **Human-readable drift reports (HTML)**
- **Machine-readable drift status & alerts (JSON)**
- **Traffic simulation for reproducible drift demos**
- **Safe model retraining with hot reload**
- **Automated testing (pytest)**
- **Static code quality checks (Ruff)**
- **CI with GitHub Actions**
- **Dockerized & published to GitHub Container Registry (GHCR)**

---

## 🧠 Why This Project

Most ML projects stop at **“model deployed”**.

In real production systems, the hard problems start **after deployment**:
- data distributions change
- models silently degrade
- retraining must be controlled and safe
- systems must be observable and reproducible

This project focuses on **operational ML**, not notebooks.

---

## 🏗️ System Architecture (High Level)

1. Client sends inference request
2. FastAPI validates input schema
3. Model generates prediction
4. Inputs + outputs are logged to database
5. Drift detection compares recent data vs training baseline
6. Drift status & alerts are exposed via API
7. Retraining can be triggered safely at runtime
8. CI/CD ensures reliability and reproducibility

---

## 📂 Project Structure

```text
ml-monitoring-system/
├── src/
│   ├── app.py              # FastAPI application
│   ├── train.py            # Model training & baseline generation
│   ├── model.py            # Model loading, inference, reload
│   ├── db.py               # SQLite logging
│   ├── drift.py            # Drift detection & metrics
│   ├── retrain.py          # Safe retraining logic
│   ├── simulate_traffic.py # Drift simulation
│   └── schemas.py          # API schemas
├── tests/                  # Automated tests
├── artifacts/              # Model, reference data, DB
├── reports/                # Drift HTML reports
├── .github/workflows/      # CI/CD pipelines
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🔍 API Endpoints

### Health & Inference

- **GET /health** : Returns service status and active model version
- **POST /predict** :  Runs inference and logs request + prediction

### Drift Monitoring

- **GET /drift/status** : Returns JSON drift summary (for dashboards/alerts)
- **GET /drift/report** : Generates a downloadable HTML drift report
- **GET /drift/alert?threshold=0.5** : Returns OK or ALERT based on drift severity

### Model Lifecycle

- **POST /retrain** : Safely retrains the model and hot-reloads it at runtime

---

## 🧪 Drift Demo (Reproducible)

This project includes a traffic simulator to demonstrate drift behavior.

#### 1️⃣ Start the API
```sh
uvicorn src.app:app --reload
```

#### 2️⃣ Send baseline traffic (no drift)
```sh
python -m src.simulate_traffic --mode baseline --n 200
```

Check:

```http
GET /drift/status
```

Expected:
```json
{
  "drift_detected": false,
  "share_drifted_features": 0.0
}

```

#### 3️⃣ Send shifted traffic (introduce drift)

```sh
python -m src.simulate_traffic --mode shifted --n 200
```

Check again:

```http
GET /drift/status
```

Expected:

```json
{
  "drift_detected": true,
  "share_drifted_features": 1.0
}
```

---


## 🔁 Retraining Workflow

1. Drift is detected via /drift/status
2. Alert level is exposed via /drift/alert
3. Retraining is triggered manually via /retrain
4. New model version is trained and saved
5. API reloads the new model without restart
6. Predictions immediately use the updated model

Model versions increment automatically (v1, v2, v3, …).

---

## 🐳 Docker

Build locally

```sh
docker build -t ml-monitoring-system .
```

Run

```sh
docker run --rm -p 8000:8000 ml-monitoring-system
```

## 🔄 CI/CD

**Continuous Integration**

On every push / pull request:

- Ruff linting
- Pytest test suite
- Docker build validation

**Continuous Delivery**

On version tags (v*):

- Docker image is built and published to GitHub Container Registry (GHCR)

---

## 🧠 Engineering Principles Demonstrated

- Training–serving consistency
- Observability over silent failure
- Versioned artifacts
- Safe concurrency (retraining lock)
- Reproducibility
- Infrastructure-as-code mindset
- Production-first ML design

---

## 🛠️ Tech Stack

1. Python 3.11
2. FastAPI
3. scikit-learn
4. Evidently
5. SQLite
6. Docker
7. GitHub Actions
8. Ruff
9. Pytest

---

## 🚧 Future Improvements

- Automatic retraining triggered by alerts
- Model registry (MLflow)
- Metrics export (Prometheus)
- Cloud deployment (AWS / GCP)
- Authentication for retrain endpoints

---

___Thank You___
