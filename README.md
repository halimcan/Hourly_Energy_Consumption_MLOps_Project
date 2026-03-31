# 🚀 Production-Grade MLOps System with Business-Aware Monitoring

## 📌 Overview

This project is a full **end-to-end MLOps system** built using real-world time series data.

* Energy consumption dataset (hourly, multiple regions)
* **2014–2017 → training**
* **2018 → simulated production**

👉 The goal is not just to build a model, but to build a **system that behaves like a real production ML environment**.

---

## 📊 Dataset

* Source: PJM Hourly Energy Consumption Dataset
* Kaggle: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption

The dataset contains **hourly electricity consumption (MW)** across multiple U.S. regions.

PJM (Pennsylvania–New Jersey–Maryland Interconnection) operates a large part of the U.S. power grid, covering multiple states.

⚠️ Note:

* Regions vary over time
* Some regions have missing periods
* This makes it closer to **real-world imperfect data**

---

## 🎯 Key Idea

This system bridges:

**Model Performance → Business Impact → Decision Making**

Instead of optimizing only for RMSE:

* 💰 Cost-aware evaluation
* ⚠️ Under/over prediction penalties
* 🔁 Retrain decision signals
* 👤 Human-in-the-loop approval

---

## 🤖 Modeling Approach

Models implemented:

* Baseline model
* XGBoost
* LSTM

### 🏆 Model Selection

The winning model is **not chosen by RMSE alone**.

Selection criteria:

* 📉 RMSE
* 💰 Cost Weighted Error (CWE)
* ⚠️ Underprediction ratio

👉 The final model minimizes **business cost and risk**, not just error.

---

## 🧩 System Architecture

* **FastAPI** → Model serving
* **Prometheus + Grafana** → Monitoring
* **MLflow** → Experiment tracking
* **Prefect** → Orchestration
* **Evidently** → Drift detection
* **Streamlit** → Business dashboard
* **Live Simulator** → Production-like traffic

---

## 🔄 End-to-End Flow

```text
Data → Model → API → Metrics → Drift → Retrain Signal → Human Approval → Retrain
```

---

## 📊 Business-Aware Metrics

### 💰 Cost Weighted Error (CWE)

```
CWE = error × penalty_factor
```

* Underprediction → high penalty
* Overprediction → lower penalty

---

### ⚠️ Risk Metrics

* Underprediction ratio
* Peak cost
* P95 cost
* Rolling cost trends

👉 Focus: **risk & financial impact**

---

## 📈 Monitoring & Retraining

* Real-time metrics (latency, volume, RMSE)
* Drift detection with Evidently
* Retrain signals (**NOT automatic**)

### 🚨 Strategy

1. Degradation detected
2. Signal generated
3. Human reviews
4. Retraining approved

👉 Reflects real-world production constraints

---

## 🧠 Design Philosophy

* ❌ No auto-retraining
* 👤 Human-in-the-loop
* 💰 Business metrics over pure accuracy
* 🔄 Continuous simulation for realism

---

## 🛠 How to Run

### 🟢 Local

```bash
make up-local
```

Stop:

```bash
make down-local
```

---

### 🐳 Docker (Production-like Setup)

```bash
docker compose up --build
```

👉 Recommended when you want:

* Isolated environment
* Reproducible setup
* Production-like behavior

👉 Includes:

* All services containerized
* Networked services (API, monitoring, MLflow, etc.)
* Clean dependency management

---

### 📦 Version

Use branch:

```
step6-finalized
```

---

## 📂 Project Structure

```text
.
├── src/
│   ├── api/
│   ├── training/
│   ├── monitoring/
│   ├── models/
│   ├── metrics/
│   ├── registry/
│   └── validation/
├── ui/
├── data/
├── reports/
├── logs/
├── docker-compose.yml
├── run_all.sh
```

---

## 🧰 Tech Stack

* **Modeling**: Python, XGBoost, LSTM, Baseline
* **Serving**: FastAPI
* **Monitoring**: Prometheus, Grafana, Evidently
* **Tracking**: MLflow
* **Orchestration**: Prefect
* **UI**: Streamlit
* **Infrastructure**: Docker

---

## 🔮 Future Improvements

* 📦 Data validation (Great Expectations)
* 🚨 Alerts (Email / Slack)
* 🔁 CI/CD (GitHub Actions / Jenkins)
* 📊 Streamlit:

  * Model comparison
  * What-if simulation (sliders)
* 🌓 Shadow deployment
* ☁️ Cloud deployment










