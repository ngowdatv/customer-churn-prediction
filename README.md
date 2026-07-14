# Customer Churn Prediction API | Machine Learning + FastAPI + MLflow

##  Overview

Customer Churn Prediction is an end-to-end Machine Learning application designed to identify customers who are likely to discontinue a service.

The project implements a complete ML lifecycle, starting from data preprocessing and model training to experiment tracking, model serialization, and real-time prediction through a FastAPI-based REST API.

The system enables businesses to proactively identify high-risk customers and take retention actions using data-driven insights.

---

## Project Objectives

- Build a predictive model to identify potential customer churn.
- Develop a complete Machine Learning pipeline.
- Track experiments and model performance using MLflow.
- Deploy the trained model as a scalable REST API.
- Provide interactive API testing using Swagger UI.

---

# System Architecture
             Customer Data
                  |
                  ↓
        Data Preprocessing
                  |
                  ↓
      Feature Engineering Pipeline
                  |
                  ↓
        Machine Learning Model
                  |
                  ↓
    Model Evaluation & Optimization
                  |
                  ↓
          MLflow Tracking
                  |
                  ↓
      Serialized Model Artifact
    (model.pkl + columns.pkl)
                  |
                  ↓
          FastAPI REST API
                  |
                  ↓
          Swagger UI Testing
                  |
                  ↓
          Production Deployment

          
---

#  Key Features

### 🔹 Machine Learning Pipeline
- Automated data preprocessing
- Feature transformation and encoding
- Model training and evaluation
- Performance metric tracking

### 🔹 Model Management
- Trained model stored using Joblib
- Feature column management
- MLflow experiment tracking
- Reproducible model experiments

### 🔹 API Development
- FastAPI-based REST service
- Pydantic data validation
- Real-time customer churn prediction
- Interactive Swagger documentation

### 🔹 Deployment Ready
- Production-oriented project structure
- Cloud deployment support
- Scalable API architecture

---

#  Technology Stack

| Category | Technologies |
|----------|-------------|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| API Framework | FastAPI |
| Server | Uvicorn |
| Experiment Tracking | MLflow |
| Model Serialization | Joblib |
| Deployment | Render |
| Documentation | Swagger UI |

---

# Machine Learning Workflow
Data Collection
|
↓
Data Cleaning
|
↓
Exploratory Data Analysis
|
↓
Feature Engineering
|
↓
Model Training
|
↓
Performance Evaluation
|
↓
Model Serialization
|
↓
API Deployment


---

#  Model Evaluation

The trained model is evaluated using industry-standard metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

These metrics help measure the model's ability to correctly identify customers who are likely to churn.