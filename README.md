# 🌍 AI-Powered Early Warning System for Global Vaccination Coverage

### Developed by: Health Data Scientist — Morocco 🇲🇦
[![Streamlit App](https://static.streamlit.io/badge-svg.svg)](https://streamlit.io/)
[![Python 3.9+](https://img.shields.shields.shields.shields.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## 📌 Project Overview
This project is an advanced, machine learning-based **Early Warning System** engineered to forecast the risk of DTP3 (Diphtheria, Tetanus, and Pertussis) vaccination dropouts across global populations by the **2030 horizon**. 

By training a multi-layer Neural Network on historical global health datasets from **UNICEF** and incorporating **Explainable AI (XAI)** through SHAP, this system acts as a predictive radar. It empowers international health organizations to transition from reactive crisis management to proactive resource allocation, directly supporting the **UN 2030 Sustainable Development Goals (SDGs)** for global health and well-being.

---

## 🎯 The Core Problem & Value Proposition
* **The Proxy Metric:** DTP3 coverage is globally recognized by the World Health Organization (WHO) and UNICEF as the primary proxy indicator for national immunization system performance.
* **The Systemic Gap:** A widening gap between the first dose (DTP1) and the third dose (DTP3) indicates a systemic structural failure in healthcare retention, supply chains, or access.
* **The Mission:** Waiting for retrospective survey data means arriving too late. This tool uses predictive forecasting to flag at-risk countries years in advance, allowing for targeted financial and logistical interventions before coverage collapses.
* ---

## ⚙️ End-to-End Data & ML Pipeline

The architecture follows a rigorous ETL (Extract, Transform, Load) and machine learning workflow:

```text
[ Raw UNICEF Data ] ──> [ Data Reshaping & Merging ] ──> [ Feature Engineering ]
                                                                      │
[ Streamlit Web App ] <── [ 3-Tab UI Analytics ] <── [ MLP Regressor & SHAP ]
```


### 1. Data Ingestion & Harmonization (The ETL Pipeline)
* **Reshaping:** Implements a dynamic wide-to-long transformation of UNICEF time-series data using `pandas` melt operations, ensuring longitudinal data consistency.
* **Feature Enrichment:** Executes systematic merges across DTP1, DTP3, and Gap datasets using `iso3` and `Year` keys. It enriches the master dataset with regional and economic metadata (`Region`, `Income_Group`).
* **Normalization:** Converts categorical descriptors into numerical feature codes (`Region_Code`, `Subregion_code`) to optimize neural network training stability.


### 2. Predictive Modeling (The Neural Network)
* Deploys a **Multi-Layer Perceptron (MLPRegressor)** architecture trained to capture highly complex, non-linear spatiotemporal relationships in healthcare data.
* Extrapolates historical indicators into a reliable 2030 predictive vector for every recorded country.


### 3. Explainable AI (XAI) Pipeline
* Implements a **SHAP (SHapley Additive exPlanations)** engine built on cooperative Game Theory.
* Computes exact Shapley feature importance scores globally across the entire model and locally for individual countries, converting a traditional "black-box" neural network into a transparent, audit-ready clinical tool.

---

## 🔍 Key Insights & Clinical Recommendations

* **The Dominant Driver:** Global SHAP analysis definitively demonstrates that **historical DTP1 Coverage is the overwhelming predictor** of 2030 DTP3 outcomes. It completely eclipses temporal progression (`Year`) and macro-regional categorizations.
* **Strategic Takeaway:** The AI model mathematically validates that ensuring a child receives their *first* dose is the single strongest guarantee of operational success. Global health budgets should heavily prioritize initial access programs, as high DTP1 coverage inherently creates a stable pipeline for full immunization compliance.

---

## 📊 Streamlit App Web Architecture

The frontend is deployed as an interactive data science dashboard structured into three strategic modules designed for both technical and non-technical stakeholders:

* **📉 Tab 1: Global Risk Distribution** Displays a high-impact categorical bar chart breaking down at-risk nations into **Critical**, **At Risk**, and **Watchlist** tiers. Accompanied by a unified data engine displaying the comprehensive list of the **15 Watchlist Countries** sorted by structural priority.
* **🧠 Tab 2: Global AI Drivers** An enterprise summary visualization powered by SHAP that highlights the mathematical weight of each model input, providing immediate scientific validity to policymakers.
* **🇲🇦 Tab 3: Country-Specific Logic** Features an interactive dropdown menu allowing users to select any sovereign nation and deconstruct its unique 2030 prediction vector via dynamic SHAP Waterfall plots.

---

## 🛠️ Local Installation & Deployment

To run this predictive analytics platform on your local machine, follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/vaccination-early-warning.git](https://github.com/your-username/vaccination-early-warning.git)
cd vaccination-early-warning
2. Install Dependencies
```

Ensure you have Python 3.9+ installed, then install the required data science dependencies:
```
pip install pandas numpy scikit-learn matplotlib seaborn shap streamlit
```
3. Launch the Application

Run the Streamlit server from your terminal:
```
streamlit run app.py
```
🧰 Tech Stack Summary

Frontend UI: Streamlit

Core ML Engine: Scikit-Learn (Multi-Layer Perceptron Neural Networks)

Model Explainability: SHAP Engine

Data Pipelines & Analytics: Pandas, NumPy, Matplotlib, Seaborn 


