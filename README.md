# 📊 Credit Risk Assessment & Default Prediction System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Algorithm-XGBoost-EB6424.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-ready, end-to-end Machine Learning pipeline and web platform designed to evaluate **credit default risk** for loan applicants. Built on financial data containing over **307,000+ customer records** and **120+ features**, this project automates data preprocessing, handles extreme class imbalance, trains tuned ensemble models, and delivers predictions through an interactive Flask web application supporting both **real-time single evaluations** and **high-throughput batch CSV processing**.

---

## 🧠 Project Mind Map

The mind map below highlights the core dimensions of the system from data sourcing to production serving:

```mermaid
mindmap
  root((Credit Risk Prediction))
    Data Ingestion & Exploration
      307K+ Loan Applications
      122 Raw Attributes
      Outlier & Missing Value Profiling
      Target Class Imbalance 91.9% vs 8.1%
    Data Transformation
      Median SimpleImputer
      RobustScaler Outlier Protection
      One-Hot Encoding with Strict Alignment
      Dimensional Interaction Features
    Model Engineering
      Ensemble Benchmarking
        Logistic Regression
        Random Forest Classifier
        Gradient Boosting
        XGBoost Classifier
      5-Fold Cross-Validation
      GridSearchCV Hyperparameter Tuning
    Feature Selection
      Tree Feature Importances
      Top 10 Risk Indicators
      Pre-computed Fallback Imputation
    Production Deployment
      Flask RESTful Web Interface
      Real-Time Form Scoring
      Batch CSV Upload & Processing
      Downloadable Scored Reports
```

---

## 🏗️ System Architecture Workflow

```mermaid
flowchart TD
    A[Raw Loan Data application_train.csv] --> B[Data Ingestion Component]
    B --> C[Data Transformation Component]
    
    subgraph Preprocessing_Pipeline [Data Preprocessing Pipeline]
        C --> D1[Median Imputation]
        D1 --> D2[RobustScaler Normalization]
        D2 --> D3[One-Hot Encoding & Schema Alignment]
        D3 --> D4[Feature Store train.npy / test.npy]
    end

    D4 --> E[Model Trainer Component]
    
    subgraph Model_Selection [Model Training & Tuning]
        E --> F1[Base Model Evaluation: XGBoost, Random Forest, GBDT]
        F1 --> F2[GridSearchCV 5-Fold Cross Validation]
        F2 --> F3[Export Best Model model.pkl]
        F2 --> F4[Top Feature Extractor top_features.json]
    end

    F3 --> G[Flask Web Application app.py]
    F4 --> G
    
    subgraph Web_Serving [Inference Service]
        G --> H1[Single Applicant Scoring form.html]
        G --> H2[Batch CSV Inference upload.html]
        H2 --> H3[Export Scored CSV with Predictions]
    end
```

---

## 📑 Feature Engineering & Key Indicators

Through feature importance analysis on tree ensembles, the model isolates the **top 10 most influential risk drivers** used to optimize the real-time scoring interface:

| Rank | Feature Name | Description | Financial Interpretation |
| :---: | :--- | :--- | :--- |
| **1** | `EXT_SOURCE_3` | Normalized score from external data source 3 | Independent external credit risk rating |
| **2** | `EXT_SOURCE_2` | Normalized score from external data source 2 | Secondary credit bureau risk assessment |
| **3** | `DAYS_ID_PUBLISH` | Days before the application client changed ID | Identity and verification stability |
| **4** | `EXT_SOURCE_1` | Normalized score from external data source 1 | Primary credit bureau risk metric |
| **5** | `DAYS_REGISTRATION` | Days before application client modified registration | Residence and geographic stability |
| **6** | `DAYS_BIRTH` | Client's age in days (negative value) | Applicant maturity and career stage |
| **7** | `REGION_POPULATION_RELATIVE` | Normalized population of client's residential region | Regional economic strength & urban density |
| **8** | `DAYS_LAST_PHONE_CHANGE` | Days since the client updated contact phone | Contact stability and fraud resistance |
| **9** | `AMT_INCOME_TOTAL` | Total annual income of applicant | Direct debt-service capacity |
| **10** | `DAYS_EMPLOYED` | Days employed prior to loan application | Job security and income continuity |

### 🧮 Domain-Specific Features & Ratios Engineered

To capture non-linear relationships and direct debt-servicing ability, several domain-specific financial indicators and interaction features were engineered:

#### 1. Financial Capacity & Debt Burden Ratios
| Engineered Feature | Formula | Business & Risk Interpretation |
| :--- | :--- | :--- |
| `CREDIT_INCOME_PERCENT` | `AMT_CREDIT / AMT_INCOME_TOTAL` | Measures total credit exposure as a multiple of annual income (leverage indicator). |
| `ANNUITY_INCOME_PERCENT` | `AMT_ANNUITY / AMT_INCOME_TOTAL` | Measures monthly payment burden against earnings (debt-to-income liquidity strain). |
| `CREDIT_TERM` | `AMT_ANNUITY / AMT_CREDIT` | Effective repayment rate representing expected payback duration and interest load. |
| `GOODS_CREDIT_PERCENT` | `AMT_GOODS_PRICE / AMT_CREDIT` | Down-payment proxy indicating how much of the loan directly funds the purchase. |

#### 2. Employment Stability & Demographic Indicators
| Engineered Feature | Formula | Business & Risk Interpretation |
| :--- | :--- | :--- |
| `DAYS_EMPLOYED_PERCENT` | `DAYS_EMPLOYED / DAYS_BIRTH` | Percentage of an applicant's lifetime spent in continuous employment. |
| `AGE_YEARS` | `-1 * (DAYS_BIRTH / 365.25)` | Applicant's age converted from negative days to chronological years. |
| `EMPLOYED_YEARS` | `-1 * (DAYS_EMPLOYED / 365.25)` | Career longevity with the current employer. |
| `PHONE_CHANGE_YEARS` | `-1 * (DAYS_LAST_PHONE_CHANGE / 365.25)` | Years since contact info update (indicator of residency and stability). |

#### 3. Polynomial Interaction Terms (Degree 2 & 3)
Tree models and linear classifiers benefit significantly from multiplicative interactions between external bureau scores and client age:
| Interaction Feature | Formula | Impact on Target Correlation |
| :--- | :--- | :--- |
| `EXT_2_x_EXT_3` | `EXT_SOURCE_2 * EXT_SOURCE_3` | Strongest single negative correlation (-0.194) with default risk. |
| `EXT_1_x_EXT_2_x_EXT_3` | `EXT_SOURCE_1 * EXT_SOURCE_2 * EXT_SOURCE_3` | Triple external bureau consensus rating (-0.190 correlation). |
| `EXT_2_x_EXT_3_x_AGE` | `EXT_SOURCE_2 * EXT_SOURCE_3 * DAYS_BIRTH` | Cross-validates external scores against applicant maturity (-0.181 correlation). |
| `EXT_2_POW2_x_EXT_3` | `(EXT_SOURCE_2 ^ 2) * EXT_SOURCE_3` | Amplifies non-linear credit score confidence (-0.176 correlation). |

---

## 🤖 Models & Evaluation

The training framework executes comparative benchmark runs followed by grid search hyperparameter tuning:

| Model | Technique | Key Hyperparameters Evaluated | Primary Strength |
| :--- | :--- | :--- | :--- |
| **XGBoost Classifier** | Gradient Tree Boosting | `max_depth: [5, 7]`, `n_estimators: [10, 20]` | Superior speed & handles complex non-linear splits |
| **Random Forest** | Bagging Ensemble | `n_estimators: [100, 200]`, `max_depth: [5, 10]` | Resilient to overfitting on high-dimensional data |
| **Gradient Boosting** | Sequential Boosting | `n_estimators: [100]`, `criterion: friedman_mse` | Minimizes residual loss across iterations |
| **Logistic Regression** | Linear Classification | `C: [0.0001]`, L2 Regularization | High-interpretability baseline benchmark |

---

## 📂 Project Directory Structure

```text
CREDIT_PREDICTION_PROJECT/
│
├── .gitignore                      # Git exclusion rules for large datasets & venvs
├── README.md                       # Comprehensive documentation
├── app.py                          # Flask web application routes & endpoints
├── requirements.txt                # Required Python libraries and dependencies
│
├── artifacts/                      # Serialized pipelines and model assets
│   ├── model.pkl                   # Production-ready trained ML model
│   ├── preprocessor.pkl            # Imputer, Scaler, and categorical encoders
│   └── top_features.json           # Top 10 extracted features & feature means
│
├── config/                         # Configuration declarations
│   └── model.yaml                  # Model search spaces & GridSearchCV hyperparams
│
├── notebooks/                      # Exploratory analysis & prototyping
│   └── home_credit.ipynb           # Comprehensive EDA, correlations & experimentation
│
├── src/                            # Modular source code
│   ├── __init__.py
│   ├── exception.py                # Standardized custom exception handling
│   ├── logger.py                   # Automated file and console logging
│   │
│   ├── components/                 # Core pipeline stages
│   │   ├── __init__.py
│   │   ├── data_ingestion.py       # Raw dataset loading & train/test partitioning
│   │   ├── data_transformation.py  # Imputation, RobustScaler & categorical encoding
│   │   └── model_trainer.py        # Model benchmarking & GridSearchCV tuning
│   │
│   ├── pipeline/                   # Execution pipelines
│   │   ├── __init__.py
│   │   ├── train_pipeline.py       # End-to-end training orchestrator
│   │   └── predict_pipeline.py     # Single & batch inference pipeline
│   │
│   └── utils/                      # Helper utilities
│       ├── __init__.py
│       ├── extract_top_features.py # Feature importance extractor
│       └── main_utils.py           # Object serialization & YAML reader
│
├── templates/                      # Flask UI templates
│   ├── form.html                   # Interactive single applicant evaluation form
│   └── upload.html                 # Batch CSV file upload and download interface
│
└── tests/                          # Automated unit and integration tests
    ├── test_data_ingestion.py
    ├── test_data_transformation.py
    ├── test_extract_top_features.py
    ├── test_model_trainer.py
    ├── test_predict_pipeline.py
    └── test_train_pipeline.py
```

---

## 💻 Web Application Capabilities

| Feature | Single Prediction Mode (`/`) | Batch Prediction Mode (`/batch_predict`) |
| :--- | :--- | :--- |
| **Input Method** | Dynamic interactive HTML form | CSV file upload |
| **Features Required** | Top 10 key risk indicators (remaining filled via population means) | Full application CSV dataset |
| **Processing Time** | Real-time (< 100ms) | Asynchronous bulk processing |
| **Output Format** | Binary classification (`Default Risk` vs `Low Risk`) | Downloadable CSV with appended prediction column |
| **Target User** | Loan officer assisting in-person client | Credit Risk team conducting portfolio batch audits |

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
* Python 3.10 or higher
* Git

### 2. Clone the Repository
```bash
git clone https://github.com/prakharmav/credit_risk.git
cd credit_risk
```

### 3. Set Up Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000/`**

---

## 🔄 Retraining the Pipeline

To re-execute data ingestion, preprocessing, and model training from scratch:

```bash
python -m src.pipeline.train_pipeline
```

The updated model and transformation pipelines will automatically be serialized into the `artifacts/` directory.

---

## 🧪 Running Unit Tests

Run test suites using `pytest`:

```bash
pytest tests/
```

---

## 📌 Detailed Project Summary

Credit risk assessment represents one of the most critical decision-making processes in retail banking, consumer finance, and fintech underwriting. Traditional scoring methods often rely on rigid rule-based scorecards that fail to capture subtle, non-linear relationships across hundreds of financial and demographic indicators.

This project delivers a modern, production-grade Machine Learning solution built from the ground up:

* **End-to-End Data Pipeline:** Managed the complete lifecycle across 307,511 historical loan applications and 122 raw attributes, resolving severe class imbalance (8.07% default rate), pervasive missing values, and high-dimensional categorical features.
* **Robust Preprocessing & Feature Engineering:** Implemented median imputation resilient to outliers, `RobustScaler` transformation, and strict dummy alignment. Engineered high-correlation domain indicators (credit-to-income, annuity burden, loan term) and degree-3 polynomial interaction terms between external credit bureau sources.
* **Model Benchmarking & Fine-Tuning:** Compared baseline linear models (Logistic Regression) against tree-based ensembles (Random Forest, Gradient Boosting, and XGBoost). Leveraged 5-fold cross-validated `GridSearchCV` to prevent data leakage and maximize generalization performance.
* **Decoupled Production Serving:** Developed a modular Flask web application capable of both real-time, low-latency scoring for in-person applicants and asynchronous high-throughput batch evaluation for entire enterprise loan books.

---

## 🏁 Conclusion & Key Takeaways

1. **Information Synergy Beats Volume:** While the raw dataset contained over 120 features, feature importance analysis revealed that external bureau scores (`EXT_SOURCE_1/2/3`) combined with identity and residency stability (`DAYS_BIRTH`, `DAYS_ID_PUBLISH`, `DAYS_EMPLOYED`) account for the vast majority of predictive variance.
2. **Debt Serviceability Over Gross Income:** Raw income (`AMT_INCOME_TOTAL`) alone is an incomplete risk signal; normalized financial ratios (such as `ANNUITY_INCOME_PERCENT` and `CREDIT_TERM`) provided substantially stronger separation between defaulting and repaying borrowers.
3. **Operational Business Impact:**
   * **Default Reduction:** Early identification of high-probability default profiles helps reduce Non-Performing Loan (NPL) ratios.
   * **Turnaround Speed:** Reduces loan underwriting turnaround from multiple business days to instantaneous sub-100ms algorithmic scoring.
   * **Auditing Scalability:** Batch CSV evaluation allows financial institutions to perform periodic stress-testing and batch portfolio re-evaluations with minimal operational overhead.
4. **Future Enhancements & Scalability Roadmap:**
   * **Model Explainability (XAI):** Integrate SHAP (SHapley Additive exPlanations) and LIME to deliver legally compliant, applicant-facing adverse action notices.
   * **Containerization & CI/CD:** Dockerize the Flask application and establish automated GitHub Actions workflows for continuous integration and model regression testing.
   * **MLOps Integration:** Incorporate MLflow for experiment tracking, model registry, and automated data drift monitoring.

---

## 👥 Author
* **Prakhar Raj** — [GitHub](https://github.com/prakharmav)

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
