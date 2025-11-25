# Beyond the FICO Score: A Feature Engineering Approach to Modern Credit Risk

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-GPU-green)
![Status](https://img.shields.io/badge/Status-Complete-success)
![Domain](https://img.shields.io/badge/Domain-Fintech-orange)

> **"Traditional credit models rely on static snapshots. This project builds a dynamic risk engine."**

## 📖 The Context
In the lending world, the difference between a profitable portfolio and a massive loss often lies in the gray areas. A borrower might look great on paper (high income) but actually be drowning in monthly obligations. Traditional "black box" models often miss these nuances.

For this project, I moved beyond standard hyperparameter tuning to build a **domain-driven machine learning pipeline**. My goal was to engineer a model that mimics the intuition of a human underwriter—assessing "true affordability" and "payment strain"—but operates at the scale of an algorithm.

## 🛠️ The Strategy
Instead of throwing raw data at a model, I focused on three strategic pillars:

### 1. Domain-Driven Feature Engineering
Raw columns like `annual_income` rarely tell the whole story. I engineered synthetic features to capture the **real** financial picture:
* **True Affordability:** Calculated `available_income` (post-debt cash flow) to see how tight a borrower's budget actually is.
* **Payment Strain:** Derived `payment_to_income` ratios to understand the specific burden of *this* new loan.
* **Composite Risk Scoring:** Created a weighted index combining credit history, interest volatility, and debt burden into a single signal.

### 2. Protecting Against Bias & Leakage
Financial data is messy. High-cardinality features like `grade_subgrade` can confuse models or introduce bias.
* **Solution:** I implemented **K-Fold Target Encoding**. This allows the model to learn the historical risk of specific categories without "peeking" at future data (leakage), ensuring the model is production-safe.

### 3. Enterprise-Grade Validation
A model is useless if it's unstable.
* **Solution:** I utilized **8-Fold Stratified Cross-Validation**. By simulating performance across different random customer segments, I ensured the model remains stable even if the applicant pool fluctuates.

---

## 💻 The Logic (Code Highlight)
The most impactful part of this project was the "translation layer"—turning financial concepts into vectorizable features.

```python
def create_risk_features(df):
    """
    Transforms raw applicant data into actionable risk metrics
    mimicking underwriter logic.
    """
    df = df.copy()

    # 1. Real Affordability (Capacity to Pay)
    # Measures income remaining AFTER accounting for current debt load
    df["available_income"] = df["annual_income"] * (1 - df["debt_to_income_ratio"])
    
    # 2. Payment Strain
    # What % of monthly cash flow is eaten by THIS specific loan?
    df["monthly_payment"] = df["loan_amount"] * df["interest_rate"] / 1200
    df["payment_to_income"] = df["monthly_payment"] / (df["annual_income"] / 12 + 1)

    # 3. Composite Risk Index
    # A weighted score combining DTI, Credit Score, and Loan Cost.
    # We weight DTI highest (0.40) as it is the leading indicator of distress.
    df["composite_risk_score"] = (
        df["debt_to_income_ratio"] * 0.40
        + (850 - df["credit_score"]) / 850 * 0.35
        + df["interest_rate"] / 100 * 0.25
    )

    return df
```

Evaluation & Business Impact

Goal: Quantify model performance and translate technical metrics into business value.

### 1. Model Performance (ROC-AUC)

The final XGBoost model achieved an ROC-AUC score of 0.8931 on the hold-out test set.



This score indicates excellent discriminatory power. The ROC Curve plots the True Positive Rate against the False Positive Rate. An AUC of 0.89 means the model is highly effective at ranking borrowers from "riskiest" to "safest," confirming that the engineered features provided strong predictive signals that raw data alone missed.

### 2. Optimization: Solving the Imbalance

Credit datasets are inherently imbalanced (the majority of loans are repaid). A standard model (default threshold of 0.50) often prioritizes raw accuracy over risk detection, leading to a high number of **False Positives** (approving borrowers who actually default).

- **The Strategy:** I implemented Threshold Tuning, shifting the decision boundary from 50% to 65% probability.

- **The Result:** This stricter standard drastically reduced financial exposure to potential defaults (False Positives).

    
- **Business Trade-off:** While this slightly increased the rejection rate for borderline candidates, it prioritized Capital Preservation—the primary KPI for risk management.



### 3. Validation: Feature Importance

Did the domain knowledge actually help?

XGBoost feature importance analysis confirmed that engineered behavioral features were the top drivers of the model's decisions, outperforming raw demographic data:

1. *available_income* **(Engineered):** The most critical predictor, proving that residual cash flow matters more than gross salary.

2. *payment_strain* **(Engineered):** Successfully flagged applicants where the loan payment was too high relative to their monthly liquidity.

3. *debt_to_income_ratio* **(Raw):** A standard but essential risk metric.

 ## **Tech Stack**

- Core: Python, Pandas, NumPy

- Modeling: XGBoost (GPU Accelerated)

- Validation: Scikit-Learn (StratifiedKFold)

- Tuning: Optuna


   

## Tech Stack

  * **Core:** Python, Pandas, NumPy
  * **Modeling:** XGBoost (GPU Accelerated)
  * **Validation:** Scikit-Learn (StratifiedKFold)
  * **Tuning:** Optuna

## How to Run

1.  Clone the repo:
    ```bash
    git clone https://github.com/ernselito/loan-risk-prediction.git
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the notebook:
    ```bash
    jupyter notebook predicting-loan-payback.ipynb
    ```

-----

*If you are interested in discussing financial modeling or ML engineering, feel free to connect\!*
