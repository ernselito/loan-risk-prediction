
# Beyond the FICO Score: A Feature Engineering Approach to Modern Credit RiskPermalink

 **"Traditional credit models rely on static snapshots. This project builds a dynamic risk engine."**

## The Context

In the lending world, the difference between a profitable portfolio and a massive loss often lies in the gray areas. A borrower might look great on paper (high income) but actually be drowning in monthly obligations. Traditional "black box" models often miss these nuances.

For this project, I moved beyond standard hyperparameter tuning to build a **domain-driven machine learning pipeline**. My goal was to engineer a model that mimics the intuition of a human underwriter—assessing "true affordability" and "payment strain"—but operates at the scale of an algorithm.

## The Strategy

Instead of throwing raw data at a model, I focused on three strategic pillars:

### 1\. Domain-Driven Feature Engineering

Raw columns like `annual_income` rarely tell the whole story. I engineered synthetic features to capture the **real** financial picture:

  * **True Affordability:** Calculated `available_income` (post-debt cash flow) to see how tight a borrower's budget actually is.
  * **Payment Strain:** Derived `payment_to_income` ratios to understand the specific burden of *this* new loan.
  * **Composite Risk Scoring:** Created a weighted index combining credit history, interest volatility, and debt burden into a single signal.

### 2\. Protecting Against Bias & Leakage

Financial data is messy. High-cardinality features like `job_title` or `sub_grade` can confuse models or introduce bias.

  * **Solution:** I implemented **K-Fold Target Encoding**. This allows the model to learn the historical risk of specific categories without "peeking" at future data (leakage), ensuring the model is production-safe.

### 3\. Enterprise-Grade Validation

A model is useless if it's unstable.

  * **Solution:** I utilized **8-Fold Stratified Cross-Validation**. By simulating performance across different random customer segments, I ensured the model remains stable even if the applicant pool fluctuates.

-----

## The Logic (Code Highlight)

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

## Business Impact ("The So What?")

By prioritizing feature engineering over model complexity, this pipeline achieved:

1.  **Lower Default Exposure:** The `available_income` metric successfully flagged high-earners who were actually over-leveraged—a segment often missed by basic models.
2.  **Explainability:** Decisions can be explained to compliance teams using human-readable metrics like "Payment Strain" rather than abstract vector weights.
3.  **Scalability:** The XGBoost architecture (optimized with GPU histograms) allows this logic to assess thousands of applications in milliseconds.

## 🔧 Tech Stack

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
