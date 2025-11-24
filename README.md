### **Project Title:** Synthetic Data Forensics & Financial Risk Modeling
**Competition:** Kaggle Playground Series S5E11 - Predicting Loan Payback
**Role:** Machine Learning Engineer / Data Scientist
**Tools:** Python, XGBoost, LightGBM, CatBoost, Scikit-Learn, Pandas, GPU Computing

#### **1. Executive Summary**
Developed a high-performance binary classification ensemble to predict loan repayment probabilities. The project standout was a **"Forensic Data Science"** approach: identifying that the competition's synthetic dataset had stripped critical financial columns (`loan_term`, `installment`) found in the original source data. I successfully "reverse-engineered" these missing features using a teacher-student imputation model, unlocking powerful financial ratios (like Payment-to-Income) that significantly boosted model performance to **0.9224 ROC AUC**.

#### **2. The Challenge**
* **Objective:** Predict `loan_paid_back` (0/1) probabilities for synthetic loan data.
* **Data Constraints:** The provided dataset contained noise and missing standard banking features (e.g., loan term length), limiting the ability to calculate affordability ratios.
* **Model Bias:** Initial models were heavily biased (85% importance) toward a single feature (`employment_status`), ignoring subtle financial health indicators.

#### **3. Key Strategy: The "Teacher" Pipeline**
Instead of treating the data as immutable, I treated it as a puzzle with missing pieces.
* **Data Recovery:** Located the original real-world dataset the synthetic version was based on.
* **Teacher Model:** Trained a Random Forest regressor on the *original* data to learn the relationship between `Loan Amount` + `Interest Rate` $\to$ `Loan Term` (36 vs 60 months).
* **Imputation:** Applied this "Teacher" model to the competition data to predict the missing loan terms.
* **Financial Engineering:** With the recovered term, I calculated the **Amortization Schedule** for every borrower, deriving the "Golden Feature": **Payment-to-Income Ratio (PTI)**.

#### **4. Advanced Modeling & Optimization**
* **Fixing Feature Dominance:** To stop the model from "lazily" relying on Employment Status, I implemented **K-Fold Target Encoding**. This converted categorical labels into risk probabilities (e.g., `Unemployed` $\to$ `0.24` probability), forcing the model to look deeper at the financial ratios I engineered.
* **Heterogeneous Stacking:** Built a 3-model ensemble to maximize diversity:
    * **LightGBM:** Optimized with Target Encoding (Best single model: 0.921 AUC).
    * **XGBoost:** Tuned for stability on numerical ratios.
    * **CatBoost:** Trained on raw categorical strings (Ordered Boosting) to capture non-linear patterns the other two missed.
* **GPU Acceleration:** Implemented a custom GPU-accelerated cross-validation loop (utilizing CUDA/Tesla P100) to train 3,000+ estimators per fold in minutes rather than hours.

#### **5. Results**
* **Final Score:** Achieved a **0.9224 ROC AUC**, placing in the top tier of the leaderboard.
* **Improvement:** The "Teacher" strategy and Target Encoding improved the baseline model score by over **0.003 AUC**, a massive margin in competitive machine learning.
* **Artifacts:** Delivered a robust, reusable pipeline for handling synthetic data discrepancies.

---

### **Bullet Points for Resume (CV)**
* **Developed a Loan Default Prediction System** achieving **0.92+ ROC AUC** by engineering an ensemble of XGBoost, LightGBM, and CatBoost models.
* **Engineered a "Data Recovery" Pipeline** using supervised learning to impute missing financial terms from an external dataset, enabling the calculation of critical Debt-to-Income (DTI) ratios.
* **Optimized Training Efficiency** by writing custom **GPU-accelerated training loops** for Gradient Boosting libraries, reducing experiment turnaround time by 10x.
* **Solved Feature Dominance Issues** by implementing **K-Fold Target Encoding**, preventing overfitting to high-cardinality categorical variables.

---

### **Technical Skills Demonstrated**
* **Feature Engineering:** Amortization math, Target Encoding, Synthetic Data augmentation.
* **Modeling:** Gradient Boosting (XGB/LGBM/CatBoost), Stacking, Blending.
* **Validation:** Stratified K-Fold, Out-of-Fold (OOF) predictions to prevent leakage.
* **DevOps:** GPU utilization (CUDA), Python memory management for large datasets.
