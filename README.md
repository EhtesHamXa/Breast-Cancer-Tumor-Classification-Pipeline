# Breast Cancer Tumor Classification Pipeline 🔬

An end-to-end Machine Learning pipeline to classify breast cancer tumors as Malignant or Benign using clinical measurement data.

## 🚀 Overview

This project demonstrates a rigorous data science workflow—from exploratory data analysis and feature engineering to hyperparameter tuning and model evaluation. The pipeline prioritizes "Recall" as the primary business metric, ensuring that potentially malignant tumors are never misclassified as benign (minimizing False Negatives).

## 🧠 Architecture & Tech Stack

- **Machine Learning Framework:** Scikit-Learn
- **Algorithms:** Logistic Regression, Random Forest Ensembles, XGBoost
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn (Correlation Heatmaps)

## 📊 Methodology

1. **Exploratory Data Analysis (EDA):** Generated pair-plots and correlation heatmaps to identify highly collinear features (e.g., radius vs. perimeter) and remove redundant data.
2. **Feature Scaling:** Applied `StandardScaler` to ensure distance-based metrics functioned correctly without being skewed by large numerical ranges.
3. **Hyperparameter Tuning:** Utilized `GridSearchCV` with Cross-Validation to exhaustively search the parameter space and optimize the Random Forest depth and estimator count.
4. **Evaluation Metrics:** Evaluated models using Confusion Matrices, focusing heavily on maximizing Recall over absolute Accuracy due to the critical medical context.

## 🛠️ How to Run

1. **Install Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost
   ```
2. **Execute Pipeline:**
   ```bash
   python main.py
   ```

---
*Developed as a demonstration of applied statistical modeling and business-metric aligned Machine Learning.*
