# Predicting Car Insurance Claims Using Logistic Regression

## Project Overview
This project applies **Logistic Regression** to predict whether a customer will file a claim on their car insurance. By analyzing **On the Road** car insurance data, we identify the most predictive feature for insurance claims. The goal is to build a simple, accurate model that insurance companies can use to make informed decisions.

## Dataset Description
- **Dataset Name**: `car_insurance.csv`
- **Target Variable**: `outcome` (0: No claim, 1: Made a claim)
- **Objective**: Identify the single feature that results in the most accurate model.
- **Feature Types**:
  - **Categorical**: `age`, `gender`, `driving_experience`, `education`, etc.
  - **Numerical**: `credit_score`, `annual_mileage`, `speeding_violations`, etc.

## Exploratory Data Analysis (EDA)
### 1. Data Summary
- Displays dataset shape and summary statistics.
- Examines the class balance of `outcome`.

### 2. Data Visualization
- **Class Balance**: Bar plot of `outcome` distribution.
- **Correlation Heatmap**: Shows relationships between numerical features and `outcome`.
- **Univariate Analysis**: Histograms for numerical features.
- **Boxplots**: Examines categorical features' relationships with `outcome`.
- **Pairplots**: Explores pairwise relationships among numerical features, separated by `outcome`.
- **Grouped Means**: Summarizes the average value of `outcome` for each category in categorical features.

## Model Development
### 1. Feature Selection
- Excludes `id` and `outcome` columns.
- Loops through all other features and fits a separate **Logistic Regression model** for each feature.

### 2. Model Implementation
- **Fitting Logistic Regression**:
  - Uses `statsmodels.formula.api.logit`.
  - Formula: `outcome ~ feature`

- **Prediction & Accuracy Calculation**:
  - Predicts claim probability using a threshold of 0.5.
  - Compares predictions with actual outcomes.
  - Computes accuracy using `sklearn.metrics.accuracy_score`.

### 3. Identifying the Best Feature
- Stores accuracy scores for all features in a DataFrame.
- Identifies the feature with the **highest accuracy**.

## Results
- **Best Feature**: `driving_experience`
- **Accuracy**: **77.71%**

This feature proved to be the most predictive in determining whether a customer would file a car insurance claim.

## Repository Structure
```
├── data
│   ├── car_insurance.csv  # Dataset
├── notebooks
│   ├── EDA.ipynb          # Exploratory Data Analysis
│   ├── modeling.ipynb     # Model Implementation
├── scripts
│   ├── preprocess.py      # Data Cleaning
│   ├── train_model.py     # Model Training
├── README.md              # Project Description
```

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/insurance-claims-prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the EDA notebook:
   ```bash
   jupyter notebook notebooks/EDA.ipynb
   ```
4. Train the model:
   ```bash
   python scripts/train_model.py
   ```

## Technologies Used
- **Python** (Pandas, NumPy, Scikit-Learn, Statsmodels, Matplotlib, Seaborn)
- **Jupyter Notebook** for interactive analysis
- **GitHub** for version control

## Contact
For questions or collaboration, feel free to reach out!

---
This project demonstrates how **machine learning** can be applied to real-world insurance challenges, helping companies optimize their pricing and claim estimation strategies.
