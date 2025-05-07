# 🧠 HR Analytics Dashboard - Employee Retention

This project analyzes employee retention using machine learning models to predict whether an employee will leave the company.  
It includes data visualization, preprocessing, model training, and evaluation.

## 📂 Dataset Description

The dataset contains information about employee performance and demographics:

| Column Name            | Description                                   |
| ---------------------- | --------------------------------------------- |
| `satisfaction_level`   | Employee satisfaction level                   |
| `last_evaluation`      | Last performance evaluation score             |
| `number_project`       | Number of projects assigned                   |
| `average_montly_hours` | Monthly working hours                         |
| `time_spend_company`   | Tenure in the company (years)                 |
| `Work_accident`        | Had a work accident? (1 = Yes, 0 = No)        |
| `left`                 | Left the company? (1 = Yes, 0 = No)           |
| `promotion_last_5years`| Promotion in last 5 years? (1 = Yes, 0 = No)  |
| `sales`                | Department                                    |
| `salary`               | Salary level (low, medium, high)              |

---

## 🚀 Tasks Overview

### ✅ TASK #1: Import Libraries and Dataset
- Imported necessary libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`).
- Loaded the HR dataset.

### ✅ TASK #2: Dataset Overview
- Checked basic dataset information.
- Viewed summary statistics.

### ✅ TASK #3: Data Visualization
- Visualized **numerical features** vs **target** (`left`).
- Visualized **categorical features** vs **target**.

### ✅ TASK #4: Data Preprocessing
- Checked for missing values (0 missing).
- Encoded categorical features (`sales`, `salary`).
- Visualized correlation heatmap.
- Split data into **Train** and **Test** sets:
  - **Train shape:** `(11,999, 17)`
  - **Test shape:** `(3,000, 17)`

---

## 🤖 Machine Learning Models

### 1️⃣ Logistic Regression
- **Accuracy:** `76.77%`
- **Classification Report:**

| Class        | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| 0 (Stayed)   | 0.80      | 0.93   | 0.86     | 2286    |
| 1 (Left)     | 0.53      | 0.24   | 0.33     | 714     |

- Shows high false negatives — many employees who left were predicted to stay.

---

### 2️⃣ Random Forest Classifier
- **Accuracy:** `99.20%`
- **Classification Report:**

| Class        | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| 0 (Stayed)   | 0.99      | 1.00   | 0.99     | 2286    |
| 1 (Left)     | 1.00      | 0.97   | 0.98     | 714     |

- Excellent performance with almost no false positives or negatives.

---

## 📊 Conclusion

- **Random Forest** significantly outperforms **Logistic Regression** across all metrics:
  - Better accuracy, precision, recall, and F1-score.
  - More reliable for practical applications like predicting employee attrition.

**Recommendation:**  
Use **Random Forest** for identifying employees at risk of leaving the company.

---

## 📌 Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
