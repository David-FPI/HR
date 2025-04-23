import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
st.set_page_config(layout="wide")
st.title("🧠 HR Analytics Dashboard - Employee Retention")

@st.cache_data
def load_data():
    return pd.read_csv("HR.csv")

df = load_data()
st.markdown("## TASK #1: Import Libraries and Dataset")
st.markdown("We begin with reading the HR dataset and importing essential libraries.")

st.markdown("## TASK #2: Read Dataset")
st.dataframe(df.head())

st.markdown("### Dataset Description")
st.markdown("""
- `satisfaction_level`: Employee satisfaction level
- `last_evaluation`: Last performance evaluation score
- `number_project`: Number of projects assigned
- `average_montly_hours`: Monthly working hours
- `time_spend_company`: Tenure in the company (years)
- `Work_accident`: Had a work accident? (1 = Yes, 0 = No)
- `left`: Left the company? (1 = Yes, 0 = No)
- `promotion_last_5years`: Promotion in last 5 years (1 = Yes, 0 = No)
- `sales`: Department
- `salary`: Salary level (low, medium, high)
""" )

st.markdown("## TASK #3: Dataset Overview")
st.subheader("Step 3.1 | Dataset Basic Information")
buffer = st.empty()
buffer.text(str(df.info()))

st.subheader("Step 3.2 | Summary Statistics")
st.dataframe(df.describe())
st.dataframe(df.describe(include='object'))

st.markdown("## TASK #4: Data Visualization")
continuous_features = ['satisfaction_level', 'last_evaluation', 'number_project',
                        'average_montly_hours', 'time_spend_company']
df[[col for col in df.columns if col not in continuous_features]] = \
    df[[col for col in df.columns if col not in continuous_features]].astype('object')

st.subheader("Step 4.1 | Numerical Features vs Target")
fig, axes = plt.subplots(3, 2, figsize=(16, 10))
for ax, col in zip(axes.flatten(), continuous_features):
    sns.kdeplot(data=df, x=col, hue='left', fill=True, ax=ax, palette={0: '#009c05', 1: 'darkorange'})
    ax.set_title(f"{col} vs Left")
axes[2,1].axis('off')
plt.tight_layout()
st.pyplot(fig)

st.subheader("Step 4.2 | Categorical Features vs Target")
cat_features = ['Work_accident', 'promotion_last_5years', 'sales', 'salary']
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
for i, ax in enumerate(axes.flatten()):
    sns.countplot(x=cat_features[i], hue='left', data=df, ax=ax, palette={0: '#009c05', 1: 'darkorange'})
    ax.set_title(f"{cat_features[i]} vs Left")
plt.tight_layout()
st.pyplot(fig)

st.markdown("## TASK #5: Data Preprocessing")
st.subheader("Step 5.1 | Missing Values")
fig, ax = plt.subplots()
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="Blues", ax=ax)
st.pyplot(fig)

st.subheader("Step 5.2 | Encoding Categorical Features")
df_encoded = pd.get_dummies(df, columns=['sales'], drop_first=True)
le = LabelEncoder()
df_encoded['salary'] = le.fit_transform(df_encoded['salary'])

st.subheader("Step 5.3 | Correlation Heatmap")
corr = df_encoded.corr(numeric_only=True)
plt.figure(figsize=(15, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
st.pyplot(plt)

st.subheader("Step 5.4 | Train-Test Split")
X = df_encoded.drop('left', axis=1)
y = df_encoded['left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
st.write("Train shape:", X_train.shape)
st.write("Test shape:", X_test.shape)

st.markdown("## TASK #6: Logistic Regression")

# Đảm bảo y là int (nhị phân 0/1)
df_encoded['left'] = df_encoded['left'].astype(int)
X = df_encoded.drop('left', axis=1)
y = df_encoded['left']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)
# Kiểm tra phân bố và kiểu dữ liệu
st.write("Class distribution in y_train:", y_train.value_counts())
st.write("Data type of y_train:", y_train.dtype)
# Huấn luyện Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred_log)
st.text(f"Accuracy: {accuracy * 100:.2f}%")
st.text("Classification Report:\n" + classification_report(y_test, y_pred_log))
# Confusion Matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title("Confusion Matrix - Logistic Regression")
st.pyplot(fig)


st.markdown("## TASK #7: Random Forest")
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
st.text(f"Accuracy: {accuracy_score(y_test, y_pred_rf)*100:.2f}%")
st.text("Classification Report:\n" + classification_report(y_test, y_pred_rf))
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens', ax=ax)
ax.set_title("Confusion Matrix - Random Forest")
st.pyplot(fig)


st.markdown("## TASK #8: Random Forest")
# Model names
models = ['Logistic Regression', 'Random Forest']

# Accuracy
accuracy = [77, 99]

# Precision, Recall, F1-Score for Class 0
precision_0 = [0.80, 0.99]
recall_0 = [0.93, 1.00]
f1_score_0 = [0.86, 0.99]

# Precision, Recall, F1-Score for Class 1
precision_1 = [0.53, 1.00]
recall_1 = [0.24, 0.97]
f1_score_1 = [0.33, 0.98]

# Plotting Accuracy
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.bar(models, accuracy, color=['blue', 'green'])
plt.title('Accuracy Comparison')
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')

# Plotting Precision, Recall, F1-Score for Class 0
x = np.arange(len(models))
width = 0.2

plt.subplot(2, 2, 2)
plt.bar(x - width, precision_0, width, label='Precision (Class 0)', color='blue')
plt.bar(x, recall_0, width, label='Recall (Class 0)', color='orange')
plt.bar(x + width, f1_score_0, width, label='F1-Score (Class 0)', color='green')
plt.xticks(x, models)
plt.title('Class 0 (Did not leave) Metrics Comparison')
plt.ylim(0, 1.2)
plt.legend()

# Plotting Precision, Recall, F1-Score for Class 1
plt.subplot(2, 2, 3)
plt.bar(x - width, precision_1, width, label='Precision (Class 1)', color='blue')
plt.bar(x, recall_1, width, label='Recall (Class 1)', color='orange')
plt.bar(x + width, f1_score_1, width, label='F1-Score (Class 1)', color='green')
plt.xticks(x, models)
plt.title('Class 1 (Left the company) Metrics Comparison')
plt.ylim(0, 1.2)
plt.legend()

plt.tight_layout()
plt.show()

# After training and evaluating both the Logistic Regression and Random Forest classifiers, let's compare their performance metrics and draw some conclusions.

st.markdown("""
### **1. Accuracy:**
- **Logistic Regression**: 77%
- **Random Forest**: 99%

**Conclusion:** Random Forest significantly outperforms Logistic Regression in terms of accuracy, achieving a near-perfect score. This indicates that Random Forest is more effective at correctly predicting both classes in the dataset.

### **2. Precision, Recall, and F1-Score:**

#### **A. For Class 0 (Did not leave the company):**

- **Logistic Regression:**
  - Precision: 0.80
  - Recall: 0.93
  - F1-Score: 0.86

- **Random Forest:**
  - Precision: 0.99
  - Recall: 1.00
  - F1-Score: 0.99

#### **B. For Class 1 (Left the company):**

- **Logistic Regression:**
  - Precision: 0.53
  - Recall: 0.24
  - F1-Score: 0.33

- **Random Forest:**
  - Precision: 1.00
  - Recall: 0.97
  - F1-Score: 0.98

### **3. Confusion Matrix:**

- **Logistic Regression**: Shows a significant number of false negatives (employees who left the company but were predicted to stay).
- **Random Forest**: Has almost no false positives or false negatives, further confirming its superior performance.

### **Overall Conclusion:**

The Random Forest model consistently outperforms the Logistic Regression model across all key metrics: accuracy, precision, recall, and F1-score. The Random Forest model's ability to handle complex patterns in the data makes it a better choice for this classification task.

For practical applications, especially when identifying employees at risk of leaving the company, Random Forest is the more reliable model, offering both high precision and recall. Therefore, it would be recommended to use **Random Forest** over **Logistic Regression** for this dataset.
""")
