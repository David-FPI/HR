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
