import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and test data
@st.cache_resource
def load_model():
    return joblib.load("xgboost_accident_model.pkl")

@st.cache_data
def load_data():
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv").values.ravel()
    return X_test, y_test

model = load_model()
X_test, y_test = load_data()

# Get probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Streamlit UI
st.title("🚨 Accident Severity Risk Prediction")
st.markdown("Adjust the threshold to see how it affects classification results.")

threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.01)
y_pred = (y_proba >= threshold).astype(int)

# Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
st.subheader("Classification Report")
st.write(pd.DataFrame(report).transpose())

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'High'], yticklabels=['Low', 'High'], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
st.subheader("Precision-Recall Trade-off")
fig2, ax2 = plt.subplots()
ax2.plot(thresholds, precision[:-1], label='Precision')
ax2.plot(thresholds, recall[:-1], label='Recall')
ax2.set_xlabel("Threshold")
ax2.set_ylabel("Score")
ax2.set_title("Precision vs. Recall")
ax2.legend()
st.pyplot(fig2)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
st.subheader("ROC Curve")
fig3, ax3 = plt.subplots()
ax3.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
ax3.plot([0, 1], [0, 1], linestyle='--')
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.set_title("ROC Curve")
ax3.legend()
st.pyplot(fig3)