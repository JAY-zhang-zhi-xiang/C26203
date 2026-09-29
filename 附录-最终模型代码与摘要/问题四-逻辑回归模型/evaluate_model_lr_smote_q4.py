import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc
import numpy as np

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

# --- Configuration ---
model_path = "./Question4/classification_model_lr_smote.joblib"
test_data_path = "./Question4/test_data.csv"
output_dir = "./Question4/"

# --- Load Model and Data ---
print(f"Loading model from {model_path}")
pipeline = joblib.load(model_path)

print(f"Loading test data from {test_data_path}")
test_df = pd.read_csv(test_data_path)
X_test = test_df.drop('target', axis=1)
y_test = test_df['target']

# --- Predictions ---
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# --- Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(cm)

# --- Visualizations ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['正常', '异常'], yticklabels=['正常', '异常'])
plt.title('Confusion Matrix (Logistic Regression with SMOTE)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(f"{output_dir}confusion_matrix_lr_smote.png")
print(f"Confusion matrix saved to {output_dir}confusion_matrix_lr_smote.png")

# ROC and PR Curves
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall (PR) Curve')
plt.legend(loc="lower left")

plt.suptitle('ROC and PR Curves (Logistic Regression with SMOTE)', fontsize=16)
plt.savefig(f"{output_dir}roc_pr_curves_lr_smote.png")
print(f"ROC and PR curves saved to {output_dir}roc_pr_curves_lr_smote.png")