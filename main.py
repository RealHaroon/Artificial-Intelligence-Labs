# ============================================
# Lab 1: Diabetes Diagnostic Tool
# Binary Classification with Logistic Regression
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_curve, roc_auc_score,
    classification_report
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

# ============================================
# STEP 1: LOAD DATA
# ============================================
print("="*60)
print("STEP 1: LOADING DATA")
print("="*60)

# Load the dataset (assumes you've downloaded diabetes.csv from Kaggle)
# Download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
df = pd.read_csv('diabetes.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nColumn names and types:")
print(df.dtypes)

print(f"\nBasic statistics:")
print(df.describe())

print(f"\nMissing values:")
print(df.isnull().sum())

print(f"\nClass distribution:")
print(df['Outcome'].value_counts())
print(f"Diabetes positive rate: {df['Outcome'].mean():.2%}")

# ============================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================
print("\n" + "="*60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*60)

# Check for zeros in features that shouldn't be zero
# (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("\nZero values (likely missing data):")
for col in zero_cols:
    zero_count = (df[col] == 0).sum()
    print(f"{col}: {zero_count} ({zero_count/len(df)*100:.1f}%)")

# Replace zeros with NaN for proper imputation
df_clean = df.copy()
df_clean[zero_cols] = df_clean[zero_cols].replace(0, np.nan)

# Visualization 1: Outcome distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].bar(['No Diabetes', 'Diabetes'], df['Outcome'].value_counts().sort_index())
axes[0, 0].set_title('Class Distribution')
axes[0, 0].set_ylabel('Count')

# Visualization 2: Feature distributions by outcome
features_to_plot = ['Glucose', 'BMI', 'Age']
for idx, feature in enumerate(features_to_plot):
    ax = axes[(idx+1)//2, (idx+1)%2]
    df.boxplot(column=feature, by='Outcome', ax=ax)
    ax.set_title(f'{feature} by Diabetes Outcome')
    ax.set_xlabel('Outcome (0=No, 1=Yes)')
    plt.sca(ax)
    plt.xticks([1, 2], ['No Diabetes', 'Diabetes'])

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=100, bbox_inches='tight')
print("\nEDA plots saved as 'eda_plots.png'")
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=100, bbox_inches='tight')
print("Correlation matrix saved as 'correlation_matrix.png'")
plt.show()

# ============================================
# STEP 3: PREPARE DATA FOR MODELING
# ============================================
print("\n" + "="*60)
print("STEP 3: DATA PREPARATION")
print("="*60)

# Separate features and target
X = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split: 70% train, 15% validation, 15% test (with stratification)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)  # 0.176 of 85% ≈ 15% of total

print(f"\nTraining set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")

print(f"\nClass distribution in training set:")
print(y_train.value_counts(normalize=True))

# ============================================
# STEP 4: BUILD PIPELINE & TRAIN MODEL
# ============================================
print("\n" + "="*60)
print("STEP 4: MODEL TRAINING")
print("="*60)

# Create a Pipeline with imputation, scaling, and Logistic Regression
# We'll tune C (regularization) and class_weight on validation set

# Test different configurations
configs = [
    {'C': 0.1, 'class_weight': None},
    {'C': 1.0, 'class_weight': None},
    {'C': 1.0, 'class_weight': 'balanced'},
    {'C': 10.0, 'class_weight': 'balanced'},
]

best_f1 = 0
best_config = None
best_pipeline = None

print("\nTuning hyperparameters on validation set:")
print("-" * 60)

for config in configs:
    # Build pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=config['C'],
            class_weight=config['class_weight'],
            random_state=42,
            max_iter=1000
        ))
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_val_pred = pipeline.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"C={config['C']}, class_weight={config['class_weight']}: "
          f"Val Accuracy={val_acc:.4f}, Val F1={val_f1:.4f}")
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_config = config
        best_pipeline = pipeline

print(f"\n✓ Best configuration: C={best_config['C']}, "
      f"class_weight={best_config['class_weight']}")
print(f"Best validation F1-score: {best_f1:.4f}")

# ============================================
# STEP 5: EVALUATE ON TEST SET
# ============================================
print("\n" + "="*60)
print("STEP 5: MODEL EVALUATION ON TEST SET")
print("="*60)

# Get predictions and probabilities
y_test_pred = best_pipeline.predict(X_test)
y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]

# Calculate metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_proba)

print(f"\nTest Set Performance (Threshold = 0.5):")
print("-" * 60)
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}")
print(f"ROC-AUC:   {test_roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Visualize confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
axes[0].set_title('Confusion Matrix (Threshold=0.5)')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Plot 2: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
axes[1].plot(fpr, tpr, label=f'ROC Curve (AUC={test_roc_auc:.4f})', linewidth=2)
axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=100, bbox_inches='tight')
print("\nEvaluation plots saved as 'model_evaluation.png'")
plt.show()

# ============================================
# STEP 6: THRESHOLD TUNING
# ============================================
print("\n" + "="*60)
print("STEP 6: THRESHOLD TUNING")
print("="*60)

# Find optimal threshold using validation set
y_val_proba = best_pipeline.predict_proba(X_val)[:, 1]

# Test thresholds from 0.3 to 0.7
thresholds_to_test = np.arange(0.3, 0.71, 0.05)
threshold_results = []

for thresh in thresholds_to_test:
    y_val_pred_thresh = (y_val_proba >= thresh).astype(int)
    prec = precision_score(y_val, y_val_pred_thresh)
    rec = recall_score(y_val, y_val_pred_thresh)
    f1 = f1_score(y_val, y_val_pred_thresh)
    threshold_results.append({
        'threshold': thresh,
        'precision': prec,
        'recall': rec,
        'f1': f1
    })

threshold_df = pd.DataFrame(threshold_results)
print("\nThreshold tuning results (on validation set):")
print(threshold_df.to_string(index=False))

# Select threshold with best F1
best_threshold = threshold_df.loc[threshold_df['f1'].idxmax(), 'threshold']
print(f"\n✓ Optimal threshold: {best_threshold:.2f}")

# Re-evaluate on test set with optimal threshold
y_test_pred_tuned = (y_test_proba >= best_threshold).astype(int)

tuned_accuracy = accuracy_score(y_test, y_test_pred_tuned)
tuned_precision = precision_score(y_test, y_test_pred_tuned)
tuned_recall = recall_score(y_test, y_test_pred_tuned)
tuned_f1 = f1_score(y_test, y_test_pred_tuned)

print(f"\nTest Set Performance (Threshold = {best_threshold:.2f}):")
print("-" * 60)
print(f"Accuracy:  {tuned_accuracy:.4f}")
print(f"Precision: {tuned_precision:.4f}")
print(f"Recall:    {tuned_recall:.4f}")
print(f"F1-Score:  {tuned_f1:.4f}")

# Visualize threshold impact
plt.figure(figsize=(10, 6))
plt.plot(threshold_df['threshold'], threshold_df['precision'], 
         marker='o', label='Precision', linewidth=2)
plt.plot(threshold_df['threshold'], threshold_df['recall'], 
         marker='s', label='Recall', linewidth=2)
plt.plot(threshold_df['threshold'], threshold_df['f1'], 
         marker='^', label='F1-Score', linewidth=2)
plt.axvline(best_threshold, color='red', linestyle='--', 
            label=f'Optimal Threshold={best_threshold:.2f}')
plt.xlabel('Classification Threshold')
plt.ylabel('Score')
plt.title('Precision, Recall, F1 vs Threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('threshold_tuning.png', dpi=100, bbox_inches='tight')
print("\nThreshold tuning plot saved as 'threshold_tuning.png'")
plt.show()

# ============================================
# STEP 7: SAVE MODEL
# ============================================
print("\n" + "="*60)
print("STEP 7: SAVING MODEL")
print("="*60)

# Save the trained pipeline and optimal threshold
model_artifacts = {
    'pipeline': best_pipeline,
    'threshold': best_threshold,
    'feature_names': list(X.columns),
    'config': best_config
}

joblib.dump(model_artifacts, 'diabetes_model.pkl')
print("✓ Model saved as 'diabetes_model.pkl'")

# Test loading
loaded_artifacts = joblib.load('diabetes_model.pkl')
print("✓ Model loaded successfully")
print(f"  - Features: {loaded_artifacts['feature_names']}")
print(f"  - Optimal threshold: {loaded_artifacts['threshold']}")

print("\n" + "="*60)
print("LAB 1 COMPLETE!")
print("="*60)
print("\nSummary:")
print(f"  - Model: Logistic Regression (C={best_config['C']}, "
      f"class_weight={best_config['class_weight']})")
print(f"  - Test ROC-AUC: {test_roc_auc:.4f}")
print(f"  - Test F1-Score (optimal threshold): {tuned_f1:.4f}")
print(f"  - Optimal threshold: {best_threshold:.2f}")
