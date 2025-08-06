# train_model.py
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import time
import sklearn
import logging

from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    precision_recall_fscore_support
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV

# Setup
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load dataset
def load_heart_data(file_path="heart1.csv"):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded dataset from {file_path} with shape {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ File {file_path} not found.")
    except Exception as e:
        raise Exception(f"❌ Error loading {file_path}: {str(e)}")

# Load + preprocess safely
def load_and_preprocess():
    df = load_heart_data()

    if df.isnull().sum().sum() > 0:
        raise ValueError("❌ Dataset contains missing values.")
    if "target" not in df.columns:
        raise ValueError("❌ 'target' column missing from dataset.")

    X = df.drop("target", axis=1)
    y = df["target"]

    # Split BEFORE SMOTE/Scaling to avoid leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logging.info(f"Original train distribution: {Counter(y_train)}")

    # Apply SMOTE only on training set
    try:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        logging.info(f"After SMOTE train distribution: {Counter(y_train_res)}")
    except Exception as e:
        logging.warning(f"SMOTE failed: {e}")
        X_train_res, y_train_res = X_train, y_train

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_res, y_test, scaler, X.columns

# Evaluation
def evaluate_model(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred) * 100
    auc = roc_auc_score(y_test, y_proba) * 100
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)

    print(f"✅ Accuracy: {acc:.2f}%")
    print(f"✅ ROC-AUC: {auc:.2f}%")
    print(f"✅ Precision: {p:.3f}  Recall: {r:.3f}  F1: {f1:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title(f"{model.__class__.__name__} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{model.__class__.__name__}_confusion_matrix.png")
    plt.close()

    return auc

# Feature importance
def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importances)), importances[sorted_idx], align='center')
        plt.yticks(range(len(importances)), np.array(feature_names)[sorted_idx])
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.close()

# Tuners (optimized for ROC-AUC)
def tune_and_train_logreg(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['liblinear', 'lbfgs']
    }
    model = LogisticRegression(max_iter=1000, random_state=42)
    grid = GridSearchCV(model, param_grid, scoring='roc_auc', cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def tune_and_train_rf(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }
    model = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(model, param_grid, scoring='roc_auc', cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def tune_and_train_xgboost(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    model = XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=42)
    grid = GridSearchCV(model, param_grid, scoring='roc_auc', cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

# Train and compare
def train_all_models(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': tune_and_train_logreg(X_train, y_train),
        'Random Forest': tune_and_train_rf(X_train, y_train),
        'XGBoost': tune_and_train_xgboost(X_train, y_train)
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():
        print(f"\n🔍 Evaluating {name}...")
        score = evaluate_model(model, X_test, y_test)
        if score > best_score:
            best_model = model
            best_score = score
            best_name = name

    print(f"\n🏆 Best Model: {best_name} with ROC-AUC: {best_score:.2f}%")
    return best_model

# Save everything
def save_artifacts(model, scaler, feature_names):
    joblib.dump(model, "heart_disease_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(list(feature_names), "feature_columns.pkl")
    print("💾 Model, scaler, and feature list saved.")

# Main
def main():
    start = time.time()
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess()

    best_model = train_all_models(X_train, y_train, X_test, y_test)

    # Calibrate the final model for better probability output
    calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
    calibrated_model.fit(X_train, y_train)

    save_artifacts(calibrated_model, scaler, feature_names)
    plot_feature_importance(calibrated_model, feature_names)

    print(f"⏱️ Training complete in {time.time() - start:.2f} seconds.")
    print(f"🔢 scikit-learn version: {sklearn.__version__}")

if __name__ == "__main__":
    main()
