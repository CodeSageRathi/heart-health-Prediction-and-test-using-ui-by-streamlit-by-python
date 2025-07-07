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
    accuracy_score, confusion_matrix, roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Setup
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load dataset from CSV
def load_heart_data(file_path="heart1.csv"):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded dataset from {file_path} with shape {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ File {file_path} not found. Please ensure the file exists in the correct directory.")
    except Exception as e:
        raise Exception(f"❌ Error loading {file_path}: {str(e)}")

# Load and preprocess
def load_and_preprocess():
    df = load_heart_data()

    if df.isnull().sum().sum() > 0:
        raise ValueError("❌ Dataset contains missing values.")
    if "target" not in df.columns:
        raise ValueError("❌ 'target' column missing from dataset.")

    X = df.drop("target", axis=1)
    y = df["target"]

    logging.info(f"Original class distribution: {Counter(y)}")

    try:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        logging.info(f"After SMOTE class distribution: {Counter(y_res)}")
    except Exception as e:
        logging.warning(f"SMOTE failed: {e}")
        X_res, y_res = X, y

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    return X_train, X_test, y_train, y_test, scaler, X.columns

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred) * 100
    auc = roc_auc_score(y_test, y_proba) * 100 if y_proba is not None else None

    print(f"✅ Accuracy: {acc:.2f}%")
    if auc:
        print(f"✅ ROC-AUC: {auc:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title(f"{model.__class__.__name__} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{model.__class__.__name__}_confusion_matrix.png")
    plt.close()

    return acc

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

# Grid search for logistic regression
def tune_and_train_logreg(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['liblinear', 'lbfgs']
    }
    model = LogisticRegression(max_iter=1000, random_state=42)
    grid = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

# Grid search for random forest
def tune_and_train_rf(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }
    model = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

# Grid search for XGBoost
def tune_and_train_xgboost(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    grid = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

# Train and compare models
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

    print(f"\n🏆 Best Model: {best_name} with Accuracy: {best_score:.2f}%")
    return best_model

# Save model and scaler
def save_model_and_scaler(model, scaler):
    joblib.dump(model, "heart_disease_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("💾 Model saved as 'heart_disease_model.pkl'")
    print("💾 Scaler saved as 'scaler.pkl'")
    print(f"🔢 scikit-learn version: {sklearn.__version__}")

# Main
def main():
    start = time.time()
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess()
    best_model = train_all_models(X_train, y_train, X_test, y_test)

    print("⚠️ Skipping calibration for now...")

    save_model_and_scaler(best_model, scaler)
    plot_feature_importance(best_model, feature_names)

    print(f"⏱️ Training complete in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    main()