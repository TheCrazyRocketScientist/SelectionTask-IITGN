import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score, classification_report

# Create directory to save models
save_path = "saved_models"
os.makedirs(save_path, exist_ok=True)

# Load Data
df = pd.read_csv("X_train.csv")
X = df.drop(columns=["Label"])  # Features
y = df["Label"]  # Target

# Explicit Train-Test Split (Use last 20% as Test)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Feature Selection using RFE
rfe_selector = RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=25)
X_train_selected = rfe_selector.fit_transform(X_train, y_train)
X_test_selected = rfe_selector.transform(X_test)

# Convert NumPy arrays back to DataFrames for compatibility
selected_feature_names = X.columns[rfe_selector.support_]
X_train_selected = pd.DataFrame(X_train_selected, columns=selected_feature_names, index=X_train.index)
X_test_selected = pd.DataFrame(X_test_selected, columns=selected_feature_names, index=X_test.index)

# TimeSeriesSplit for Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

# Define Models
models = {
    "RandomForest": RandomForestClassifier(),
    "LightGBM": LGBMClassifier(verbose=-1),
    "XGBoost": XGBClassifier(verbosity=0, eval_metric="logloss")
}

# Define Hyperparameters for Grid Search
param_grid = {
    "RandomForest": {"n_estimators": [50, 100], "max_depth": [5, 10]},
    "LightGBM": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
    "XGBoost": {"n_estimators": [50, 100], "max_depth": [3, 6]}
}

# Train Models with Grid Search, Evaluate & Save Best Model
for model_name, model in models.items():
    grid_search = GridSearchCV(model, param_grid[model_name], cv=tscv, scoring="f1", n_jobs=-1, refit=True)
    grid_search.fit(X_train_selected, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best Model for {model_name}: {grid_search.best_params_}")

    # Evaluate on Test Set
    y_pred = best_model.predict(X_test_selected)
    f1 = f1_score(y_test, y_pred)
    
    print(f"F1 Score for {model_name}: {f1:.4f}")
    print(f"Classification Report for {model_name}:")
    report = classification_report(y_test, y_pred)
    print(report)

    # Save the best model
    model_filename = os.path.join(save_path, f"{model_name}_best_model.pkl")
    joblib.dump(best_model, model_filename)
    print(f"Saved {model_name} model to {model_filename}")

    # Save classification report for XGBoost
    if model_name == "XGBoost":
        report_filename = os.path.join(save_path, "XGBoost_classification_report.txt")
        with open(report_filename, "w") as f:
            f.write(report)
        print(f"Saved XGBoost classification report to {report_filename}")

    # Feature Importance Plot
    if hasattr(best_model, "feature_importances_"):
        plt.figure(figsize=(8, 5))
        feature_importance = best_model.feature_importances_
        pd.Series(feature_importance, index=selected_feature_names).nlargest(10).plot(kind='barh')
        plt.title(f"Top 10 Feature Importances - {model_name}")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature Name")
        plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
corr_matrix = X_train_selected.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

print("Selected Features:")
print(selected_feature_names.tolist())