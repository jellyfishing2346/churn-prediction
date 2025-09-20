import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
try:
    import shap
except ImportError:
    print("shap package not found. Please install it using 'pip install shap'.")
    shap = None

# 1. Batch training/testing script
DATASETS = [
    'customer_churn_sample.csv',
    'customer_churn_test.csv',
    'customer_data.csv'
]

results = []

for file in DATASETS:
    if not os.path.exists(file):
        print(f"File not found: {file}")
        continue
    df = pd.read_csv(file)
    # Preprocessing
    X = df.drop(['Churn'], axis=1)
    # Drop columns that are not numeric or not intended for encoding
    if 'CustomerID' in X.columns:
        X = X.drop('CustomerID', axis=1)
    if 'CustomerType' in X.columns:
        X['CustomerType'] = LabelEncoder().fit_transform(X['CustomerType'])
    # Drop any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    y = df['Churn']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{file} Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred))
    results.append({'file': file, 'accuracy': acc})
    # Save model and scaler for API use
    joblib.dump(model, f"model_{file.replace('.csv','')}.joblib")
    joblib.dump(scaler, f"scaler_{file.replace('.csv','')}.joblib")
    # SHAP explainability
    if shap is not None:
        import matplotlib.pyplot as plt
        explainer = shap.TreeExplainer(model)
        try:
            shap_values = explainer.shap_values(X_test)
            # Handle binary/multiclass output
            if isinstance(shap_values, list):
                # For binary classification, use the positive class
                shap_values_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_values_to_plot = shap_values
            shap.summary_plot(shap_values_to_plot, X_test, show=False)
            plt.savefig(f"shap_summary_{file.replace('.csv','')}.png")
            plt.close()
        except Exception as e:
            print(f"SHAP error for {file}: {e}")

# Save results
pd.DataFrame(results).to_csv('batch_test_results.csv', index=False)
print('Batch testing complete. Models, scalers, and SHAP summaries saved.')
