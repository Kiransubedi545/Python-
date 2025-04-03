import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

# Load dataset
data_path = r"C:\Users\kiran\Desktop\data\Compressive Strength\Concrete_data.csv"
df = pd.read_csv(data_path)

# Features and target
X = df.drop('Concrete_compressive_strength', axis=1)
y = df['Concrete_compressive_strength']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Fold setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Metrics storage
r2_list, rmse_list, mae_list, mape_list = [], [], [], []
acc_list, prec_list, rec_list, f1_list = [], [], [], []

# Run K-Fold
for train_idx, test_idx in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Regression metrics
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Classification conversion (>40 MPa)
    y_test_class = (y_test > 40).astype(int)
    y_pred_class = (y_pred > 40).astype(int)
    
    acc = accuracy_score(y_test_class, y_pred_class)
    prec = precision_score(y_test_class, y_pred_class)
    rec = recall_score(y_test_class, y_pred_class)
    f1 = f1_score(y_test_class, y_pred_class)
    
    # Append results
    r2_list.append(r2)
    rmse_list.append(rmse)
    mae_list.append(mae)
    mape_list.append(mape)
    acc_list.append(acc)
    prec_list.append(prec)
    rec_list.append(rec)
    f1_list.append(f1)

# Display average performance
print("\n📊 K-Fold Validation Results (Random Forest, 5 Folds):")
print(f"R²:      {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
print(f"RMSE:    {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
print(f"MAE:     {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
print(f"MAPE:    {np.mean(mape_list):.2f}% ± {np.std(mape_list):.2f}%")
print(f"Accuracy:{np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
print(f"Precision:{np.mean(prec_list):.4f} ± {np.std(prec_list):.4f}")
print(f"Recall:  {np.mean(rec_list):.4f} ± {np.std(rec_list):.4f}")
print(f"F1 Score:{np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
