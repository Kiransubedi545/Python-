import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# Metrics from each model
results = [
   {"Model": "Linear Regression",       "R2": 0.6276, "RMSE": 9.7965, "MAE": 7.7456, "MAPE": 29.27, "Accuracy": 0.7816, "Precision": 0.7576, "Recall": 0.6329, "F1": 0.6897},
    {"Model": "Ridge Regression",        "R2": 0.6276, "RMSE": 9.7962, "MAE": 7.752,  "MAPE": 29.34, "Accuracy": 0.7816, "Precision": 0.7576, "Recall": 0.6329, "F1": 0.6897},
    {"Model": "Decision Tree Regression", "R2": 0.8348, "RMSE": 6.5254, "MAE": 4.2938, "MAPE": 13.33, "Accuracy": 0.8689, "Precision": 0.8514, "Recall": 0.7975, "F1": 0.8235},
    {"Model": "Random Forest Regression", "R2": 0.8842, "RMSE": 5.4632, "MAE": 3.7341, "MAPE": 12.32, "Accuracy": 0.8981, "Precision": 0.9028, "Recall": 0.8228, "F1": 0.8609},
    {"Model": "Support Vector Regression", "R2": 0.8727, "RMSE": 5.7276, "MAE": 3.9849, "MAPE": 12.68, "Accuracy": 0.8738, "Precision": 0.8442, "Recall": 0.8228, "F1": 0.8333},
    {"Model": "XGBoost Regression",       "R2": 0.9041, "RMSE": 4.9698, "MAE": 3.2579, "MAPE": 10.06, "Accuracy": 0.9175, "Precision": 0.9079, "Recall": 0.8734, "F1": 0.8903},
    {"Model": "KNN Regression",           "R2": 0.7143, "RMSE": 8.5801, "MAE": 6.8005, "MAPE": 23.50, "Accuracy": 0.8204, "Precision": 0.7692, "Recall": 0.7595, "F1": 0.7643},
    {"Model": "MLP Regression",           "R2": 0.8764, "RMSE": 5.6435, "MAE": 3.9299, "MAPE": 11.80, "Accuracy": 0.9126, "Precision": 0.9296, "Recall": 0.8354, "F1": 0.8800},
    {"Model": "Light GBM Regression",     "R2": 0.9126, "RMSE": 4.7470, "MAE": 3.2413, "MAPE": 10.03, "Accuracy": 0.9223, "Precision": 0.9091, "Recall": 0.8861, "F1": 0.8974},
    {"Model": "Gaussian Process Regression", "R2": -4.425, "RMSE": 37.3886, "MAE": 33.3198, "MAPE": 95.18, "Accuracy": 0.6359, "Precision": 1.0000, "Recall": 0.0506, "F1": 0.0964},
    {"Model": "CatBoost Regression",      "R2": 0.9093, "RMSE": 4.8355, "MAE": 3.6251, "MAPE": 11.67, "Accuracy": 0.9175, "Precision": 0.9189, "Recall": 0.8606, "F1": 0.8889},
    {"Model": "Keras ANN Regression",     "R2": 0.8522, "RMSE": 6.1723, "MAE": 4.6818, "MAPE": 14.20, "Accuracy": 0.8883, "Precision": 0.8500, "Recall": 0.8608, "F1": 0.8553},
] 

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Sort by R² descending
results_df = results_df.sort_values(by='R2', ascending=False)

# Display
print("📊 Final Model Comparison:\n")
print(results_df.to_string(index=False))

# Plot R² comparison
plt.figure(figsize=(10, 6))
plt.barh(results_df['Model'], results_df['R2'])
plt.xlabel('R² Score')
plt.title('Model Comparison - R² Scores')
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.show()


df = pd.DataFrame(results)

# Invert metrics where lower is better
df['RMSE'] = 1 / df['RMSE']
df['MAE'] = 1 / df['MAE']
df['MAPE'] = 1 / df['MAPE']

# Normalize all metrics to 0–1
metrics = ['R2', 'RMSE', 'MAE', 'MAPE', 'Accuracy', 'Precision', 'Recall', 'F1']
for col in metrics:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Select top 5 models by R²
top_df = df.sort_values(by='R2', ascending=False).head(5)

# Radar chart setup
labels = metrics
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for i, row in top_df.iterrows():
    values = row[labels].tolist()
    values += values[:1]
    ax.plot(angles, values, label=row['Model'], linewidth=2)
    ax.fill(angles, values, alpha=0.15)

ax.set_title("Top 5 Models - Multi-Metric Radar Chart", fontsize=16, y=1.08)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_rlabel_position(0)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(df.set_index('Model')[['R2', 'RMSE', 'Accuracy', 'F1']], annot=True, cmap='YlGnBu')
plt.title("Model Performance Heatmap")
plt.tight_layout()
plt.show()

metrics = ['R2', 'RMSE', 'MAE', 'MAPE', 'Accuracy', 'Precision', 'Recall', 'F1']
df_norm = df.copy()
for col in metrics:
    df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())

plt.figure(figsize=(12, 6))
parallel_coordinates(df_norm, 'Model', colormap='Set2')
plt.title("Parallel Coordinates Plot - Model Performance")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(df['R2'], df['Accuracy'], s=df['F1']*1000, alpha=0.6, c='teal', edgecolors='k')

for i, row in df.iterrows():
    plt.text(row['R2'], row['Accuracy'], row['Model'], fontsize=8, ha='center', va='center')

plt.xlabel("R² Score")
plt.ylabel("Accuracy")
plt.title("Bubble Chart: R² vs Accuracy (Bubble size = F1 Score)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(columns='Model').corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Between Performance Metrics")
plt.tight_layout()
plt.show()
