import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# Example structure – fill with your actual metrics from each model
results = [
    {"Model": "Linear Regression",       "R2": 0.75, "RMSE": 6.3, "MAE": 5.1, "MAPE": 12.3, "Accuracy": 0.81, "Precision": 0.83, "Recall": 0.78, "F1": 0.80},
    {"Model": "Ridge Regression",        "R2": 0.76, "RMSE": 6.1, "MAE": 5.0, "MAPE": 12.0, "Accuracy": 0.82, "Precision": 0.84, "Recall": 0.79, "F1": 0.81},
    {"Model": "Decision Tree",           "R2": 0.85, "RMSE": 4.8, "MAE": 3.6, "MAPE": 9.2,  "Accuracy": 0.88, "Precision": 0.89, "Recall": 0.87, "F1": 0.88},
    {"Model": "Random Forest",           "R2": 0.91, "RMSE": 3.9, "MAE": 2.9, "MAPE": 7.0,  "Accuracy": 0.91, "Precision": 0.93, "Recall": 0.90, "F1": 0.91},
    {"Model": "SVR",                     "R2": 0.78, "RMSE": 5.9, "MAE": 4.7, "MAPE": 11.5, "Accuracy": 0.80, "Precision": 0.81, "Recall": 0.78, "F1": 0.79},
    {"Model": "XGBoost",                 "R2": 0.93, "RMSE": 3.5, "MAE": 2.7, "MAPE": 6.2,  "Accuracy": 0.93, "Precision": 0.94, "Recall": 0.92, "F1": 0.93},
    {"Model": "KNN",                     "R2": 0.74, "RMSE": 6.5, "MAE": 5.3, "MAPE": 13.1, "Accuracy": 0.78, "Precision": 0.79, "Recall": 0.76, "F1": 0.77},
    {"Model": "MLP Regressor",           "R2": 0.88, "RMSE": 4.2, "MAE": 3.1, "MAPE": 8.0,  "Accuracy": 0.89, "Precision": 0.90, "Recall": 0.88, "F1": 0.89},
    {"Model": "LightGBM",                "R2": 0.92, "RMSE": 3.7, "MAE": 2.8, "MAPE": 6.5,  "Accuracy": 0.92, "Precision": 0.93, "Recall": 0.91, "F1": 0.92},
    {"Model": "Gaussian Process",        "R2": 0.79, "RMSE": 5.7, "MAE": 4.5, "MAPE": 10.8, "Accuracy": 0.83, "Precision": 0.85, "Recall": 0.81, "F1": 0.83},
    {"Model": "CatBoost",                "R2": 0.93, "RMSE": 3.4, "MAE": 2.6, "MAPE": 6.1,  "Accuracy": 0.94, "Precision": 0.95, "Recall": 0.93, "F1": 0.94},
    {"Model": "Keras ANN",               "R2": 0.91, "RMSE": 3.8, "MAE": 2.9, "MAPE": 6.8,  "Accuracy": 0.91, "Precision": 0.92, "Recall": 0.90, "F1": 0.91},
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