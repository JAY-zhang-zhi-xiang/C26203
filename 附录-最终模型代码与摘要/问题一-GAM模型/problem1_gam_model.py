import pandas as pd
import numpy as np
from pygam import LinearGAM, s, f
import matplotlib.pyplot as plt
import os

# --- 1. Setup and Data Preparation ---
def parse_ga_to_days(ga_str):
    if isinstance(ga_str, str) and 'w' in ga_str:
        parts = ga_str.split('w')
        weeks = int(parts[0])
        days = 0
        if '+' in parts[1]:
            days = int(parts[1].replace('+', ''))
        return weeks * 7 + days
    try:
        return int(ga_str)
    except (ValueError, TypeError):
        return None

# Create output directory for plots
output_dir = './gam_model_plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load data
file_path = './data.xlsx'
data = pd.read_excel(file_path)

# Clean and transform data
data.columns = data.columns.str.strip()
data['GA_days'] = data['检测孕周'].apply(parse_ga_to_days)

if data['Y染色体浓度'].dtype == 'object':
    data['Y染色体浓度'] = data['Y染色体浓度'].str.replace('%', '', regex=False).astype(float)

data = data.rename(columns={
    'Y染色体浓度': 'Y_concentration',
    '孕妇BMI': 'BMI',
    '年龄': 'Age'
})

# Log transform and handle zeros/negatives
data['log_Y'] = np.log(data['Y_concentration'].replace(0, 1e-9))
data['log_BMI'] = np.log(data['BMI'].replace(0, 1e-9))
data['log_Age'] = np.log(data['Age'].replace(0, 1e-9))

# Final dataset for the model
model_data = data[['log_Y', 'GA_days', 'log_BMI', 'log_Age']].dropna()

X = model_data[['GA_days', 'log_BMI', 'log_Age']].values
y = model_data['log_Y'].values

# --- 2. Build and Fit GAM Model ---
# Model: log(Y) ~ s(GA_days) + s(log_BMI) + s(log_Age)
gam = LinearGAM(s(0, n_splines=20, lam=0.6) + s(1, n_splines=20, lam=0.6) + s(2, n_splines=20, lam=0.6)).fit(X, y)

import io
from contextlib import redirect_stdout

# --- 3. Save Model Summary ---
summary_path = './problem1_gam_model_summary.txt'
with io.StringIO() as buf, redirect_stdout(buf):
    gam.summary()
    summary_output = buf.getvalue()

with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary_output)

print(f"GAM model summary saved to {summary_path}")

# --- 4. Generate and Save Partial Dependence Plots ---
plt.style.use('seaborn-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

titles = ['GA_days', 'log_BMI', 'log_Age']

for i, ax in enumerate(axes):
    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
    
    ax.plot(XX[:, i], pdep, color='blue')
    ax.plot(XX[:, i], confi, color='grey', linestyle='--')
    ax.set_title(f'Partial Dependence for {titles[i]}', fontsize=14)
    ax.set_xlabel(titles[i], fontsize=12)
    if i == 0:
        ax.set_ylabel('log(Y_concentration) Contribution', fontsize=12)

fig.suptitle('GAM Partial Dependence Plots', fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plot_path = os.path.join(output_dir, 'gam_partial_dependence_plots.png')
plt.savefig(plot_path)
plt.close()

print(f"GAM partial dependence plots saved to {plot_path}")
print("\nScript finished successfully.")