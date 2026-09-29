import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from pygam import LinearGAM, s


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


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'model_comparison_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = './newtest.xlsx'
    print(f'Reading data from: {file_path}')
    data = pd.read_excel(file_path)
    data.columns = data.columns.str.strip()

    # GA to days
    data['GA_days'] = data['检测孕周'].apply(parse_ga_to_days)

    # Y concentration may contain '%'
    if data['Y染色体浓度'].dtype == 'object':
        data['Y染色体浓度'] = data['Y染色体浓度'].astype(str).str.replace('%', '', regex=False)
    data['Y染色体浓度'] = pd.to_numeric(data['Y染色体浓度'], errors='coerce')

    # Rename for convenience
    data = data.rename(columns={
        'Y染色体浓度': 'Y_concentration',
        '孕妇BMI': 'BMI',
        '年龄': 'Age'
    })

    # Log transforms with small epsilon to avoid -inf
    eps = 1e-9
    data['log_Y'] = np.log((data['Y_concentration']).replace(0, eps))
    data['log_BMI'] = np.log((data['BMI']).replace(0, eps))
    data['log_Age'] = np.log((data['Age']).replace(0, eps))

    model_data = data[['log_Y', 'GA_days', 'log_BMI', 'log_Age']].dropna()

    X = model_data[['GA_days', 'log_BMI', 'log_Age']]
    y = model_data['log_Y']

    # OLS
    X_ols = sm.add_constant(X)
    ols_model = sm.OLS(y, X_ols).fit()
    ols_predictions = ols_model.predict(X_ols)
    ols_residuals = ols_model.resid

    # GAM with preset lambda (from previous CV)
    lam = 0.1778
    gam_model = LinearGAM(s(0, lam=lam) + s(1, lam=lam) + s(2, lam=lam)).fit(X, y)
    gam_predictions = gam_model.predict(X)

    # Residual plots for OLS
    print('Generating OLS residual plots...')
    plt.style.use('seaborn-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, (var, name) in enumerate(zip(['GA_days', 'log_BMI', 'log_Age'], ['GA (days)', 'log(BMI)', 'log(Age)'])):
        axes[i].scatter(X[var], ols_residuals, alpha=0.3, edgecolors='k', s=20)
        axes[i].axhline(y=0, color='r', linestyle='--')
        axes[i].set_xlabel(name, fontsize=12)
        axes[i].set_title(f'OLS Residuals vs. {name}', fontsize=14)
        if i == 0:
            axes[i].set_ylabel('Residuals', fontsize=12)
    fig.suptitle('OLS Model Residual Analysis (newtest)', fontsize=18, y=1.02)
    plt.tight_layout()
    residual_plot_path = os.path.join(output_dir, 'ols_residual_plots_newtest.png')
    plt.savefig(residual_plot_path)
    plt.close(fig)
    print(f'Residual plots saved to {residual_plot_path}')

    # Fitted vs Actuals
    print('Generating Fitted vs. Actuals plot...')
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(y, ols_predictions, alpha=0.4, label='OLS Predictions', s=30, c='orange')
    ax.scatter(y, gam_predictions, alpha=0.4, label='GAM Predictions', s=30, c='green')
    min_y, max_y = y.min(), y.max()
    ax.plot([min_y, max_y], [min_y, max_y], 'r--', lw=2, label='Perfect Fit')
    ax.set_xlabel('Actual log(Y_concentration)', fontsize=12)
    ax.set_ylabel('Predicted log(Y_concentration)', fontsize=12)
    ax.set_title('Model Predictions vs. Actual Values (newtest)', fontsize=16)
    ax.legend()
    ax.grid(True)

    fit_plot_path = os.path.join(output_dir, 'fitted_vs_actuals_comparison_newtest.png')
    plt.savefig(fit_plot_path)
    plt.close(fig)
    print(f'Fitted vs. Actuals plot saved to {fit_plot_path}')

    print('\nComparison on newtest finished successfully.')


if __name__ == '__main__':
    main()