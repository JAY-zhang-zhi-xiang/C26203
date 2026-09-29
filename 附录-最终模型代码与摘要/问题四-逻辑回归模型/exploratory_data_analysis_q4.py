import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei']  # Or another font that supports Chinese
plt.rcParams['axes.unicode_minus'] = False  # To display the minus sign correctly

def load_data(file_path):
    """Loads data from an Excel file."""
    print(f"Loading data from {file_path}...")
    return pd.read_excel(file_path)

def summarize_data(df):
    """Prints a summary of the dataframe."""
    print("\n--- Data Summary ---")
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDescriptive Statistics:\n", df.describe())

def plot_distributions(df, output_dir):
    """Plots distributions of numerical features."""
    print("\n--- Plotting Distributions ---")
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.distplot(df[feature].dropna(), kde=True)  # Use distplot and drop NaNs
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plot_path = os.path.join(output_dir, f"dist_{feature}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved distribution plot for {feature} to {plot_path}")

def plot_correlations(df, output_dir):
    """Plots a heatmap of correlations between numerical features."""
    print("\n--- Plotting Correlations ---")
    plt.figure(figsize=(16, 12))
    numerical_df = df.select_dtypes(include=np.number)
    sns.heatmap(numerical_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Heatmap of Numerical Features")
    plot_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved correlation heatmap to {plot_path}")

def main():
    """Main function to run the EDA."""
    # --- Configuration ---
    file_path = "./q4data.xlsx"
    output_dir = "./Question4/eda_plots"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Run EDA ---
    df = load_data(file_path)
    summarize_data(df)
    plot_distributions(df, output_dir)
    plot_correlations(df, output_dir)

    print("\n--- EDA for Question 4 is complete. Check the 'Question4/eda_plots' directory for plots. ---")

if __name__ == "__main__":
    main()