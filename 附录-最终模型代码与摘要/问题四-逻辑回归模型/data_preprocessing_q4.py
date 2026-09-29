import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(file_path):
    """Loads data from an Excel file."""
    return pd.read_excel(file_path)

def preprocess_data(df):
    """Preprocesses the data for modeling."""
    print("--- Starting Data Preprocessing ---")

    # 1. Define Target Variable
    # Any non-null value in '染色体的非整倍体' indicates an abnormality (1), while null (NaN) values are normal (0).
    df['target'] = df['染色体的非整倍体'].notna().astype(int)
    print(f"Target variable 'target' created. Distribution:\n{df['target'].value_counts()}")

    # 2. Handle Missing Values
    # For simplicity, we'll fill numerical missing values with the median.
    numerical_cols = df.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled missing values in '{col}' with median ({median_val}).")

    # 3. Feature Selection
    # Define features to be used for modeling.
    # We exclude identifiers, date columns, and the original target columns.
    features = [
        '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
        'X染色体浓度', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
        'GC含量', '原始读段数', '唯一比对的读段数', '在参考基因组上比对的比例',
        '重复读段的比例', '被过滤掉读段数的比例', '年龄', '孕妇BMI'
    ]
    
    X = df[features]
    y = df['target']
    
    print(f"\nSelected {len(features)} features for modeling.")
    print("Final feature list:", features)

    # 4. Split Data
    # Splitting the data into training and testing sets (80/20 split).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nData split into training and testing sets.")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test

def main():
    """Main function to run the preprocessing."""
    # --- Configuration ---
    input_file = "./q4data.xlsx"
    output_dir = "./Question4/"

    # --- Run Preprocessing ---
    df = load_data(input_file)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # --- Save Processed Data ---
    # Combine features and target for saving
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_path = f"{output_dir}train_data.csv"
    test_path = f"{output_dir}test_data.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nProcessed training data saved to {train_path}")
    print(f"Processed testing data saved to {test_path}")
    print("\n--- Data Preprocessing Complete ---")

if __name__ == "__main__":
    main()