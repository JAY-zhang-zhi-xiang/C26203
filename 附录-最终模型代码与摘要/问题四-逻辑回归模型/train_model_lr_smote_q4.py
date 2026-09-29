import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

def train_model(train_data_path, model_output_path):
    """
    Trains a classification model on SMOTE-balanced data and saves it.

    Args:
        train_data_path (str): Path to the training data CSV file.
        model_output_path (str): Path to save the trained model.
    """
    print("--- Starting Model Training on SMOTE Data ---")

    # 1. Load Data
    train_df = pd.read_csv(train_data_path)
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']

    # Standardize features and train a logistic regression model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    print("Model pipeline created with StandardScaler and Logistic Regression.")

    # 3. Train the Model
    pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # 4. Save the Model
    joblib.dump(pipeline, model_output_path)
    print(f"Trained model saved to {model_output_path}")
    print("\n--- Model Training Complete ---")

def main():
    """Main function to run the model training."""
    # --- Configuration ---
    train_data_path = "./Question4/train_data_smote.csv"
    model_output_path = "./Question4/classification_model_lr_smote.joblib"

    # --- Run Training ---
    train_model(train_data_path, model_output_path)

if __name__ == "__main__":
    main()