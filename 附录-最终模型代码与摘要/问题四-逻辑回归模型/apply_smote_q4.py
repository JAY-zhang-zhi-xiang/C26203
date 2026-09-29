import pandas as pd
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors

def smote(X, N, k):
    """
    A simplified implementation of Synthetic Minority Over-sampling Technique (SMOTE).

    Parameters:
    - X: array-like, shape (n_samples, n_features)
        Matrix containing the minority class samples.
    - N: int
        Amount of SMOTE N%.
    - k: int
        Number of nearest neighbors.

    Returns:
    - synthetic_samples: array, shape (N/100 * n_minority_samples, n_features)
        The synthetic samples.
    """
    n_minority_samples, n_features = X.shape
    
    if n_minority_samples == 0:
        return np.array([])

    # Find k-nearest neighbors for each minority sample
    knn = NearestNeighbors(n_neighbors=k + 1)
    knn.fit(X)
    _, nn_indices = knn.kneighbors(X)
    
    nn_indices = nn_indices[:, 1:]

    # Generate synthetic samples
    n_synthetic_samples_to_generate = int(N / 100 * n_minority_samples)
    if n_synthetic_samples_to_generate == 0:
        return np.array([])
        
    synthetic_samples = np.zeros((n_synthetic_samples_to_generate, n_features))

    for i in range(n_synthetic_samples_to_generate):
        random_sample_index = np.random.randint(0, n_minority_samples)
        random_sample = X[random_sample_index]

        random_neighbor_index = np.random.choice(nn_indices[random_sample_index])
        random_neighbor = X[random_neighbor_index]

        diff = random_neighbor - random_sample
        gap = np.random.rand()
        synthetic_samples[i] = random_sample + gap * diff

    return synthetic_samples

# --- Main script ---

# File paths
input_data_path = './Question4/train_data.csv'
output_data_path = './Question4/train_data_smote.csv'

# Load data
print(f"加载数据从: {input_data_path}")
data = pd.read_csv(input_data_path)

# Separate features and target
X = data.drop('target', axis=1)
y = data['target']

# Identify minority and majority classes
minority_class_label = y.value_counts().idxmin()
majority_class_label = y.value_counts().idxmax()

X_minority = X[y == minority_class_label].values
X_majority = X[y == majority_class_label].values
y_minority = y[y == minority_class_label]
y_majority = y[y == majority_class_label]

# Calculate the number of samples to generate
n_majority = len(X_majority)
n_minority = len(X_minority)

if n_minority > 0:
    N = int(((n_majority - n_minority) / n_minority) * 100)
    print(f"SMOTE N set to: {N}")

    # Apply SMOTE
    print("应用 SMOTE...")
    synthetic_samples_X = smote(X_minority, N, k=5)

    if synthetic_samples_X.size > 0:
        # Combine original and synthetic data
        X_resampled = np.vstack((X_majority, X_minority, synthetic_samples_X))
        y_resampled = np.hstack((y_majority.values, y_minority.values, [minority_class_label] * len(synthetic_samples_X)))

        resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
        resampled_data['target'] = y_resampled

        # Save the resampled data
        print(f"保存SMOTE处理后的数据到: {output_data_path}")
        resampled_data.to_csv(output_data_path, index=False)

        print("SMOTE 处理完成.")
        print(f"原始样本分布:\n{y.value_counts()}")
        print(f"SMOTE后样本分布:\n{resampled_data['target'].value_counts()}")
    else:
        print("没有生成新的样本，SMOTE未执行。")
else:
    print("少数类样本数量为0，无法进行SMOTE。")