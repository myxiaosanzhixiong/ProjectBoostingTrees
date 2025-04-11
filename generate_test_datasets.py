import numpy as np
import os
import matplotlib.pyplot as plt


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_linear_dataset(n_samples=1000, n_features=2, noise=0.3, test_size=0.2, random_state=42):
    """Generate a linearly separable dataset."""
    np.random.seed(random_state)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate target (linear decision boundary)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Add noise
    noise_mask = np.random.rand(n_samples) < noise
    y[noise_mask] = 1 - y[noise_mask]

    # Split into train and test
    n_train = int(n_samples * (1 - test_size))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, X_test, y_train, y_test


def generate_circles_dataset(n_samples=1000, noise=0.1, test_size=0.2, random_state=42):
    """Generate a dataset with concentric circles."""
    np.random.seed(random_state)

    # Parameters
    n_samples_per_class = n_samples // 2

    # Inner circle
    radius1 = 0.5
    radius2 = 1.5

    # Generate inner circle
    linspace = np.linspace(0, 2 * np.pi, n_samples_per_class)
    x1_inner = radius1 * np.cos(linspace)
    x2_inner = radius1 * np.sin(linspace)
    X_inner = np.column_stack([x1_inner, x2_inner])
    y_inner = np.zeros(n_samples_per_class)

    # Generate outer circle
    x1_outer = radius2 * np.cos(linspace)
    x2_outer = radius2 * np.sin(linspace)
    X_outer = np.column_stack([x1_outer, x2_outer])
    y_outer = np.ones(n_samples_per_class)

    # Combine
    X = np.vstack([X_inner, X_outer])
    y = np.hstack([y_inner, y_outer])

    # Add noise
    X += noise * np.random.randn(*X.shape)

    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X, y = X[shuffle_idx], y[shuffle_idx]

    # Split into train and test
    n_train = int(n_samples * (1 - test_size))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, X_test, y_train, y_test


def generate_moons_dataset(n_samples=1000, noise=0.1, test_size=0.2, random_state=42):
    """Generate a dataset with two interleaving half circles."""
    np.random.seed(random_state)

    # Parameters
    n_samples_per_class = n_samples // 2

    # First half circle
    t = np.linspace(0, np.pi, n_samples_per_class)
    x1_1 = np.cos(t)
    x2_1 = np.sin(t)

    # Second half circle
    x1_2 = 1 - np.cos(t)
    x2_2 = 1 - np.sin(t) - 0.5

    # Combine both half moons
    x1 = np.hstack([x1_1, x1_2])
    x2 = np.hstack([x2_1, x2_2])

    # Create the dataset
    X = np.column_stack([x1, x2])
    y = np.hstack([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])

    # Add noise
    X += noise * np.random.randn(*X.shape)

    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X, y = X[shuffle_idx], y[shuffle_idx]

    # Split into train and test
    n_train = int(n_samples * (1 - test_size))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, X_test, y_train, y_test


def generate_imbalanced_dataset(n_samples=1000, n_features=2, imbalance_ratio=0.1, noise=0.1, test_size=0.2,
                                random_state=42):
    """Generate an imbalanced binary classification dataset."""
    np.random.seed(random_state)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Determine number of positive samples
    n_positive = int(n_samples * imbalance_ratio)
    n_negative = n_samples - n_positive

    # Generate target (linear decision boundary)
    y = np.zeros(n_samples)
    positive_indices = np.random.choice(n_samples, n_positive, replace=False)
    y[positive_indices] = 1

    # Create a linear decision boundary for the features
    weights = np.random.randn(n_features)
    X[y == 1] += weights  # Shift positive samples

    # Add noise
    X += noise * np.random.randn(*X.shape)

    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X, y = X[shuffle_idx], y[shuffle_idx]

    # Split into train and test
    n_train = int(n_samples * (1 - test_size))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, X_test, y_train, y_test


def generate_noisy_dataset(n_samples=1000, n_features=2, noise_level=0.3, test_size=0.2, random_state=42):
    """Generate a dataset with significant label noise."""
    np.random.seed(random_state)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate target (quadratic decision boundary for more complexity)
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 2).astype(int)

    # Add label noise
    noise_mask = np.random.rand(n_samples) < noise_level
    y[noise_mask] = 1 - y[noise_mask]

    # Split into train and test
    n_train = int(n_samples * (1 - test_size))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, X_test, y_train, y_test


def generate_clustered_dataset(n_samples=1000, n_clusters=3, n_features=2, test_size=0.2, random_state=42):
    """Generate a dataset with multiple clusters per class."""
    np.random.seed(random_state)

    # Number of samples per cluster
    samples_per_cluster = n_samples // (2 * n_clusters)

    X_list = []
    y_list = []

    # Generate clusters for class 0
    for i in range(n_clusters):
        center = np.random.randn(n_features) * 5  # Random cluster center
        cluster_data = center + np.random.randn(samples_per_cluster, n_features)
        X_list.append(cluster_data)
        y_list.append(np.zeros(samples_per_cluster))

    # Generate clusters for class 1
    for i in range(n_clusters):
        center = np.random.randn(n_features) * 5  # Random cluster center
        cluster_data = center + np.random.randn(samples_per_cluster, n_features)
        X_list.append(cluster_data)
        y_list.append(np.ones(samples_per_cluster))

    # Combine all clusters
    X = np.vstack(X_list)
    y = np.hstack(y_list)

    # Shuffle
    shuffle_idx = np.random.permutation(len(y))
    X, y = X[shuffle_idx], y[shuffle_idx]

    # Split into train and test
    n_train = int(len(y) * (1 - test_size))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, X_test, y_train, y_test


def generate_high_dimensional_dataset(n_samples=500, n_features=100, n_informative=10, test_size=0.2, random_state=42):
    """Generate a high-dimensional dataset with only a few informative features."""
    np.random.seed(random_state)

    # Generate features (most are just noise)
    X = np.random.randn(n_samples, n_features)

    # Generate target based on only a few informative features
    informative_features = np.random.choice(n_features, n_informative, replace=False)

    # Create linear combination of informative features
    y = np.zeros(n_samples)
    for feature in informative_features:
        y += X[:, feature]

    # Convert to binary classification
    y = (y > 0).astype(int)

    # Split into train and test
    n_train = int(n_samples * (1 - test_size))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, X_test, y_train, y_test


def generate_correlated_dataset(n_samples=1000, n_features=10, correlation=0.8, test_size=0.2, random_state=42):
    """Generate a dataset with highly correlated features."""
    np.random.seed(random_state)

    # Generate base features
    X_base = np.random.randn(n_samples, n_features // 2)

    # Generate correlated features
    X_correlated = np.zeros((n_samples, n_features // 2))
    for i in range(n_features // 2):
        X_correlated[:, i] = correlation * X_base[:, i] + (1 - correlation) * np.random.randn(n_samples)

    # Combine all features
    X = np.hstack([X_base, X_correlated])

    # Generate target (based on all features)
    y = (np.sum(X, axis=1) > 0).astype(int)

    # Split into train and test
    n_train = int(n_samples * (1 - test_size))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, X_test, y_train, y_test


def generate_step_function_dataset(n_samples=1000, test_size=0.2, random_state=42):
    """Generate a dataset with sharp decision boundaries."""
    np.random.seed(random_state)

    # Generate features
    X = np.random.randn(n_samples, 2) * 3

    # Generate step function target (creating a grid pattern)
    y = ((X[:, 0] > 0) & (X[:, 1] > 0) | (X[:, 0] < 0) & (X[:, 1] < 0)).astype(int)

    # Split into train and test
    n_train = int(n_samples * (1 - test_size))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, X_test, y_train, y_test


def save_dataset(X_train, X_test, y_train, y_test, name, directory='datasets'):
    """Save a dataset to CSV files."""
    ensure_dir(directory)

    # Combine features and target for train set
    train_data = np.column_stack([X_train, y_train])
    test_data = np.column_stack([X_test, y_test])

    # Create header
    n_features = X_train.shape[1]
    header = ','.join([f'feature_{i + 1}' for i in range(n_features)] + ['target'])

    # Save to CSV
    np.savetxt(f'{directory}/{name}_train.csv', train_data, delimiter=',', header=header, comments='')
    np.savetxt(f'{directory}/{name}_test.csv', test_data, delimiter=',', header=header, comments='')


def visualize_dataset(X_train, y_train, name, directory='datasets'):
    """Visualize a 2D dataset and save the plot."""
    ensure_dir(directory)

    # Only visualize if we have 2D data
    if X_train.shape[1] != 2:
        return

    plt.figure(figsize=(8, 6))

    # Plot training data
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, alpha=0.8, edgecolors='k')

    plt.title(f'{name.replace("_", " ").title()} Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Class')
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'{directory}/{name}_visualization.png')
    plt.close()


def main():
    """Generate all test datasets."""
    # Create datasets directory
    datasets_dir = 'datasets'
    ensure_dir(datasets_dir)

    # Generate linear dataset
    print("Generating linear dataset...")
    X_train, X_test, y_train, y_test = generate_linear_dataset()
    save_dataset(X_train, X_test, y_train, y_test, 'linear', datasets_dir)
    visualize_dataset(X_train, y_train, 'linear', datasets_dir)

    # Generate circles dataset
    print("Generating circles dataset...")
    X_train, X_test, y_train, y_test = generate_circles_dataset()
    save_dataset(X_train, X_test, y_train, y_test, 'circles', datasets_dir)
    visualize_dataset(X_train, y_train, 'circles', datasets_dir)

    # Generate moons dataset
    print("Generating moons dataset...")
    X_train, X_test, y_train, y_test = generate_moons_dataset()
    save_dataset(X_train, X_test, y_train, y_test, 'moons', datasets_dir)
    visualize_dataset(X_train, y_train, 'moons', datasets_dir)

    # Generate imbalanced dataset
    print("Generating imbalanced dataset...")
    X_train, X_test, y_train, y_test = generate_imbalanced_dataset()
    save_dataset(X_train, X_test, y_train, y_test, 'imbalanced', datasets_dir)
    visualize_dataset(X_train, y_train, 'imbalanced', datasets_dir)

    # Generate noisy dataset
    print("Generating noisy dataset...")
    X_train, X_test, y_train, y_test = generate_noisy_dataset()
    save_dataset(X_train, X_test, y_train, y_test, 'noisy', datasets_dir)
    visualize_dataset(X_train, y_train, 'noisy', datasets_dir)

    # Generate clustered dataset
    print("Generating clustered dataset...")
    X_train, X_test, y_train, y_test = generate_clustered_dataset()
    save_dataset(X_train, X_test, y_train, y_test, 'clustered', datasets_dir)
    visualize_dataset(X_train, y_train, 'clustered', datasets_dir)

    # Generate high dimensional dataset
    print("Generating high dimensional dataset...")
    X_train, X_test, y_train, y_test = generate_high_dimensional_dataset()
    save_dataset(X_train, X_test, y_train, y_test, 'high_dimensional', datasets_dir)
    # Can't visualize high dimensional data in 2D

    # Generate correlated dataset
    print("Generating correlated dataset...")
    X_train, X_test, y_train, y_test = generate_correlated_dataset()
    save_dataset(X_train, X_test, y_train, y_test, 'correlated', datasets_dir)
    visualize_dataset(X_train[:, :2], y_train, 'correlated_first2dim', datasets_dir)

    # Generate step function dataset
    print("Generating step function dataset...")
    X_train, X_test, y_train, y_test = generate_step_function_dataset()
    save_dataset(X_train, X_test, y_train, y_test, 'step_function', datasets_dir)
    visualize_dataset(X_train, y_train, 'step_function', datasets_dir)

    print(f"All datasets generated and saved to the '{datasets_dir}' directory.")


if __name__ == "__main__":
    main()