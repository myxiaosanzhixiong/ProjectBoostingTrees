import numpy as np
import os
import matplotlib.pyplot as plt
import time

from GradientBoostingClassifier import GradientBoostingClassifier


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_dataset(name, directory='datasets'):
    """Load a dataset from CSV files."""
    # Read data
    train_data = np.loadtxt(f'{directory}/{name}_train.csv', delimiter=',', skiprows=1)
    test_data = np.loadtxt(f'{directory}/{name}_test.csv', delimiter=',', skiprows=1)

    # Separate features and target
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1].astype(int)

    X_test = test_data[:, :-1]
    y_test = test_data[:, -1].astype(int)

    return X_train, X_test, y_train, y_test


def accuracy_score(y_true, y_pred):
    """Calculate accuracy score."""
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, zero_division=0):
    """Calculate precision score."""
    # True positives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    # False positives
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tp + fp == 0:
        return zero_division

    return tp / (tp + fp)


def recall_score(y_true, y_pred, zero_division=0):
    """Calculate recall score."""
    # True positives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    # False negatives
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fn == 0:
        return zero_division

    return tp / (tp + fn)


def f1_score(y_true, y_pred, zero_division=0):
    """Calculate F1 score."""
    prec = precision_score(y_true, y_pred, zero_division)
    rec = recall_score(y_true, y_pred, zero_division)

    if prec + rec == 0:
        return zero_division

    return 2 * (prec * rec) / (prec + rec)


def roc_curve(y_true, y_score):
    """Compute ROC curve."""
    # Sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # Compute ROC curve
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    # Add an extra threshold position to make sure that the curve starts at (0, 0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if tps[-1] <= 0:
        # No positive samples, ROC undefined
        return [np.nan], [np.nan], [np.nan]

    tps = tps / tps[-1]
    fps = fps / fps[-1]

    thresholds = np.r_[y_score[threshold_idxs], y_score[threshold_idxs][-1] + 1]

    return fps, tps, thresholds


def roc_auc_score(y_true, y_score):
    """Compute Area Under the ROC Curve (AUC)."""
    if len(np.unique(y_true)) != 2:
        return np.nan

    fpr, tpr, _ = roc_curve(y_true, y_score)

    # Calculate area using the trapezoidal rule
    width = np.diff(fpr)
    height = (tpr[:-1] + tpr[1:]) / 2

    return np.sum(width * height)


def confusion_matrix(y_true, y_pred, num_classes=2):
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        cm[int(y_true[i]), int(y_pred[i])] += 1
    return cm


def evaluate_model(X_train, X_test, y_train, y_test):
    """Train and evaluate the gradient boosting classifier."""

    # Initialize the model with default parameters
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        random_state=42
    )

    # Measure training time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Measure prediction time
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    prediction_time = time.time() - start_time

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Calculate ROC AUC if we have both classes
    if len(np.unique(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = np.nan

    # Get feature importances
    feature_importances = model.feature_importance()

    return {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'feature_importances': feature_importances,
        'training_time': training_time,
        'prediction_time': prediction_time
    }


def plot_roc_curve(y_test, y_proba, dataset_name, results_dir):
    """Plot ROC curve and save the figure."""
    plt.figure(figsize=(8, 6))

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name.replace("_", " ").title()}')
    plt.legend(loc="lower right")

    # Save the plot
    plt.savefig(f'{results_dir}/{dataset_name}_roc_curve.png')
    plt.close()


def plot_confusion_matrix(y_test, y_pred, dataset_name, results_dir):
    """Plot confusion matrix and save the figure."""
    plt.figure(figsize=(8, 6))

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {dataset_name.replace("_", " ").title()}')
    plt.colorbar()

    # Add labels
    classes = ['Negative (0)', 'Positive (1)']
    tick_marks = [0, 1]
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Save the plot
    plt.savefig(f'{results_dir}/{dataset_name}_confusion_matrix.png')
    plt.close()


def plot_decision_boundary(X, y, model, dataset_name, results_dir):
    """Plot decision boundary for 2D datasets."""
    # Only plot for 2D datasets
    if X.shape[1] != 2:
        return

    plt.figure(figsize=(10, 8))

    # Define the grid
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict on the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    # Plot the data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)

    plt.title(f'Decision Boundary - {dataset_name.replace("_", " ").title()}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter, label='Class')

    # Save the plot
    plt.savefig(f'{results_dir}/{dataset_name}_decision_boundary.png')
    plt.close()


def plot_feature_importance(importances, dataset_name, results_dir):
    """Plot feature importances and save the figure."""
    # Only plot if we have a reasonable number of features
    if len(importances) > 20:
        return

    plt.figure(figsize=(10, 6))

    # Get feature indices sorted by importance
    indices = np.argsort(importances)[::-1]

    # Plot feature importances
    plt.bar(range(len(importances)), importances[indices])
    plt.title(f'Feature Importance - {dataset_name.replace("_", " ").title()}')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.xticks(range(len(importances)), indices)

    # Save the plot
    plt.savefig(f'{results_dir}/{dataset_name}_feature_importance.png')
    plt.close()


def plot_summary_metrics(metrics_data, results_dir):
    """Plot summary metrics across all datasets."""
    plt.figure(figsize=(14, 8))

    # Extract data
    datasets = []
    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_values = []
    roc_auc_values = []

    for result in metrics_data:
        datasets.append(result['dataset'])
        accuracy_values.append(result['accuracy'])
        precision_values.append(result['precision'])
        recall_values.append(result['recall'])
        f1_values.append(result['f1'])
        roc_auc_values.append(result['roc_auc'])

    # Set up bar positions
    x = np.arange(len(datasets))
    width = 0.15

    # Plot each metric as a group of bars
    plt.bar(x - 2 * width, accuracy_values, width, label='Accuracy')
    plt.bar(x - width, precision_values, width, label='Precision')
    plt.bar(x, recall_values, width, label='Recall')
    plt.bar(x + width, f1_values, width, label='F1')
    plt.bar(x + 2 * width, roc_auc_values, width, label='ROC AUC')

    # Add labels and legend
    plt.xlabel('Dataset')
    plt.ylabel('Score')
    plt.title('Summary Metrics Across Datasets')
    plt.xticks(x, [name.replace('_', ' ').title() for name in datasets], rotation=45, ha='right')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)

    plt.tight_layout()
    plt.ylim(0, 1.1)

    # Save the plot
    plt.savefig(f'{results_dir}/summary_metrics.png')
    plt.close()


def save_results_to_csv(all_results, results_dir):
    """Save metrics to CSV file."""
    with open(f'{results_dir}/evaluation_results.csv', 'w', newline='') as f:
        # Write header
        f.write('dataset,accuracy,precision,recall,f1,roc_auc,training_time,prediction_time\n')

        # Write data
        for result in all_results:
            f.write(f"{result['dataset']},{result['accuracy']:.4f},{result['precision']:.4f},"
                    f"{result['recall']:.4f},{result['f1']:.4f},{result['roc_auc']:.4f},"
                    f"{result['training_time']:.4f},{result['prediction_time']:.4f}\n")


def main():
    # Create results directory
    results_dir = 'results'
    ensure_dir(results_dir)

    # List of dataset names
    dataset_names = [
        'linear',
        'circles',
        'moons',
        'imbalanced',
        'noisy',
        'clustered',
        'high_dimensional',
        'correlated',
        'step_function'
    ]

    # Collect results for all datasets
    all_results = []

    for dataset_name in dataset_names:
        print(f"Evaluating on {dataset_name} dataset...")

        # Load dataset
        try:
            X_train, X_test, y_train, y_test = load_dataset(dataset_name)
        except FileNotFoundError:
            print(f"Dataset {dataset_name} not found. Skipping.")
            continue

        # Evaluate model
        results = evaluate_model(X_train, X_test, y_train, y_test)

        # Save results
        all_results.append({
            'dataset': dataset_name,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'roc_auc': results['roc_auc'],
            'training_time': results['training_time'],
            'prediction_time': results['prediction_time']
        })

        # Create visualizations
        plot_roc_curve(y_test, results['y_proba'], dataset_name, results_dir)
        plot_confusion_matrix(y_test, results['y_pred'], dataset_name, results_dir)
        plot_decision_boundary(X_test, y_test, results['model'], dataset_name, results_dir)
        plot_feature_importance(results['feature_importances'], dataset_name, results_dir)

    # Save metrics to CSV
    save_results_to_csv(all_results, results_dir)

    # Plot summary metrics
    plot_summary_metrics(all_results, results_dir)

    print(f"Evaluation complete. Results saved to the '{results_dir}' directory.")


if __name__ == "__main__":
    main()