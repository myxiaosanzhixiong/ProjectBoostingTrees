# **CS584**
## Illinois Institute of Technology
- Name: Jian Zhang  CWID: A20467790
- Name: Hisham Mohammed  CWID: A20584812
- Name: Manthan Surjuse  CWID: A20588887

# Gradient Boosting Classifier Implementation

This project implements a Gradient Boosting Classifier from first principles, following the approach described in Sections 10.9-10.10 of "The Elements of Statistical Learning" (2nd Edition) by Hastie, Tibshirani, and Friedman. The implementation includes a complete binary classification algorithm with decision trees as base learners.

## Overview

Gradient Boosting is a powerful machine learning technique that builds an ensemble of weak prediction models (typically decision trees) to create a strong predictive model. The algorithm works by iteratively adding models that correct the errors made by previous models in the ensemble.

This implementation focuses on binary classification and includes:

- A custom `DecisionTreeRegressor` as the base learner
- A `GradientBoostingClassifier` that uses these trees to predict class probabilities
- Comprehensive testing and evaluation on various synthetic datasets

## Algorithm Details

The implemented gradient boosting algorithm:

1. Initializes prediction with log-odds of the base rate
2. For each boosting iteration:
   - Computes the negative gradient of the loss function (residuals)
   - Fits a regression tree to these residuals
   - Updates the model with a scaled version of the tree's predictions
   - Tracks feature importance across all trees

## Features

- **Binary Classification**: Predicts probabilities for binary outcomes
- **Customizable Parameters**: Control model complexity with parameters like max_depth, learning_rate, and n_estimators
- **Feature Importance**: Calculates importance scores for each feature
- **Stochastic Gradient Boosting**: Supports subsampling for improved generalization
- **Comprehensive Testing**: Unit tests and evaluation on diverse datasets

## Usage

### Basic Example

```python
from GradientBoostingClassifier import GradientBoostingClassifier
import numpy as np

# Create synthetic data
X_train = np.random.randn(100, 5)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
X_test = np.random.randn(20, 5)

# Initialize and train the model
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Get feature importances
importances = model.feature_importance()
```

### Parameters

The `GradientBoostingClassifier` accepts the following parameters:

- `n_estimators` (default=100): Number of boosting stages (trees) to perform
- `learning_rate` (default=0.1): Shrinks the contribution of each tree
- `max_depth` (default=3): Maximum depth of the individual regression trees
- `min_samples_split` (default=2): Minimum samples required to split a node
- `min_impurity_decrease` (default=0.0): Minimum impurity decrease required for splitting
- `subsample` (default=1.0): Fraction of samples used for fitting each tree
- `random_state` (default=None): Controls randomness for reproducibility

## Test Datasets

The project includes a script to generate various synthetic datasets to evaluate the classifier:

- **Linear**: Linearly separable data with noise
- **Circles**: Concentric circles for testing non-linear decision boundaries
- **Moons**: Two interleaving half circles
- **Imbalanced**: Dataset with class imbalance
- **Noisy**: Dataset with significant label noise
- **Clustered**: Multiple clusters per class
- **High Dimensional**: Many features but few informative ones
- **Correlated**: Features with high correlation
- **Step Function**: Sharp decision boundaries

## Evaluation

The evaluation framework calculates the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

It also generates visualizations:

- ROC curves
- Confusion matrices
- Decision boundaries (for 2D datasets)
- Feature importance plots

## Running Tests and Evaluation

To run the unit tests:

```bash
python test_GradientBoostingClassifier.py
```

To generate test datasets:

```bash
python generate_test_datasets.py
```

To evaluate the model on all datasets:

```bash
python evaluate_on_datasets.py
```

## Questions from the Project Requirements

### What does the model you have implemented do and when should it be used?

The implemented model is a Gradient Boosting Classifier that is suited for binary classification problems. It works by building an ensemble of decision trees sequentially, where each tree corrects the errors of the previous ones. The model is particularly effective in:

- Handling complex non-linear relationships between features and target
- Working with mixed data types after appropriate encoding
- Capturing feature interactions
- Providing feature importance metrics

It should be used when:
- You have a binary classification problem
- You need interpretable results (through feature importance)
- You're dealing with complex, non-linear relationships
- You have sufficient training data
- Prediction speed at inference time is not the primary concern

### How did you test your model to determine if it is working reasonably correctly?

The model was tested through multiple approaches:

1. **Unit Tests**: Testing individual components and the whole model:
   - Initialization with different parameters
   - Correctness of fit and predict methods
   - Probability calibration
   - Feature importance calculation
   - Reproducibility with fixed random seeds
   - Performance on complex datasets
   - Proper error handling

2. **Synthetic Datasets**: Evaluating performance on diverse synthetic datasets to test the model's ability to handle:
   - Linear and non-linear decision boundaries
   - Class imbalance
   - Noisy labels
   - Clustered data
   - High dimensionality
   - Correlated features
   - Sharp decision boundaries

3. **Metrics Analysis**: Calculating and analyzing performance metrics such as accuracy, precision, recall, F1 score, and ROC AUC to ensure the model performs well across different scenarios.

### What parameters have you exposed to users of your implementation in order to tune performance?

The implementation exposes several parameters to tune performance:

- **n_estimators**: Controls the number of boosting stages. Higher values often lead to better performance but increase training time and risk of overfitting.
- **learning_rate**: Controls the contribution of each tree to the final outcome. Lower values require more trees but can lead to better generalization.
- **max_depth**: Limits the complexity of individual trees. Lower values reduce overfitting.
- **min_samples_split**: Controls the minimum number of samples required to split a node, preventing splits with too few samples.
- **min_impurity_decrease**: Sets a threshold for the minimum impurity decrease required for a split.
- **subsample**: Allows for stochastic gradient boosting by using only a fraction of the data for each tree, which can improve generalization.
- **random_state**: Ensures reproducibility with the same initialization.

### Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

The implementation has some limitations:

1. **Multi-class Classification**: The current implementation only supports binary classification. With more time, this could be extended using strategies like one-vs-rest or using multi-class loss functions.

2. **Categorical Features**: The implementation expects numerical features. Categorical features would need to be one-hot encoded before use. A more sophisticated version could include native handling of categorical features.

3. **Large Datasets**: The current implementation loads all data into memory and may struggle with very large datasets. This could be addressed with batch processing or out-of-core learning.

4. **Missing Values**: The implementation doesn't handle missing values natively. This could be addressed by implementing missing value splits like those in XGBoost.

5. **Regularization**: While the implementation uses depth and subsampling for regularization, it lacks L1/L2 regularization. This could be added to improve performance on high-dimensional data.

These limitations are not fundamental to gradient boosting algorithms and could be addressed with additional development effort to enhance the implementation.

