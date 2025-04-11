# CS584
Illinois Institute of Technology

Name:Jian Zhang
CWID:A20467790

Name:Hisham Mohammed 
Cwid:A20584812

Name: Manthan Surjuse
Cwid: A20588887


# Gradient Boosting Classifier Implementation

This is an implementation of the Gradient Boosting Tree Classification algorithm as described in Sections 10.9-10.10 of "The Elements of Statistical Learning" (2nd Edition) by Hastie, Tibshirani, and Friedman.

## Project Structure

```
gradient-boosting/
├── GradientBoostingClassifier.py  # Main implementation file
├── test_GradientBoostingClassifier.py  # Unit tests
├── generate_test_datasets.py  # Script to generate test datasets
├── evaluate_on_datasets.py  # Script to evaluate model on test datasets
├── datasets/  # Directory containing generated test datasets
│   ├── linear_train.csv
│   ├── linear_test.csv
│   ├── circles_train.csv
│   ├── circles_test.csv
│   └── ...
├── results/  # Directory containing evaluation results
│   ├── evaluation_results.csv
│   ├── summary_metrics.png
│   └── ...
└── README.md  # This file
```

## Installation

The implementation requires only NumPy as a dependency for the core algorithm:

```bash
pip install numpy
```

For running the tests, you'll also need:

```bash
pip install scikit-learn pytest
```

## Usage Example

```python
from GradientBoostingClassifier import GradientBoostingClassifier
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(
    n_samples=1000, 
    n_features=10, 
    n_informative=5,
    random_state=42
)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the model
gbm = GradientBoostingClassifier(
    n_estimators=100,  # Number of trees
    learning_rate=0.1,  # Shrinkage parameter
    max_depth=3,       # Maximum depth of each tree
    subsample=0.8,     # Fraction of samples to use for fitting trees
    random_state=42    # For reproducibility
)

gbm.fit(X_train, y_train)

# Make predictions
y_pred = gbm.predict(X_test)
probabilities = gbm.predict_proba(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Examine feature importances
importances = gbm.feature_importance()
for i, importance in enumerate(importances):
    print(f"Feature {i}: {importance:.4f}")
```

## Implementation Details

The implementation consists of two main classes:

1. `DecisionTreeRegressor`: A simple regression tree used as the weak learner in the gradient boosting ensemble.
2. `GradientBoostingClassifier`: The main class that implements the gradient boosting algorithm for binary classification.

The algorithm follows these steps:
1. Initialize the model with a constant value (log-odds of the positive class)
2. For each boosting iteration:
   - Calculate the negative gradient (residuals)
   - Fit a regression tree to the negative gradient
   - Update the model by adding the tree's predictions (scaled by the learning rate)
3. Convert final log-odds to probabilities using the sigmoid function

This implementation currently supports only binary classification problems.

## Questions

### What does the model you have implemented do and when should it be used?

The Gradient Boosting Classifier is an ensemble learning method that combines multiple weak prediction models (decision trees) to create a strong classifier. It works by sequentially adding trees that focus on correcting the errors made by previous trees.

**When to use it:**

1. **Complex, non-linear relationships**: Gradient boosting excels at capturing complex relationships between features and target variables without requiring feature engineering.

2. **Tabular data**: It's particularly effective on structured/tabular data, where it often outperforms other algorithms.

3. **Imbalanced datasets**: With proper tuning, gradient boosting can handle class imbalance well.

4. **When feature importance is needed**: The algorithm provides a clear understanding of which features contribute most to predictions.

5. **When prediction accuracy is the priority**: Gradient boosting typically achieves high accuracy compared to simpler algorithms.

It may not be the best choice when:
- Extremely fast inference is required (as it requires evaluating multiple trees)
- Training data is very limited (may overfit)
- The problem can be solved with a simpler, more interpretable model

### How did you test your model to determine if it is working reasonably correctly?

I implemented a comprehensive testing strategy to ensure the model works correctly:

1. **Basic functionality tests**:
   - Verified that model initialization respects all parameters
   - Checked that fit and predict methods work without errors
   - Ensured predict_proba outputs valid probabilities that sum to 1

2. **Performance benchmarks**:
   - Tested on standard datasets (like make_classification and make_hastie_10_2)
   - Verified that the model achieves reasonable accuracy (>70%)
   - Calculated AUC-ROC to ensure good discrimination ability

3. **Feature importance validation**:
   - Verified that feature importances are calculated
   - Checked that importances align with known informative features

4. **Subsampling verification**:
   - Tested that models with subsampling can still achieve good performance
   - Compared subsampled models to full-data models

5. **Learning rate effects**:
   - Tested different learning rates to ensure they impact model training
   - Verified that the model works with both high and low learning rates

6. **Edge cases**:
   - Tested on very small datasets
   - Verified proper error handling for multiclass data

7. **Reproducibility**:
   - Confirmed that models with the same random seed produce identical results
   - Verified that different random seeds produce different models

These tests ensure that the implementation not only works correctly but also follows the expected behavior described in the literature.

### What parameters have you exposed to users of your implementation in order to tune performance?

My implementation exposes several key parameters for tuning model performance:

1. **n_estimators** (default=100):  
   The number of boosting stages (trees) to perform. More trees typically improve performance until a point of diminishing returns. Increasing this parameter may improve model performance at the cost of training time.

   ```python
   # More trees for potentially better performance
   gbm = GradientBoostingClassifier(n_estimators=500)
   ```

2. **learning_rate** (default=0.1):  
   Shrinks the contribution of each tree. There is a trade-off between learning_rate and n_estimators. Lower learning rates require more trees but often lead to better test accuracy.

   ```python
   # Lower learning rate often gives better generalization
   gbm = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000)
   ```

3. **max_depth** (default=3):  
   Maximum depth of individual regression trees. Deeper trees can model more complex relationships but are more prone to overfitting.

   ```python
   # Simpler trees to avoid overfitting
   gbm = GradientBoostingClassifier(max_depth=2)
   ```

4. **min_samples_split** (default=2):  
   The minimum number of samples required to split a node. Higher values prevent creating nodes with too few samples, which helps prevent overfitting.

   ```python
   # Require more samples in a node to split
   gbm = GradientBoostingClassifier(min_samples_split=10)
   ```

5. **min_impurity_decrease** (default=0.0):  
   A split will only happen if it decreases the impurity by at least this amount. This helps control tree growth and prevent splits that don't significantly improve the model.

   ```python
   # Only make meaningful splits
   gbm = GradientBoostingClassifier(min_impurity_decrease=0.01)
   ```

6. **subsample** (default=1.0):  
   The fraction of samples used for fitting each base learner. Values less than 1.0 introduce randomness (stochastic gradient boosting), which can improve generalization.

   ```python
   # Stochastic gradient boosting
   gbm = GradientBoostingClassifier(subsample=0.8)
   ```

7. **random_state** (default=None):  
   Controls the randomness in the boosting process. Setting this parameter ensures reproducible results.

   ```python
   # For reproducible results
   gbm = GradientBoostingClassifier(random_state=42)
   ```

Example of parameter tuning:

```python
# Model tuned for better generalization
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_samples_split=5,
    subsample=0.8,
    random_state=42
)
```

### Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

The current implementation has several limitations:

1. **Multiclass Classification**: 
   - **Issue**: Currently only supports binary classification.
   - **Potential Solution**: Implement one-vs-rest approach or the multiclass gradient boosting algorithm with softmax loss. This is feasible with additional time and would require modifying gradient calculation and prediction functions.

2. **Handling Large Datasets**:
   - **Issue**: The implementation loads all data into memory, which can be problematic for very large datasets.
   - **Potential Solution**: Implement out-of-core learning or streaming approaches. This would be a significant architectural change but is feasible.

3. **Missing Values**:
   - **Issue**: The current implementation doesn't handle missing values natively.
   - **Potential Solution**: Add logic to handle missing values during tree building, similar to how XGBoost and other libraries do. This is a feasible enhancement that would require modifications to the decision tree algorithm.

4. **Categorical Features**:
   - **Issue**: Does not natively handle categorical features.
   - **Potential Solution**: Implement one-hot encoding internally or special treatment for categorical variables in the splitting algorithm. This is a feasible enhancement.

5. **Memory Efficiency**:
   - **Issue**: Storing all trees can consume significant memory for large numbers of estimators.
   - **Potential Solution**: Implement pruning or tree compression techniques. This is a feasible enhancement that would require additional logic after tree building.

6. **Scalability**:
   - **Issue**: Not optimized for parallel processing.
   - **Potential Solution**: Add parallel tree building using multiprocessing or thread pools. This would be a moderate enhancement requiring some refactoring.

7. **Optimization Techniques**:
   - **Issue**: The implementation uses a basic decision tree building algorithm without advanced optimizations.
   - **Potential Solution**: Add histogram-based splits, feature bunching, or other optimization techniques used in libraries like LightGBM. These would require significant algorithm revisions but are technically feasible.

None of these limitations are fundamental to the gradient boosting algorithm itself - they are implementation constraints that could be addressed with additional development time. The most important next step would be adding multiclass classification support and handling of missing values, as these would greatly increase the usability of the implementation in real-world scenarios.

## Test Datasets

The implementation comes with a script to generate various synthetic datasets to thoroughly test the model's performance across different data distributions and challenges:

1. **Linear Data**: Linearly separable data with clear decision boundaries
2. **Circles**: Non-linear data in concentric circles, requiring non-linear decision boundaries
3. **Moons**: Another non-linear dataset with two interleaving half circles
4. **Imbalanced**: Dataset with class imbalance (90% one class, 10% the other)
5. **Noisy**: Dataset with significant label noise to test robustness
6. **Clustered**: Dataset with multiple clusters per class
7. **High Dimensional**: Dataset with 100 features to test performance on high-dimensional data
8. **Correlated Features**: Dataset with highly correlated features to test feature selection
9. **Step Function**: Dataset with sharp decision boundaries

To generate these datasets:

```bash
python generate_test_datasets.py
```

This will create CSV files in the `datasets` directory and visualizations for 2D datasets.

## Evaluation

To evaluate the model on all generated datasets:

```bash
python evaluate_on_datasets.py
```

This will:
1. Train and test the model on each dataset
2. Generate performance metrics (accuracy, precision, recall, F1, ROC AUC)
3. Create visualizations including ROC curves, confusion matrices, and decision boundaries
4. Save all results to the `results` directory

## Running Tests

To run the unit tests:

```bash
python -m pytest test_GradientBoostingClassifier.py -v
```

Or simply:

```bash
python test_GradientBoostingClassifier.py
```
