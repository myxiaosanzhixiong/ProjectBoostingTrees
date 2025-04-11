import numpy as np
from collections import Counter


class DecisionTreeRegressor:
    """Simple decision tree regressor used as base learner in GradientBoostingClassifier."""

    def __init__(self, max_depth=3, min_samples_split=2, min_impurity_decrease=0.0, random_state=None):
        """
        Initialize the decision tree regressor.

        Parameters:
        -----------
        max_depth : int, default=3
            Maximum depth of the tree.
        min_samples_split : int, default=2
            Minimum number of samples required to split a node.
        min_impurity_decrease : float, default=0.0
            Minimum decrease in impurity required for a split.
        random_state : int, default=None
            Random state for reproducibility.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.tree = None

    def _mse(self, y):
        """Calculate mean squared error for a set of target values."""
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def _split_node(self, X, y, depth):
        """
        Recursively build the decision tree by splitting nodes.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        depth : int
            Current depth in the tree.

        Returns:
        --------
        dict or float
            A tree node (dict with split information) or a leaf value (float).
        """
        n_samples, n_features = X.shape

        # Base cases: maximum depth reached or not enough samples
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return np.mean(y)

        best_feature, best_threshold = None, None
        best_gain = -float('inf')
        parent_impurity = self._mse(y)

        # If all target values are the same, return that value
        if parent_impurity <= 1e-7:
            return np.mean(y)

        # Try different features and thresholds to find the best split
        rng = np.random.RandomState(self.random_state)
        feature_indices = np.arange(n_features)
        rng.shuffle(feature_indices)

        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                # Split the data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                # Skip if either split is empty
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Calculate impurity for each child
                left_impurity = self._mse(y[left_mask])
                right_impurity = self._mse(y[right_mask])

                # Calculate weighted impurity
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples

                # Calculate impurity gain
                impurity_gain = parent_impurity - weighted_impurity

                # Update best split if this one is better
                if impurity_gain > best_gain and impurity_gain >= self.min_impurity_decrease:
                    best_gain = impurity_gain
                    best_feature = feature_idx
                    best_threshold = threshold

        # If no good split found, return the mean of y
        if best_feature is None:
            return np.mean(y)

        # Create the node with the best split
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Recursively build left and right subtrees
        left_subtree = self._split_node(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._split_node(X[right_mask], y[right_mask], depth + 1)

        # Return the node
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree,
            'value': np.mean(y)  # Store the node's prediction (useful for feature importance)
        }

    def fit(self, X, y):
        """
        Build a decision tree regressor from the training set (X, y).

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns:
        --------
        self : object
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.tree = self._split_node(X, y, depth=0)
        self.feature_importances_ = np.zeros(X.shape[1])
        self._calculate_feature_importance(self.tree, 1.0)

        # Normalize feature importances
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)

        return self

    def _calculate_feature_importance(self, node, weight):
        """
        Calculate feature importance for the tree recursively.

        Parameters:
        -----------
        node : dict or float
            Current tree node.
        weight : float
            Current node weight.
        """
        if isinstance(node, dict):
            # Update feature importance for this node
            self.feature_importances_[node['feature']] += weight

            # Calculate weights for child nodes (could be based on samples if we tracked them)
            left_weight = weight * 0.5
            right_weight = weight * 0.5

            # Recursively calculate for children
            self._calculate_feature_importance(node['left'], left_weight)
            self._calculate_feature_importance(node['right'], right_weight)

    def _predict_single(self, x, node):
        """
        Predict target for a single sample by traversing the tree.

        Parameters:
        -----------
        x : array-like of shape (n_features,)
            Single input sample.
        node : dict or float
            Current tree node.

        Returns:
        --------
        float
            Predicted value.
        """
        if not isinstance(node, dict):
            return node

        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])

    def predict(self, X):
        """
        Predict regression target for X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        --------
        y : array-like of shape (n_samples,)
            The predicted values.
        """
        X = np.asarray(X)
        return np.array([self._predict_single(x, self.tree) for x in X])


class GradientBoostingClassifier:
    """
    Gradient Boosting Classifier implementation based on Sections 10.9-10.10 of
    "The Elements of Statistical Learning" (2nd Edition) by Hastie, Tibshirani, and Friedman.

    This implementation uses decision tree regressors as base learners to predict
    the log-odds of class probabilities.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_impurity_decrease=0.0,
                 subsample=1.0, random_state=None):
        """
        Initialize the Gradient Boosting Classifier.

        Parameters:
        -----------
        n_estimators : int, default=100
            The number of boosting stages (trees) to perform.
        learning_rate : float, default=0.1
            Shrinks the contribution of each tree by learning_rate.
        max_depth : int, default=3
            Maximum depth of the individual regression trees.
        min_samples_split : int, default=2
            Minimum number of samples required to split a node.
        min_impurity_decrease : float, default=0.0
            Minimum decrease in impurity required for a split.
        subsample : float, default=1.0
            Fraction of samples to be used for fitting the individual trees.
        random_state : int, default=None
            Random state for reproducibility.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.subsample = subsample
        self.random_state = random_state

        self.trees = []
        self.feature_importances_ = None
        self.initial_prediction = None

    def _sigmoid(self, x):
        """Apply sigmoid function to input."""
        return 1.0 / (1.0 + np.exp(-x))

    def _negative_gradient(self, y, y_pred):
        """
        Calculate negative gradient for the loss function.

        For binary classification with log loss, this is y - p(y=1|x).

        Parameters:
        -----------
        y : array-like of shape (n_samples,)
            True binary labels.
        y_pred : array-like of shape (n_samples,)
            Predicted probabilities.

        Returns:
        --------
        array-like of shape (n_samples,)
            Negative gradient.
        """
        return y - y_pred

    def fit(self, X, y):
        """
        Build a gradient boosted classifier from the training set (X, y).

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            Target values. Should be binary (0, 1).

        Returns:
        --------
        self : object
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Validate input
        if len(np.unique(y)) != 2:
            raise ValueError("GradientBoostingClassifier only supports binary classification")

        # Ensure y is 0, 1
        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, np.array([0, 1])):
            y_map = {cls: i for i, cls in enumerate(self.classes_)}
            y = np.array([y_map[label] for label in y])

        # Initialize with log-odds of the base rate
        pos_class_rate = np.mean(y)
        # Avoid log(0) or log(1)
        pos_class_rate = np.clip(pos_class_rate, 1e-15, 1 - 1e-15)
        self.initial_prediction = np.log(pos_class_rate / (1 - pos_class_rate))

        # Initialize model with the log-odds of the positive class
        F = np.full(y.shape, self.initial_prediction)

        # Initialize feature importances
        self.feature_importances_ = np.zeros(X.shape[1])

        # Initialize random state
        rng = np.random.RandomState(self.random_state)

        # Boosting iterations
        for i in range(self.n_estimators):
            # Convert log-odds to probabilities
            y_pred = self._sigmoid(F)

            # Calculate negative gradient (residuals)
            residuals = self._negative_gradient(y, y_pred)

            # Subsample the data if subsample < 1.0
            if self.subsample < 1.0:
                n_samples = X.shape[0]
                sample_indices = rng.choice(
                    np.arange(n_samples),
                    size=int(n_samples * self.subsample),
                    replace=False
                )
                X_subset = X[sample_indices]
                residuals_subset = residuals[sample_indices]
            else:
                X_subset = X
                residuals_subset = residuals

            # Fit a regression tree to the negative gradient
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=rng.randint(0, 2 ** 31 - 1)  # Different random state for each tree
            )
            tree.fit(X_subset, residuals_subset)

            # Update feature importances
            self.feature_importances_ += tree.feature_importances_

            # Add the tree to our ensemble
            self.trees.append(tree)

            # Update the model
            update = tree.predict(X)
            F += self.learning_rate * update

        # Normalize feature importances
        if len(self.trees) > 0:
            self.feature_importances_ /= len(self.trees)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        --------
        p : array of shape (n_samples, 2)
            The class probabilities of the input samples.
        """
        X = np.asarray(X)

        # Start with the initial prediction (log-odds)
        F = np.full(X.shape[0], self.initial_prediction)

        # Add contributions from each tree
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)

        # Convert to probabilities using sigmoid
        proba_positive = self._sigmoid(F)

        # Return probabilities for both classes
        return np.vstack([1 - proba_positive, proba_positive]).T

    def predict(self, X):
        """
        Predict class for X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        --------
        y : array of shape (n_samples,)
            The predicted classes.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def feature_importance(self):
        """
        Return feature importances.

        Returns:
        --------
        feature_importances_ : array of shape (n_features,)
            Normalized feature importances.
        """
        return self.feature_importances_