import unittest
import numpy as np
from GradientBoostingClassifier import GradientBoostingClassifier


class TestGradientBoostingClassifier(unittest.TestCase):

    def setUp(self):
        # Set random seed for reproducibility
        np.random.seed(42)

        # Create a simple binary classification dataset
        n_samples = 100
        n_features = 5

        # Generate features
        X = np.random.randn(n_samples, n_features)

        # Generate target based on a simple rule (first two features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        self.X = X
        self.y = y

        # Create a more complex dataset: two half moons
        # This is a simplified manual implementation of half-moons
        n_samples_complex = 200

        # First half moon
        t = np.linspace(0, np.pi, n_samples_complex // 2)
        x1_1 = np.cos(t)
        x2_1 = np.sin(t)

        # Second half moon
        x1_2 = 1 - np.cos(t)
        x2_2 = 1 - np.sin(t) - 0.5

        # Combine both half moons
        x1 = np.hstack([x1_1, x1_2])
        x2 = np.hstack([x2_1, x2_2])

        # Add noise
        x1 += 0.1 * np.random.randn(n_samples_complex)
        x2 += 0.1 * np.random.randn(n_samples_complex)

        # Create the complex dataset
        X_complex = np.column_stack([x1, x2])
        y_complex = np.hstack([np.zeros(n_samples_complex // 2), np.ones(n_samples_complex // 2)])

        self.X_complex = X_complex
        self.y_complex = y_complex

    def test_initialization(self):
        """Test that model initialization works with various parameters."""
        model = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=2,
            min_samples_split=5,
            subsample=0.8,
            random_state=42
        )

        self.assertEqual(model.n_estimators, 50)
        self.assertEqual(model.learning_rate, 0.05)
        self.assertEqual(model.max_depth, 2)
        self.assertEqual(model.min_samples_split, 5)
        self.assertEqual(model.subsample, 0.8)
        self.assertEqual(model.random_state, 42)

    def test_fit_predict(self):
        """Test that fit and predict methods work."""
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)

        y_pred = model.predict(self.X)
        self.assertEqual(len(y_pred), len(self.y))
        self.assertTrue(set(np.unique(y_pred)).issubset({0, 1}))

    def test_predict_proba(self):
        """Test that predict_proba returns valid probabilities."""
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)

        probas = model.predict_proba(self.X)

        # Check shape
        self.assertEqual(probas.shape, (len(self.y), 2))

        # Check that probabilities sum to 1
        row_sums = np.sum(probas, axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0))

        # Check that all values are between 0 and 1
        self.assertTrue(np.all(probas >= 0))
        self.assertTrue(np.all(probas <= 1))

    def test_feature_importance(self):
        """Test that feature importances are calculated."""
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)

        importances = model.feature_importance()

        # Check shape
        self.assertEqual(len(importances), self.X.shape[1])

        # Check that importances sum to 1
        self.assertAlmostEqual(np.sum(importances), 1.0, places=5)

        # Since we know the target depends on first two features,
        # check that they have higher importance
        self.assertTrue(importances[0] + importances[1] > 0.5)

    def test_learning_rate(self):
        """Test that learning rate affects model performance."""
        # Train two models with different learning rates
        model1 = GradientBoostingClassifier(
            n_estimators=10, learning_rate=0.1, random_state=42)
        model2 = GradientBoostingClassifier(
            n_estimators=10, learning_rate=1.0, random_state=42)

        model1.fit(self.X, self.y)
        model2.fit(self.X, self.y)

        # Predictions should be different
        y_pred1 = model1.predict(self.X)
        y_pred2 = model2.predict(self.X)

        self.assertFalse(np.array_equal(y_pred1, y_pred2))

    def test_subsample(self):
        """Test that subsampling works."""
        # Train two models, one with subsampling, one without
        model_full = GradientBoostingClassifier(
            n_estimators=10, subsample=1.0, random_state=42)
        model_sub = GradientBoostingClassifier(
            n_estimators=10, subsample=0.5, random_state=42)

        model_full.fit(self.X, self.y)
        model_sub.fit(self.X, self.y)

        # Both should achieve reasonable performance
        y_pred_full = model_full.predict(self.X)
        y_pred_sub = model_sub.predict(self.X)

        accuracy_full = np.mean(y_pred_full == self.y)
        accuracy_sub = np.mean(y_pred_sub == self.y)

        self.assertTrue(accuracy_full > 0.7)
        self.assertTrue(accuracy_sub > 0.7)

    def test_reproducibility(self):
        """Test that the same random seed gives the same results."""
        model1 = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model2 = GradientBoostingClassifier(n_estimators=10, random_state=42)

        model1.fit(self.X, self.y)
        model2.fit(self.X, self.y)

        y_pred1 = model1.predict(self.X)
        y_pred2 = model2.predict(self.X)

        # Predictions should be identical
        self.assertTrue(np.array_equal(y_pred1, y_pred2))

    def test_complex_dataset(self):
        """Test that the model can handle a more complex dataset."""
        model = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

        model.fit(self.X_complex, self.y_complex)

        y_pred = model.predict(self.X_complex)
        accuracy = np.mean(y_pred == self.y_complex)

        # Should achieve decent accuracy on this harder dataset
        self.assertTrue(accuracy > 0.8)

    def test_error_handling(self):
        """Test error handling for non-binary data."""
        X = np.random.randn(50, 3)
        y = np.array([0, 1, 2] * (50 // 3) + [0] * (50 % 3))  # Three classes

        model = GradientBoostingClassifier(n_estimators=10)

        with self.assertRaises(ValueError):
            model.fit(X, y)


if __name__ == '__main__':
    unittest.main()