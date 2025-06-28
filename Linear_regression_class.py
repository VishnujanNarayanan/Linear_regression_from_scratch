import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
        self.losses = []

    def fit(self, X, y):
        """Train the model using gradient descent"""
        m = len(X)
        X_b = np.c_[np.ones((m, 1)), X]
        self.X_b = X_b
        self.y = y
        self.theta = np.random.randn(X_b.shape[1], 1)

        for _ in range(self.n_iterations):
            predictions = X_b.dot(self.theta)
            error = predictions - y
            gradients = 2/m * X_b.T.dot(error)
            self.theta -= self.learning_rate * gradients
            self.losses.append(np.mean(error ** 2))

    def predict(self, X):
        """Make predictions using the learned theta"""
        X_b = np.c_[np.ones((len(X), 1)), X]
        return X_b.dot(self.theta)

    def plot_loss(self):
        """Plot MSE loss over iterations"""
        plt.plot(self.losses)
        plt.title("Loss Curve")
        plt.xlabel("Iterations")
        plt.ylabel("MSE Loss")
        plt.grid(True)
        plt.show()

    def plot_predictions(self):
        """Plot predicted vs actual values"""
        y_pred = self.predict(self.X_b[:, 1:])  # Remove bias
        plt.scatter(self.y, y_pred, alpha=0.6)
        plt.xlabel("Actual y")
        plt.ylabel("Predicted y")
        plt.title("Predicted vs Actual")
        plt.plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], "r--")
        plt.grid(True)
        plt.show()

    def plot_coefficients(self):
        """Plot learned coefficients for each feature"""
        feature_names = [f"x{i+1}" for i in range(self.X_b.shape[1] - 1)]
        coefs = self.theta[1:].flatten()
        plt.bar(feature_names, coefs)
        plt.title("Learned Feature Coefficients")
        plt.ylabel("Weight")
        plt.grid(True)
        plt.show()

    def evaluate(self, X, y_true):
        """Compute MSE and R²"""
        y_pred = self.predict(X)
        mse = np.mean((y_true - y_pred) ** 2)
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - ss_res / ss_total
        print(f"📏 MSE: {mse:.3f}")
        print(f"🔢 R² Score: {r2:.3f}")
        return mse, r2
