import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 1, 100).reshape(-1, 1)
y = 0.5 * X.ravel() + 0.3 * np.sin(15 * X.ravel()) + np.random.normal(0, 0.1, X.shape[0])

# High bias model (underfitting)
linear_model = LinearRegression()
linear_model.fit(X, y)

# High variance model (overfitting)
tree_model = DecisionTreeRegressor(max_depth=20)
tree_model.fit(X, y)

print("Linear Model (High Bias):", linear_model.score(X, y))
print("Deep Tree (High Variance):", tree_model.score(X, y))
