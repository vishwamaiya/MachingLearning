from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import numpy as np

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Create model
model = KNeighborsClassifier()

# 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=10)

print("Cross-validation scores:", cv_scores)
print(f"Mean accuracy: {cv_scores.mean():.3f}")
print(f"Standard deviation: {cv_scores.std():.3f}")
print(f"95% confidence interval: {cv_scores.mean():.3f} ± {1.96 * cv_scores.std():.3f}")

# More detailed cross-validation
cv_results = cross_validate(model, X, y, cv=10, 
                           scoring=['accuracy', 'precision_macro', 'recall_macro'])
print("\nDetailed results:")
for metric, scores in cv_results.items():
    if metric.startswith('test_'):
        print(f"{metric}: {scores.mean():.3f} ± {scores.std():.3f}")
