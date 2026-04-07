import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# Linear regression Model
# from sklearn.linear_model import LinearRegression

from regression_data import X_train, y_train, X_test, y_test
import numpy as np
# Test different model complexities
k_values = range(1, 21)
train_scores = []
test_scores = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)

    
    train_score = knn.score(X_train, y_train)
    pred = knn.predict(X_test)
    test_score = knn.score(X_test, y_test)
    print ("K:", k, "Train Score:", train_score, "Test Score:", test_score, "Predictions:", pred)
    
    train_scores.append(train_score)
    test_scores.append(test_score)




# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_scores, 'o-', label='Training Score')
plt.plot(k_values, test_scores, 'o-', label='Test Score')
plt.xlabel('Model Complexity (Lower K = Higher Complexity)')
plt.ylabel('R² Score')
plt.title('Model Complexity vs Performance')
plt.legend()
plt.grid(True)
plt.show()


# Find optimal complexity
optimal_k = k_values[np.argmax(test_scores)]
print(f"Optimal K (best generalization): {optimal_k}")
