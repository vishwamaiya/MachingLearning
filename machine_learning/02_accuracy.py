from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Load sample data
iris = load_iris()
X, y = iris.data, iris.target

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=5
)

# Train model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# Test model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")
