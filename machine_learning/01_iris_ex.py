from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target
#print ("Features: ",X)
#print ("Lbels:: ",len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=30)
"""
print ("X_train: ", X_train)
print ("y_train: ", y_train)
print ("X_test: ", X_test)
print ("y_test: ",y_test)

print (f"Count: X_Train: {len(X_train)}, y_train: {len(y_train)}")
"""
# Training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Testing
predictions = model.predict(X_test)
print ("Predictions: ", predictions)
print("actual values: ", y_test)

# New data is the specification of 151st flower
new_data = [ [3,5,5,2] ]
pred = model.predict(new_data)
print ("New data pred: ",pred)

