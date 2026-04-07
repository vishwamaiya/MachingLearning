from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
import pandas as pd

# Load dataset
wine = load_wine()
X, y = wine.data, wine.target

# Step 1: Initial train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=65, stratify=y
)

# Step 2: Model training
model = RandomForestClassifier(random_state=65)
model.fit(X_train, y_train)

# Step 3: Cross-validation on training data
cv_scores = cross_val_score(model, X_train, y_train, cv=20)
print(f"CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Step 4: Final evaluation on test set
y_pred = model.predict(X_test)
print("\nFinal Test Results:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# Step 5: Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
