from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


digits = load_digits()
X, y = digits.data, digits.target


fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"Label: {digits.target[i]}")
    ax.axis('off')
plt.show()


#X = X / 16.0  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy on Digits: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
