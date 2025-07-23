from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
digits = datasets.load_digits()
plt.gray()
plt.matshow(digits.images[0])
plt.title(f"Digit Label: {digits.target[0]}")
plt.show()
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))  
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")
predicted = knn.predict(X_test[:10])
print("Predicted:", predicted)
print("Actual:   ", y_test[:10])
for i in range(10):
    plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
    plt.title(f"Predicted: {predicted[i]} | Actual: {y_test[i]}")
    plt.show()
