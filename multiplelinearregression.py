import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_samples = 100
X1 = np.random.rand(num_samples, 1) * 10 

X2 = np.random.rand(num_samples, 1) * 5   
y = 2 * X1 + 3 * X2 + np.random.randn(num_samples, 1) * 2 

X = np.concatenate((X1, X2), axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X_test[:, 0], X_test[:, 1], y_test, c='b', marker='o', label='Actual Data Points')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_zlabel('Target Variable')
ax1.set_title('Actual Data Points')
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_test[:, 0], X_test[:, 1], y_pred, c='r', marker='o', label='Predicted Data Points')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.set_zlabel('Target Variable')
ax2.set_title('Predicted Data Points')

plt.tight_layout()
plt.show()
