#Gradient Descent
import numpy as np

def gradientDescent(X, y, theta, alpha, iterations):
	m = len(X)
	X = np.array(X)
	for i in range(iterations):		
		predictions = X.dot(theta)
		temp0 = alpha * sum((predictions - y) * X[:, 0].reshape(m, 1)) / m
		temp1 = alpha * sum((predictions - y) * X[:, 1].reshape(m, 1)) / m
		theta[0] = theta[0] - temp0
		theta[1] = theta[1] - temp1

	return theta
