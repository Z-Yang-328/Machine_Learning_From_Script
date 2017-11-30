# Compute cost function of linear regression
import numpy as np

def computeCost(X, y, theta):
	m = len(X)
	X = np.array(X)
	predictions = X.dot(theta)
	mse = (predictions - y) ** 2
	J = float(sum(mse) / (2 * m))
	
	return J
