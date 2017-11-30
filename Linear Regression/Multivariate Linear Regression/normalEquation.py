#Normal Equation
import numpy as np

def normalEquation(X, y):
	a = np.linalg.inv((X.T).dot(X))
	b = a.dot(X.T)
	theta = b.dot(y)

	return np.array(theta)
