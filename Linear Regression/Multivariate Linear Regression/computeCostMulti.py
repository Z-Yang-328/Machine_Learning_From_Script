#Compute cost function

def computeCostMulti(X, y, theta):
	m = len(X)
	predictions = X.dot(theta)
	mse = (predictions - y) ** 2
	J = 1/(2*m) * sum(mse)
	return J
