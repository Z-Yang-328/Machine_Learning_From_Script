#Gradient Descent
from computeCostMulti import computeCostMulti

def gradientDescentMulti(X, y, theta, alpha, iterations):
	J_history = [] 
	for iters in range(iterations):	
		predictions = X.dot(theta)
		se = (predictions - y)
		m = len(X)
		temp = [0, 0, 0]
		for i in range(len(theta)):
			temp[i] = alpha / m * sum(se * X[:,i].reshape(m, 1))
		theta = theta - temp

		J = computeCostMulti(X, y, theta)
		J_history.append(J)
	
	return theta, J_history
