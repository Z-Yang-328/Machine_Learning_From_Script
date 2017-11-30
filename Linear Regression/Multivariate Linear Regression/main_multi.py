import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from featureNormalize import featureNormalize
from computeCostMulti import computeCostMulti
from gradientDescentMulti import gradientDescentMulti
from normalEquation import normalEquation

##================ Part 1: Feature Normalization ================##

data = np.array(pd.read_table('ex1data2.txt', delimiter=','))
X = data[:,:-1]
y = data[:,-1]
m = len(X)

print('First 10 examples from the dataset: \n');
print(' x = \n',X[0:9,:],',\n y = \n',(y[0:9]));

input('Program paused. Press enter to continue.\n');

print('Normalize feature...\n')

X, mu, sigma = featureNormalize(X)

X = np.array(X)
X = np.insert(X, 0, 1, axis = 1)
y = np.array(y).reshape(m, 1)

##================ Part 2: Gradient Descent ================##
print('Running Gradient Descent...')

alpha = 0.01
iterations = 400

theta = np.zeros([3,1])

theta, J_history = gradientDescentMulti(X, y, theta, alpha, iterations)

#Plot the convergence plot
plt.plot(J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

print('Theta computed from gradient descent: \n',theta)

X_pred = np.array([1650, 3])
X_predn = (X_pred - mu)/sigma
X_predn = np.insert(X_predn, 0, 1)
y_pred = X_predn.dot(theta)

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n', np.round(float(y_pred),3))

input('Program paused. Press enter to continue.\n')

##================ Part 3: Normal Equations ================##
print('Solving with normal equations...')

data = np.array(pd.read_table('ex1data2.txt', delimiter=','))
X = data[:,:-1]
y = data[:,-1]
m = len(X)

X = np.insert(X, 0, 1, axis = 1)

theta = normalEquation(X, y)
print('Theta computed from the normal equations:\n', theta);

X_pred = np.array([1,1650,3])
y_pred = X_pred.dot(theta)
print(y_pred)
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n', np.round(float(y_pred),3))



















