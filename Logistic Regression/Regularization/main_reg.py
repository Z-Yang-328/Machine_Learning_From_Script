# Machine learning excercise 2 -- logistic regression and regularization

import numpy as np
import pandas as pd
import scipy.optimize as opt

from plotDataReg import plotData
from mapFeature import mapFeature
from optimizeReg import costFunction, gradient, logisticRegression, predict

# Load data
data = pd.read_table('ex2data2.txt', delimiter=',', header=None)

X_raw = data.iloc[:, :-1]
y_raw = data.iloc[:, -1]

plotData(X_raw, y_raw)

input('\nProgram paused. Press enter to continue.\n')
##=========== Part 1: Regularized Logistic Regression ============##

# Add polynomial features
X = mapFeature(X_raw[0], X_raw[1])
X = np.matrix(X)
m, n = X.shape
y = np.matrix(y_raw).reshape(m, 1)
initial_theta = np.zeros(n)
# Set regularization parameter
lambda_ = 1

cost = costFunction(initial_theta, X, y, lambda_)
grad = gradient(initial_theta, X, y, lambda_)

print('Cost at initial theta (zeros):{:0.3f}\n'.format(cost))
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros) - first five values only:\n')
print(' {:0.4f}\n'.format(grad[0]),'{:0.4f}\n'.format(grad[1]),'{:0.4f}\n'.format(grad[2]),
      '{:0.4f}\n'.format(grad[3]),'{:0.4f}\n'.format(grad[4]))
print('\nExpected gradients (approx) - first five values only:\n')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

input('\nProgram paused. Press enter to continue.\n')

##=========== Part 2: Optimization and Accuracies ============##

result = opt.fmin_tnc(func=costFunction, x0=initial_theta, fprime=gradient, args=(X, y, lambda_))

#result = logisticRegression(X, y, initial_theta, lambda_)
theta = result[0]
cost = costFunction(theta, X, y, lambda_)

input('\nProgram paused. Press enter to continue.\n')
print(theta)
p = predict(theta, X)

print('Train Accuracy: {:0.1f}%\n'.format(np.mean(p==y) * 100))
print('Expected accuracy (with lambda=1):83.1 (approx)\n')

