# Machine Learning Exercise 2 -- Logistic Regression and Regularization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plotData import plotData
from optimize import costFunction, gradient, logisticRegression
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict

# Load Data

data = pd.read_table('ex2data1.txt', delimiter = ',', header = None)
X_raw = data.iloc[:, :2]
y_raw = data.iloc[:, 2]

##============= Part 1: Plotting =============##

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')

plotData(X_raw, y_raw)

input('\nProgram paused. Press enter to continue.\n')


##============= Part 2: Compute Cost and Gradient =============##
X = np.insert(X_raw.values, 0, 1, axis = 1)
X = np.matrix(X)
m, n = X.shape
y = np.matrix(y_raw).reshape(m, 1)

initial_theta = np.zeros(n)
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)

print('Cost at initial theta (zeros):', cost);
print('\nExpected cost (approx): 0.693\n');
print('Gradient at initial theta (zeros): \n');
print('',np.round(grad[0],4),'\n',np.round(grad[1],4),'\n',np.round(grad[2],4));
print('\nExpected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');

input('\nProgram paused. Press enter to continue.\n')

##============= Part 3: Optimizing =============##

result = logisticRegression(X, y, initial_theta)
theta = result[0]
cost = costFunction(result[0], X, y)
print('Cost at theta found by fminunc:', np.round(cost,3))
print('Expected cost (approx): 0.203\n')
print('theta: \n')
print('',np.round(theta[0],4),'\n',np.round(theta[1],4),'\n',np.round(theta[2],4));
print('Expected theta (approx):\n')
print(' -25.161\n 0.206\n 0.201\n')

plotDecisionBoundary(theta, X_raw, y_raw)

input('\nProgram paused, press enter to continue.\n')

##============= Part 4: Predictions and Accuracies =============##

pred = predict(theta, X)
accu = np.mean(pred == y) * 100
print('Training Accuracy:{:0.1f}%'.format(accu) )
print('Expected accuracy (approx): 89.0%\n')








