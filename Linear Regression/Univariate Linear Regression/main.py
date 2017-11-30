import matplotlib.pyplot as plt
import numpy as np

from plotdata import plotdata
from computeCost import computeCost
from gradientDescent import gradientDescent

##===============Part 1:Plotting Data===============##

#Read data
with open('ex1data1.txt') as f:
    data = []
    for item in f.readlines():
        data.append(item[:-1].split(','))
 
#Assign X and y       
X = [float(d[0]) for d in data]
y = [float(d[1]) for d in data]
m = len(X)  #number of traning examples

#Plot data

plotdata(X, y)

input('Program paused, press enter to continue...')

##===============Part 2:Cost and Gradient Descent===============##

temp = []
for item in X:
	item = [1, item]
	temp.append(item)
X = np.array(temp)
y = np.array(y).reshape(m, 1)

theta = np.zeros([2, 1])

iterations = 1500
alpha = 0.01

print('\nTesting the cost function...\n')
J = computeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = {:0.2f}'.format(J))


input('Program paused, press enter to continue...')

print('\nRunning Gradient Descent\n')
theta = gradientDescent(X, y, theta, alpha, iterations)

print('Theta found by gradient descent:\n');
print(' {}'.format(np.round(float(theta[0]),4)),'\n  {}'.format(np.round(float(theta[1]), 4)));

#Plot the linear fit
plotdata(X[:,1],y, prediction = X.dot(theta), line = True) 













