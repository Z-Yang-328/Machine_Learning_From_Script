# Machine learning exercise 3

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll

##=========== Part 1: Loading and Visualizing Data =============##

print('loading Data...')

data = sio.loadmat('ex3data1.mat')
X = data['X']
y = data['y']
m, n = X.shape

num_labels = len(np.unique(y))
rand_indices = np.random.randint(0, m, 100)
sel = X[rand_indices, :]

input('Program paused, press enter to continue...')

##============ Part 2b: One-vs-All Training ============##

print('\nTraining One-vs-All Logistic Regression...\n')

lambda_ = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_)

input('\nProgram paused, press enter to continue...\n')

##================ Part 3: Predict for One-Vs-All ================##

pred = predictOneVsAll(all_theta, X)

print('\nTraining Set Accuracy: {:0.1f}%\n'.format(np.mean(pred == y)*100))