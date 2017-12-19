# Machine learning exercise 7 -- Principle component analysis

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from featureNormalize import featureNormalize
from pca import pca
from projectData import projectData
from recoverData import recoverData
from drawLine import drawLine

##================== Part 1: Load Example Dataset  ===================##
print('Visualizing example dataset for PCA.\n\n')

data = sio.loadmat('ex7data1.mat')
X = data['X']

plt.scatter(X[:,0], X[:,1])
plt.xlim([0.5, 6.5])
plt.ylim([2, 8])
plt.show()

input('Program pasued. Press enter to continue.\n')

##=============== Part 2: Principal Component Analysis ===============##
print('\nRunning PCA on example dataset.\n\n')

X_norm, mu, sigma = featureNormalize(X)

U, S, V = pca(X_norm)


print('Top eigenvector: \n')
print(' U[:,1] = {} {}\n'.format(U[0,1], U[1,1]))

input('Program paused. Press enter to continue.\n')

##=================== Part 3: Dimension Reduction ===================##

print('\nDimension reduction on example dataset.\n\n')

plt.scatter(X_norm[:, 0], X_norm[:, 1])
plt.xlim([-4, 3])
plt.ylim([-4, 3])

K = 1  # the dimension we want to reduce to
Z = projectData(X_norm, U, K)
print('Projection of the first example: {}\n'.format(Z[0].tolist()[0][0]))

X_rec = recoverData(Z, U, K)
print('Approximation of the first example: {} {}\n'.format(X_rec[0,0], X_rec[0,1]))
