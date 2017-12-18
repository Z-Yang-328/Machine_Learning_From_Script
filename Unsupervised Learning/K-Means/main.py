# Machine learning exercise 7

import numpy as np
import scipy.io as sio

from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from runKmeans import runKmeans

#==================== Part 1: Find closest centroids =====================#
data = sio.loadmat('ex7data2.mat')
X = data['X']

K = 3  # number of centroids
initial_centroids = np.array([[3,3],[6,2],[8,5]])

idx = findClosestCentroids(X, initial_centroids)

print('Closest centroids for the first 3 examples: \n')
print(idx[:3])
print('\n(the closest centroids should be 1, 3, 2 respectively)\n')

input('Program paused. Press enter to continue.\n')

#==================== Part 2: Compute means =====================#
print('\nComputing centroids means.\n\n')

centroids = computeCentroids(X, idx, K)

print('Centroids computed after initial finding of closest centroids: \n')
print(centroids)
print('\n(the centroids should be\n')
print('   [ 2.428301 3.157924 ]\n')
print('   [ 5.813503 2.633656 ]\n')
print('   [ 7.119387 3.616684 ]\n\n')

input('Program paused. Press enter to continue.\n')

#==================== Part 3: K-Means Clustering =====================#
print('\nRunning K-Means clustering on example dataset.\n\n')

# Set K-Means parameters
K = 3
max_iters = 10
initial_centroids = np.array([[3,3],[6,2],[8,5]])

centroids, idx = runKmeans(X, initial_centroids, max_iters, True)








