# Find the closest centroids for each data point

import numpy as np

def findClosestCentroids(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0]).reshape(-1, 1)

    for j in range(len(X)):
        dist = np.zeros(K)
        #cent = 0
        for i in range(K):
            dist[i] = np.sqrt(np.sum((X[j] - centroids[i]) ** 2))
        idx[j] = dist.argmin() + 1

    return idx