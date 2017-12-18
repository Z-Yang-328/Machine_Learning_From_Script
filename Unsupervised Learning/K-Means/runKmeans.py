# Run K-Means clustering

import numpy as np

from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from plotProgressKmeans import plotProgressKmeans

def runKmeans(X, centroids, max_iters, plot_progress):
    m, n = X.shape
    K = centroids.shape[0]
    centroids = centroids
    previous_centroids = centroids
    idx = np.zeros([m,1])

    for i in range(max_iters):
        print('K-Means iteration {}/{}...\n'.format(i+1, max_iters))

        idx = findClosestCentroids(X, centroids)

        if plot_progress:
            plotProgressKmeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            input('Press enter to continue.\n')

        centroids = computeCentroids(X, idx, K)

    return (centroids, idx)
