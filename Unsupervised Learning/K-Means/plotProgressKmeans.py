# Plot K-Means clustering progress

import matplotlib.pyplot as plt

from plotDataPoints import plotDataPoints
from drawLine import drawLine

def plotProgressKmeans(X, centroids, previous, idx, K, i):

    plotDataPoints(X, idx, K)

    plt.scatter(centroids[:,0], centroids[:,1], marker = 'x', edgecolors='b')

    for j in range(centroids.shape[0]):
        drawLine(centroids[j,:], previous[j,:])

    plt.title('Iteration number {}'.format(i))
    plt.show()