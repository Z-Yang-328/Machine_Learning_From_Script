# Plot decision boundary

import matplotlib.pyplot as plt
import numpy as np

from optimize import sigmoid

def plotDecisionBoundary(theta, X_raw, y_raw):

    X0 = X_raw[y_raw == 0]
    X1 = X_raw[y_raw == 1]
    theta = theta.tolist()
    t = np.arange(30, 100, 10)
    s = (-theta[0] - theta[1] * t) / theta[2]
    plt.plot(t, s)
    plt.scatter(X0[0], X0[1], marker='+')
    plt.scatter(X1[0], X1[1], marker='o')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Admitted', 'Not admitted'])
    plt.title('Decision Boundary')
    plt.show()