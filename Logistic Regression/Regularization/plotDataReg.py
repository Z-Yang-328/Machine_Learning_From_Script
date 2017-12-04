# Plot data

import matplotlib.pyplot as plt

def plotData(X, y):
    X0 = X[y == 0]
    X1 = X[y == 1]
    plt.scatter(X0[0], X0[1], marker='+')
    plt.scatter(X1[0], X1[1], marker='o')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(['y = 1', 'y = 0'])
    plt.title('Exam scores vs. Admission')
    plt.show()