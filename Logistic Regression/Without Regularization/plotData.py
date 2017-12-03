# Plot data

import matplotlib.pyplot as plt

def plotData(X, y):
    X0 = X[y == 0]
    X1 = X[y == 1]
    plt.scatter(X0[0], X0[1], marker='+')
    plt.scatter(X1[0], X1[1], marker='o')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Admitted', 'Not admitted'])
    plt.title('Exam scores vs. Admission')
    plt.show()