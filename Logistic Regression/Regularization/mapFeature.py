# Create polynomial features and add ones

import numpy as np

def mapFeature(X1, X2):
    m = len(X1)
    degree = 5
    results = np.empty([m, 1])
    for i in range(1, degree):
        for j in range(0, i):
            first = np.power(X1, i - j)
            second = np.power(X2, j)
            result = np.matrix(np.multiply(first, second)).reshape(m, 1)
            results = np.append(results, result, axis=1)

    results = np.insert(results, 0, 1, axis=1)

    return results