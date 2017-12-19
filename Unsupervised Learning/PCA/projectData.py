# Project data

def projectData(X, U, K):
    Z = X.dot(U[:,:K])

    return Z