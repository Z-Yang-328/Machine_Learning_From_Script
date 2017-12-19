# Recover data

def recoverData(Z, U, K):
    X_app = Z.dot(U[:,:K].T)

    return X_app