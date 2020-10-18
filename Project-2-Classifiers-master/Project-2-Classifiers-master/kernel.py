import numpy as np

def linear_kernel(X, Y):
    """
        Compute the linear kernel between two matrices X and Y::
            K(x, y) = <x, y>
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    return np.dot(X, Y.T)

def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    K = np.dot(X, Y.T)
    np.add(K, c, out = K)
    np.power(K, p, out=K)
    return K

def fast_rbf_kernel(X, Y, gamma, BUFF, OUT):
    X2 = X.reshape((X.shape[0], 1, X.shape[1]), order='F')
    Y2 = Y.reshape((1, Y.shape[0], Y.shape[1]), order='F')
    np.subtract(X2, Y2, out=BUFF)
    np.power(BUFF, 2, out=BUFF)
    np.sum(BUFF, axis=2, out=OUT)
    np.multiply(OUT, -gamma, out=OUT)
    np.exp(OUT, out=OUT)
    return OUT

def rbf_kernel(X, Y, gamma, mem_limit=500e6, vervose=False):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    bs = int(np.sqrt((mem_limit/8)/X.shape[1]))
    Nx, Rx = divmod(X.shape[0], bs) 
    Ny, Ry = divmod(Y.shape[0], bs)
    BUFF = np.empty((bs, bs, X.shape[1]))
    K = np.empty((X.shape[0], Y.shape[0]))
    if vervose:
        print("Bs= {}, Nx = {}, Ny = {}".format(bs, Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            fast_rbf_kernel(    X[i*bs:(i+1)*bs,:], 
                                Y[j*bs:(j+1)*bs,:], 
                                gamma, BUFF, 
                                K[i*bs:(i+1)*bs, j*bs:(j+1)*bs])
            if vervose:
                print("success: {}, {}".format(i,j))

        fast_rbf_kernel(    X[i*bs:(i+1)*bs,:], 
                            Y[Ny*bs:,:], gamma, 
                            BUFF[:, 0:Ry,:],
                            K[i*bs:(i+1)*bs, Ny*bs:])
    for j in range(Ny):
        fast_rbf_kernel(    X[Nx*bs:,:], 
                            Y[j*bs:(j+1)*bs,:], 
                            gamma, BUFF[0:Rx, :, :], 
                            K[Nx*bs:, j*bs:(j+1)*bs])

    fast_rbf_kernel(    X[Nx*bs:,:], 
                        Y[Ny*bs:,:], gamma, 
                        BUFF[0:Rx, 0:Ry,:],
                        K[Nx*bs:, Ny*bs:])
    return K

