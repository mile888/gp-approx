#!/usr/bin/env python

####  GPR approximation functions  ###


import numpy as np



def cov_kernel(x, z, ell, sf2):
    """ Squared Exponential Automatic Relevance Determination (SE ARD) kernel.
    
    # Arguments:
        x:    Sample vector of training input matrix X (1 x Nx) - Nx is the number of inputs to the GP.
        z:    Unseen smaple vector (1 x Nx) - Nx is the number of inputs to the GP.
        ell:  Length scales vector of size Nx.
        sf2:  Signal variance (scalar)
    """
    
    dist = np.sum(((x - z)**2 / ell**2), axis=1)
    return sf2 * np.exp(-0.5 * dist)



def gp(X, Y, X_new_mean, invK, hyper, alpha=None):
    """ Gaussian Process
    
    # Arguments
        X_new_mean: Input to the GP of size (1 x Nx).
        X: Training data matrix with inputs of size (N x Nx) - Nx is the number of inputs to the GP.
        Y: Training data matrix with outpyts of size (N x Ny).
        invK: Array with the inverse covariance matrices of size (Ny x N x N) - Ny number of outputs from the GP and N number of training points.
        hyper: Array with hyperparame|ters [ell_1 .. ell_Nx sf sn].
        alpha: Training data matrix with invK time outputs of size (Ny x N).

    # Returns
        mean: The estimated mean.
        var: The estimated variance
    """

    Ny = len(invK)
    N, Nx = X.shape
    
    mean  = np.zeros((Ny, 1))
    var   = np.zeros((Ny, 1))

    for output in range(Ny):
        ell = hyper[output, 0:Nx]
        sf2 = hyper[output, Nx]**2

        ks = np.zeros((N, 1))
        for i in range(N):
            ks[i] = cov_kernel(X[i, :].reshape(1,-1), X_new_mean, ell, sf2)
        kss = cov_kernel(X_new_mean, X_new_mean, ell, sf2)

        if alpha is not None:
            mean[output] =  ks.T @ alpha[output]
        else:
            mean[output] = ks.T @ invK[output, :, :] @ Y[:,output]
            
        var[output] = kss - ks.T @ invK[output, :, :] @ ks
        
    return mean, var



def gp_TA1(X, Y, X_mean, X_covar, invK, hyper, alpha=None):
    """ Gaussian Process with 1-st Taylor Approximation

    This uses a first order taylor for the mean evaluation (a normal GP mean),
    and a first order taylor for estimating the variance.

    # Arguments:
        X: Training data matrix with inputs of size NxNx - Nx number of inputs to the GP.
        Y: Training data matrix with outpyts of size (N x Ny).
        X_mean: Mean from the new unseen/tested data of size (1 x Nx)
        X_covar: Covariance from the new unseen/tested data of size (Nx x Nx)
        invK: Array with the inverse covariance matrices of size (Ny x N x N) - Ny number of outputs from the GP and N number of training points.
        hyper: Array with hyperparameters [ell_1 .. ell_Nx sf sn].
    
    # Returns
        mean: The estimated mean vector of size (Ny x 1)
        var:  The estimated variance vector of size (Ny x 1)
    """

    Ny    = len(invK)
    N, Nx = X.shape
    mean  = np.zeros((Ny, 1))
    var   = np.zeros((Ny, 1))
    v     = X - np.tile(X_mean, (N, 1))

    var_TA1 = np.zeros((Ny, 1))
    d_mean     = np.zeros((Nx, 1))
    
    for output in range(Ny):
        ell = hyper[output, :Nx]
        w = 1 / ell**2
        sf2 = hyper[output, Nx]**2
        iK = invK[output]
        
        if alpha is not None:
            alpha_a = alpha[output]
        else:
            alpha_a = iK @ Y[:,output]
                
        kss = sf2
        ks = np.zeros((N, 1))
        for i in range(N):
            ks[i] = cov_kernel(X[i, :].reshape(1,-1), X_mean, ell, sf2)
         
        # output mean
        mean[output] = ks.T @ alpha_a
        
        # output variance
        invKks = iK @ ks
        var[output] = kss - (ks.T @ invKks)
        
        d_mean = (w.reshape(-1,1) * (v * ks).T) @ alpha_a
        var_TA1[output] = var[output] + d_mean.T @ X_covar @ d_mean

    return [mean, var_TA1]



def gp_TA2(X, Y, X_mean, X_covar, invK, hyper, alpha=None):
    """ Gaussian Process with 2-nd Taylor Approximation

    This uses a first order taylor for the mean evaluation (a normal GP mean),
    and a second order taylor for estimating the variance.

    # Arguments
        X: Training data matrix with inputs of size NxNx - Nx number of inputs to the GP.
        Y: Training data matrix with outpyts of size (N x Ny).
        X_mean: Mean from the new unseen/tested data of size (1 x Nx)
        X_covar: Covariance from the new unseen/tested data of size (Nx x Nx)
        invK: Array with the inverse covariance matrices of size (Ny x N x N) - Ny number of outputs from the GP and N number of training points.
        hyper: Array with hyperparameters [ell_1 .. ell_Nx sf sn].

    # Returns
        mean: The estimated mean vector of size (Ny x 1)
        var:  The estimated variance vector of size (Ny x 1)
    """
    
    Ny    = len(invK)
    N, Nx = X.shape
    mean  = np.zeros((Ny, 1))
    var   = np.zeros((Ny, 1))
    v     = X - np.tile(X_mean, (N, 1))

    var_TA2 = np.zeros((Ny, 1))
    d_mean   = np.zeros((Nx, 1))
    dd_var = np.zeros((Nx, Nx))

    
    for output in range(Ny):
        ell = hyper[output, :Nx]
        w = 1 / ell**2
        sf2 = hyper[output, Nx]**2
        iK = invK[output]
        
        if alpha is not None:
            alpha_a = alpha[output]
        else:
            alpha_a = iK @ Y[:,output]
                
        kss = sf2
        ks = np.zeros((N, 1))
        for i in range(N):
            ks[i] = cov_kernel(X[i, :].reshape(1,-1), X_mean, ell, sf2)

            
        mean[output] = ks.T @ alpha_a
        d_mean = (w.reshape(-1,1) * (v * ks).T) @ alpha_a
        
        invKks = iK @ ks
        var[output] = kss - (ks.T @ invKks)
        
      
        for i in range(Nx):
            for j in range(Nx):
                dd_var1a = 0
                dd_var1b = 0
                dd_var2 = 0
                dd_var1a += (v[:, i].reshape(-1,1) * ks).T @ iK 
                dd_var1b += dd_var1a @ (v[:, j].reshape(-1,1) * ks)
                dd_var2 += (v[:, i].reshape(-1,1) * v[:, j].reshape(-1,1) * ks).T @ invKks
                dd_var[i,j] += -2 * w[i] * w[j] * (dd_var1b + dd_var2)                         
                
                if i==j:
                    dd_var[i, j] = dd_var[i,j] + 2 * w[i] * (kss - var[output])

        
        mean_mat = d_mean.reshape(-1,1) @ d_mean.reshape(-1,1).T
        var_TA2[output]= var[output] + np.trace(X_covar @ (.5* dd_var + mean_mat))

    return [mean, var_TA2]



def gp_EM(X, Y, X_mean, X_covar, invK, hyper, alpha=None):
    """ Gaussian Process with Exact Moment Matching

    The first and second moments are used to compute the mean and covariance of the
    posterior distribution with a stochastic input distribution. This assumes a
    zero prior mean function and the squared exponential kernel.

    # Arguments
        X: Training data matrix with inputs of size NxNx - Nx number of inputs to the GP.
        Y: Training data matrix with outpyts of size (N x Ny).
        X_mean: Mean from the new unseen/tested data of size (1 x Nx)
        X_covar: Covariance from the new unseen/tested data of size (Nx x Nx)
        invK: Array with the inverse covariance matrices of size (Ny x N x N) - Ny number of outputs from the GP and N number of training points.
        hyper: Array with hyperparameters [ell_1 .. ell_Nx sf sn].

    # Returns
        mean: The estimated mean vector of size (Ny x 1)
        var:  The estimated variance vector of size (Ny x 1).
    """
 
    hyper = np.log(hyper)
    
    Ny      = len(invK)
    N, Nx   = np.shape(X)
    mean    = np.zeros((Ny, 1))
    beta    = np.zeros((N, Ny))
    log_k   = np.zeros((N, Ny))
    v       = X - np.tile(X_mean, (N, 1))

    covariance = np.zeros((Ny, Ny))
    

    for a in range(Ny):
        ell = hyper[a, :Nx].reshape(-1,1)
        sf2 = (hyper[a, Nx]).reshape(-1,1)
        iK =  invK[a]
        beta = alpha[a].reshape(-1,1)
        
        iLambda   = np.diag(np.exp(-2 * hyper[a, :Nx]))
        S  = X_covar + np.diag(np.exp(2 * hyper[a, :Nx]))
        iS = iLambda @ (np.eye(Nx, Nx) - np.linalg.solve((np.eye(Nx,Nx) + (X_covar @ iLambda)), (X_covar @ iLambda)))

        T  = v @ iS
        c  = np.exp(2 * sf2) / np.sqrt(determinant(S)) * np.exp(np.sum(ell))
        q2 = c * np.exp(- 0.5 * np.sum((T * v), axis=1, keepdims=True))
        qb = q2 * beta 
        mean[a] = np.sum(qb, axis=0, keepdims=True)
        
        t  = np.tile(np.exp(hyper[a, :Nx]), (N, 1))
        v1 = v / t
        log_k[:, a] = 2 * sf2 - np.sum((v1 * v1), axis=1) * 0.5


    for a in range(Ny):
        ii = v / np.tile(np.exp(2 * hyper[a, :Nx]), (N, 1))
        for b in range(a + 1):
            S = X_covar @ np.diag(np.exp(-2 * hyper[a, :Nx]) + np.exp(-2 * hyper[b, :Nx])) + np.eye(Nx, Nx)
            t = np.array([[1.0 / np.sqrt(determinant(S))]])
            ij = v / np.tile(np.exp(2 * hyper[b, :Nx]), (N, 1))
            Q = np.exp(np.tile(log_k[:, a].reshape(-1,1), (1, N)) + np.tile(log_k[:, b].reshape(-1,1).T, (N, 1))
                + distance(ii, -ij, np.linalg.solve(S, X_covar * 0.5), N))
            
            A = alpha[a].reshape(-1,1) @ (alpha[b].reshape(-1,1)).T
            if b == a:
                A = A - invK[a]
            A = A * Q
            
            covariance[a, b] = t * np.sum(A)
            covariance[b, a] = covariance[a, b]
            
        covariance[a, a] = covariance[a, a] + np.exp(2 * hyper[a, Nx])
        
    covariance = covariance - mean @ mean.T    
    var_EM = np.diag(covariance).reshape(-1,1)
    
    return [mean, var_EM]


def determinant(S):
    """ Determinant
    # Arguments
        S:  Covariance 
    """
    return np.exp(np.trace(np.log(S)))


def distance(a1, b1, Q1, N):
    """ Mahalanobis distance """
    
    aQ =  a1 @ Q1
    bQ =  b1 @ Q1
    K1  = (np.tile(np.sum((aQ * a1), axis=1, keepdims=True), (1, N)) 
           + np.tile(np.sum((bQ * b1), axis=1, keepdims=True).T, (N, 1)) 
           - 2 * (aQ @ b1.T))
    return K1
