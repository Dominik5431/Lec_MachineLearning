import numpy as np
from getLogLikelihood import getLogLikelihood

def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Insert your code here for subtask 6c#####
    N,K = np.shape(gamma)
    D = np.shape(X)[1]
    
    NHat = np.zeros(K)
    for k in np.arange(K):
        for n in np.arange(N):
            NHat[k] += gamma[n,k]
            
    weights = NHat/N
    
    means = np.zeros((K,D))
    for k in np.arange(K):
        for n in np.arange(N):
            means[k] += gamma[n,k] * X[n]
        means[k] /= NHat[k]
    
    covariances = np.zeros((D,D,K))
    for k in np.arange(K):
        for n in np.arange(N):
            covariances[:,:,k] += gamma[n,k] * np.outer(X[n]-means[k], (X[n]-means[k]))
        covariances[:,:,k] /= NHat[k]
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
        
    # try:    
    #     logLikelihood = getLogLikelihood(means, weights, covariances, X)
    # except: 
    #     logLikelihood = 0
    return weights, means, covariances, logLikelihood
