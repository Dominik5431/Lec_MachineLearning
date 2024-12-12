import numpy as np
from getLogLikelihood import getLogLikelihood
from getLogLikelihood import Gaussian


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####
    N = np.shape(X)[0]
    D = np.shape(X)[1]
    K = len(weights)
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    gamma = np.zeros((N,K))
    for n in np.arange(N):
        norm = 0
        for l in np.arange(K):
            norm += weights[l] * Gaussian(means[l], covariances[:,:,l], X[n], D)  
        for k in np.arange(K):
            gamma[n,k] = weights[k] * Gaussian(means[k], covariances[:,:,k], X[n], D)
            gamma[n,k] /= norm           
    return [logLikelihood, gamma]
