import numpy as np

def Gaussian(mean, covariance, x, D):
    return 1/(2*np.pi)**(D/2) * 1/np.linalg.det(covariance)**0.5 * np.exp(-0.5 * np.dot(np.matrix.transpose(x-mean),np.dot(np.linalg.inv(covariance), x-mean))) 

def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    logLikelihood = 0
    D = np.shape(X)[1]
    for i,x in enumerate(X):
        temp = 0
        for k,w in enumerate(weights):
            temp += w * Gaussian(means[k], covariances[:,:,k], x, D)
        logLikelihood += np.log(temp)
    return logLikelihood

