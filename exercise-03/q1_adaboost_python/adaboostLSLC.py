import numpy as np
import sys
import os
dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir)
from numpy.random import choice
from leastSquares import leastSquares

def adaboostLSLC(X, Y, K, nSamples):
    # Adaboost with least squares linear classifier as weak classifier
    # for a D-dim dataset
    #
    # INPUT:
    # X         : the dataset (numSamples x numDim)
    # Y         : labeling    (numSamples x 1)
    # K         : number of weak classifiers (iteration number of Adaboost) (scalar)
    # nSamples  : number of data which are weighted sampled (scalar)
    #
    # OUTPUT:
    # alphaK    : voting weights (K x 1)
    # para      : parameters of least square classifier (K x 3) 
    #             For a D-dim dataset each least square classifier has D+1 parameters
    #             w0, w1, w2........wD

    #####Insert your code here for subtask 1e#####
    def classifier(data, weight, bias):
        return 1 if np.dot(weight,data)+bias >= 0 else -1
    def error(weightClassifier, biasClassifier, data, labels, weights):
        result = 0
        for i in np.arange(data.shape[0]):
            if np.abs(classifier(data[i], weightClassifier, biasClassifier)-labels[i])>1e-6:
                result += weights[i] 
        return result/np.sum(weights)
    numSamples = len(Y)
    weights = np.ones(numSamples)/numSamples
    alphaK = np.zeros(K)
    para = np.zeros((K,3))
    for k in np.arange(K):
        ind = np.random.choice(numSamples, size=nSamples, p=weights)
        #print(ind)
        leastSquareRes = leastSquares(X[ind], Y[ind])
        para[k,1], para[k,2] = leastSquareRes[0]
        para[k,0] = leastSquareRes[1]
        e = error((para[k,1], para[k,2]), para[k,0], X, Y, weights)
        if np.abs(e)<1e-6:
            return [alphaK, para]
        alphaK[k] = 0.5 * np.log((1-e)/e)
        weights_new = weights.copy()
        for i in np.arange(numSamples):
            weights_new[i] *= np.exp(-alphaK[k] * Y[i] * classifier(X[i], (para[k,1], para[k,2]), para[k,0]))
        weights_new /= np.sum(weights_new)
        weights = weights_new
    #print(alphaK)
    return [alphaK, para]
