import numpy as np
from numpy.random import choice
import sys
import os
dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir)

from simpleClassifier import simpleClassifier
def adaboostSimple(X, Y, K, nSamples):
    # Adaboost with decision stump classifier as weak classifier
    #
    # INPUT:
    # X         : training examples (numSamples x numDim)
    # Y         : training lables (numSamples x 1)
    # K         : number of weak classifiers to select (scalar) 
    #             (the _maximal_ iteration count - possibly abort earlier
    #              when error is zero)
    # nSamples  : number of training examples which are selected in each round (scalar)
    #             The sampling needs to be weighted!
    #             Hint - look at the function 'choice' in package numpy.random
    #
    # OUTPUT:
    # alphaK 	: voting weights (K x 1) - for each round
    # para		: parameters of simple classifier (K x 2) - for each round
    #           : dimension 1 is j
    #           : dimension 2 is theta

    #####Insert your code here for subtask 1c#####
    def classifier(data, j, theta, p):
        return 1 if p*data[j] > p * theta else -1
    def error(theta, data, labels, j, weights, p):
        result = 0
        for i in np.arange(data.shape[0]):
            if np.abs(classifier(data[i], j, theta, p)-labels[i])>1e-6:
                result += weights[i] 
        return result/np.sum(weights)
    weights = np.ones(len(Y))/len(Y)
    alphaK = np.zeros(K)
    para = np.zeros((K,2))
    for k in np.arange(K):
        ind = np.random.choice(len(Y), size=nSamples, p=weights)
        #print(ind)
        para[k,0], para[k,1] = simpleClassifier(X[ind], Y[ind])
        e = error(para[k,1], X, Y, int(para[k,0]), weights, +1)
        print('Error: ', e)
        if np.abs(e) < 1e-6:
            return alphaK, para
        alphaK[k] = 0.5 * np.log((1-e)/e)
        weights_new = weights.copy()
        for i in np.arange(len(Y)):
            weights_new[i] *= np.exp(-alphaK[k] * Y[i] * classifier(X[i], int(para[k,0]), para[k,1], +1))
        print(weights_new)
        
        weights_new /= np.sum(weights_new)
        weights = weights_new
    #print(alphaK)
    return alphaK, para
