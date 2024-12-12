import numpy as np


def eval_adaBoost_simpleClassifier(X, alphaK, para):
    # INPUT:
    # para	: parameters of simple classifier (K x 2) - for each round
    #           : dimension 1 is j
    #           : dimension 2 is theta
    # alphaK    : classifier voting weights (K x 1)
    # X         : test data points (numSamples x numDim)
    #
    # OUTPUT:
    # classLabels: labels for data points (numSamples x 1)
    # result     : weighted sum of all the K classifier (numSamples x 1)

    #####Insert your code here for subtask 1c#####
    def classifier(data, j, theta, p):
        return 1 if p*data[j] > p * theta else -1
    
    numSamples = np.shape(X)[0]
    result = np.zeros(numSamples)
    classLabels = np.zeros(numSamples)
    for i in np.arange(numSamples):
        for k in np.arange(len(alphaK)):
            result[i] += alphaK[k] * classifier(X[i], int(para[k,0]), para[k,1],+1)
        classLabels[i] = 1 if result[i]>=0 else -1  
    return classLabels, result
