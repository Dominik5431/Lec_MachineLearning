import numpy as np

def eval_adaBoost_leastSquare(X, alphaK, para):
    # INPUT:
    # para		: parameters of simple classifier (K x (D +1)) 
    #           : dimension 1 is w0
    #           : dimension 2 is w1
    #           : dimension 3 is w2
    #             and so on
    # alphaK    : classifier voting weights (K x 1)
    # X         : test data points (numSamples x numDim)
    #
    # OUTPUT:
    # classLabels: labels for data points (numSamples x 1)
    # result     : weighted sum of all the K classifier (scalar)

    #####Insert your code here for subtask 1e#####
    def classifier(data, weight, bias):
        return 1 if np.dot(weight,data)+bias >= 0 else -1
    numSamples = np.shape(X)[0]
    result = np.zeros(numSamples)
    classLabels = np.zeros(numSamples)
    for i in np.arange(numSamples):
        for k in np.arange(len(alphaK)):
            result[i] += alphaK[k] * classifier(X[i], (para[k,1], para[k,2]), para[k,0])
        classLabels[i] = 1 if result[i]>=0 else -1 
    return [classLabels, result]

