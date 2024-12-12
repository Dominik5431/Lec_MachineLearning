import numpy as np
from scipy.optimize import fmin


def simpleClassifier(X, Y):
    # Select a simple classifier
    #
    # INPUT:
    # X         : training examples (numSamples x numDim)
    # Y         : training lables (numSamples x 1)
    #
    # OUTPUT:
    # theta 	: threshold value for the decision (scalar)
    # j 		: the dimension to "look at" (scalar)

    #####Insert your code here for subtask 1b#####
    numDim = len(X[0, :])
    numSamples = len(Y)

    def classifier(data, j, theta, p):
        return 1 if p * data[j] > p * theta else -1

    def error(theta, data, labels, j, p):
        result = 0
        for i in np.arange(numSamples):
            if np.abs(classifier(data[i], j, theta, p) - labels[i]) > 1e-6:
                result += 1
        return result

    thetatest = np.zeros(numDim)
    errorval = np.zeros(numDim)
    for j in np.arange(numDim):
        dataj = X[:, j]
        thetapos = dataj.copy()
        errortest = np.zeros(len(thetapos))
        for i in np.arange(0, len(thetapos) - 1):
            thetapos[i] = 0.5 * (thetapos[i] + thetapos[i + 1])
        thetapos[-1] += 0.1
        for i, theta in enumerate(thetapos):
            errortest[i] = error(theta, X, Y, j, +1)
        errorval[j] = min(errortest)
        thetatest[j] = thetapos[np.argmin(errortest)]
    #print(errorval)
    j = np.argmin(errorval)
    theta = thetatest[j]
    return j, theta
