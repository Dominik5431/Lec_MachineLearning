import numpy as np
import sys
import os
dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir)
from numpy.random import choice
from simpleClassifier import simpleClassifier
from eval_adaBoost_simpleClassifier import eval_adaBoost_simpleClassifier

def adaboostCross(X, Y, K, nSamples, percent):
    # Adaboost with an additional cross validation routine
    #
    # INPUT:
    # X         : training examples (numSamples x numDims )
    # Y         : training lables (numSamples x 1)
    # K         : number of weak classifiers to select (scalar)
    #             (the _maximal_ iteration count - possibly abort earlier)
    # nSamples  : number of training examples which are selected in each round. (scalar)
    #             The sampling needs to be weighted!
    # percent   : percentage of the data set that is used as test data set (scalar)
    #
    # OUTPUT:
    # alphaK    : voting weights (K x 1)
    # para      : parameters of simple classifier (K x 2)
    # testX     : test dataset (numTestSamples x numDim)
    # testY     : test labels  (numTestSamples x 1)
    # error	    : error rate on validation set after each of the K iterations (K x 1)

    #####Insert your code here for subtask 1d#####
    # Randomly sample a percentage of the data as test data set
    indtrain = choice(len(Y), size=int((1-percent) * len(Y)), replace=False)
    indtest = np.delete(np.arange(len(Y)), indtrain)
    X_train = X[indtrain]
    testX = X[indtest]
    Y_train = Y[indtrain]
    testY = Y[indtest]
    error = np.zeros(K)
    
    print(X_train)
    print(Y_train)
    print(X)
    
    def classifier(data, j, theta, p):
        return 1 if p*data[j] > p * theta else -1
    def calc_error(theta, data, labels, j, weights, p):
        result = 0
        for i in np.arange(data.shape[0]):
            if np.abs(classifier(data[i], j, theta, p)-labels[i])>1e-6:
                result += weights[i] 
        return result/np.sum(weights)
    
    def test_error(testY,classLabels):
        result = 0
        for i in np.arange(len(testY)):
            if np.abs(testY[i]-classLabels[i])>1e-5:
                result+=1
        return result
    
    weights = np.ones(len(Y_train))/len(Y_train)
    alphaK = np.zeros(K)
    para = np.zeros((K,2))
    for k in np.arange(K):
        #print(weights)
        ind = np.random.choice(len(Y_train), size=nSamples, p=weights)
        #print(ind)
        para[k,0], para[k,1] = simpleClassifier(X_train[ind], Y_train[ind])
        e = calc_error(para[k,1], X_train, Y_train, int(para[k,0]), weights, +1)
        if np.abs(e)<1e-6:
            return alphaK, para, testX, testY, error
        alphaK[k] = 0.5 * np.log((1-e)/e)
        print(alphaK)
        weights_new = weights.copy()
        for i in np.arange(len(Y_train)):
            weights_new[i] *= np.exp(-alphaK[k] * Y_train[i] * classifier(X_train[i], int(para[k,0]), para[k,1], +1))
        
        classLabels, result = eval_adaBoost_simpleClassifier(testX, alphaK[:k+1], para[:k+1,:])
        #print(classLabels)
        #print(testY)
        error[k] = test_error(testY, classLabels)
        weights_new /= np.sum(weights_new)
        weights = weights_new
    #print(alphaK)
    
    return alphaK, para, testX, testY, error

