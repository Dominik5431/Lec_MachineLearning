import numpy as np


def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    #####Insert your code here for subtask 5b#####
    pos = np.arange(-5,5.0,0.1)
    estDensity = np.zeros((len(pos),2))
    estDensity[:,0] = pos
    for i,x in enumerate(pos):
        n = 0 #number of neighbors sampled so far
        h = 0
        while n<k:
            n = 0
            h += 0.01
            for j,xn in enumerate(samples):
                if np.abs(x-xn)< h:
                    n += 1
            pass
        estDensity[i,1] = k/(len(samples)*2*h)
    # Compute the number of the samples created
    return estDensity
