import numpy as np


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    #####Insert your code here for subtask 5a#####
    pos = np.arange(-5,5.0,0.1)
    estDensity = np.zeros((len(pos),2))
    estDensity[:,0] = pos
    for i,x in enumerate(pos):
        for k in np.arange(len(samples)):
            estDensity[i,1] += 1/(2*np.pi*h**2)**0.5 * np.exp(-np.abs(x-samples[k])**2/(2*h**2))
        estDensity[i,1] /= len(samples)
    # Compute the number of samples created
    return estDensity
