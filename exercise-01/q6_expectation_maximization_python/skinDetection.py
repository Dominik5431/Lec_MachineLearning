import numpy as np
from tqdm import tqdm
from estGaussMixEM import estGaussMixEM
from getLogLikelihood import getLogLikelihood


def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel

    #####Insert your code here for subtask 6g#####
    sweights, smeans, scovariances = estGaussMixEM(sdata, K, n_iter, epsilon)
    nweights, nmeans, ncovariances = estGaussMixEM(ndata, K, n_iter, epsilon)
    result = np.zeros(np.shape(img))
    for i in tqdm(np.arange(np.shape(img)[0])):
        for j in np.arange(np.shape(img)[1]):
            pcgs = np.exp(getLogLikelihood(smeans, sweights, scovariances, [img[i,j]])) #prob of color given skin
            pcgn = np.exp(getLogLikelihood(nmeans, nweights, ncovariances, [img[i,j]])) #prob of color given non-skin
            if pcgs/pcgn > theta:
                result[i,j] = [255,255,255]
            else:
                result[i,j] = [0,0,0]
    return result
