import numpy as np


def leastSquares(data, label):
    # Sum of squared error shoud be minimized
    #
    # INPUT:
    # data        : Training inputs  (num_samples x dim)
    # label       : Training targets (num_samples x 1)
    #
    # OUTPUT:
    # weights     : weights   (dim x 1)
    # bias        : bias term (scalar)

    #####Insert your code here for subtask 1a#####
    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly)
    data_w_bias = np.ones((len(data[:, 0]), len(data[0]) + 1))
    data_w_bias[:, :-1] = data
    #dimensions: (D+1)xN * Nx(D+1) * (D+1)xN * Nx1  --> (D+1)x1
    weight_w_bias = np.linalg.inv(np.transpose(data_w_bias) @ data_w_bias) @ np.transpose(data_w_bias) @ label
    weight = weight_w_bias[:-1]
    bias = weight_w_bias[-1]
    return weight, bias
