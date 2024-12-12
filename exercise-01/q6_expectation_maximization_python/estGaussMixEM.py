import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import sys

sys.path.append(
    r'C:\Users\Dominik\Documents\Uni-Dokumente\Master\2. Semester\Machine Learning\exercise-01\q6_expectation_maximization_python')
from EStep import EStep
from MStep import MStep
from regularize_cov import regularize_cov


def estGaussMixEM(data, K, n_iters, epsilon):
    # EM algorithm for estimation gaussian mixture mode
    #
    # INPUT:
    # data           : input data, N observations, D dimensional
    # K              : number of mixture components (modes)
    #
    # OUTPUT:
    # weights        : mixture weights - P(j) from lecture
    # means          : means of gaussians
    # covariances    : covariancesariance matrices of gaussians
    # logLikelihood  : log-likelihood of the data given the model

    #####Insert your code here for subtask 6e#####
    weights = np.ones(K) / K
    kmeans = KMeans(n_clusters=K, n_init=10).fit(data)
    cluster_idx = kmeans.labels_
    means = kmeans.cluster_centers_
    D = np.shape(data)[1]
    covariances = np.zeros((D, D, K))
    for j in np.arange(K):
        data_cluster = data[cluster_idx == j]
        min_dist = np.inf
        for i in np.arange(K):
            dist = np.mean(euclidean_distances(data_cluster, [means[i]], squared=True))
            if dist < min_dist:
                min_dist = dist
        covariances[:, :, j] = np.eye(D) * min_dist

    for i in np.arange(n_iters):
        for k in np.arange(K):
            covariances[:, :, k] = regularize_cov(covariances[:, :, k], epsilon)
        logLikelihood, gamma = EStep(means, covariances, weights, data)
        weights, means, covariances, logLikelihood = MStep(gamma, data)

    return [weights, means, covariances]
