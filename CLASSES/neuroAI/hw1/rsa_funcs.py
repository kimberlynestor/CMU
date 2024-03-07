"""
Name: Kimberly Nestor
Class: CMU 10-733 NeuroAI
Date: 2/28/24
Description: Helper functions to compute RDM and RSA values to compare brain matrices to model feature spaces.
"""

import numpy as np
from scipy.stats import spearmanr


# get triangle of symmetric matrix
lower_t = lambda rdm : rdm[np.tril_indices(rdm.shape[0], -1)]


def rsa_mats(mat1, mat2):
    """"func to compute representational similarity analysis
    value of two matrices i.e. similarity of two mats"""
    # compute representational dissimilarity matrices
    rdm1 = 1 - np.corrcoef(mat1)
    rdm2 = 1 - np.corrcoef(mat2)

    # get mat triangle
    lt_rdm1 = lower_t(rdm1)
    lt_rdm2 = lower_t(rdm2)

    # get spearman corr of rdm triangles
    rsa = spearmanr(lt_rdm1, lt_rdm2)[0]
    return(rsa)

