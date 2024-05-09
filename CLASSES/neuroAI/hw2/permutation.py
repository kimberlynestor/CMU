# code provided by Prof Leila Wehbe, Q2 help

import numpy as np

def corr(x, y):
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    return np.dot(x_centered,y_centered)/(np.linalg.norm(x_centered)*(np.linalg.norm(y_centered)))

def single_voxel_permutation_test(voxel_predictions,voxel_responses,shuffles=5000):
    '''performs a permuation test against the null hypothesis that 
        model responses to stimulus do not predict voxel responses to that stimulus'''
    model_corr = corr(voxel_predictions,voxel_responses)
    g_than_model_corr_counts = 0
    copy_responses = np.copy(voxel_responses)
    for _ in range(shuffles):
        np.random.shuffle(copy_responses)
        c = corr(voxel_predictions,copy_responses)
        if c > model_corr:
            g_than_model_corr_counts += 1
    return g_than_model_corr_counts/shuffles

def fdr_corrected_significant_ps(ps,alpha):
    '''returns the indices of ps that are significant under fdr correction'''
    sorted_ps = np.sort(ps)
    sorted_indices = np.argsort(ps)
    comparison = alpha*(1+np.arange(len(sorted_ps)))/len(sorted_ps)
    return sorted_indices[sorted_ps < comparison]

