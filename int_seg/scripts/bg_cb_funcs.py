"""


"""

import os
import sys

import itertools

import numpy as np
import math
import pandas as pd
from scipy import stats
from scipy.stats import norm
from statsmodels.tsa.api import VAR
import impyute as impy

import jr_funcs as jr
from config import *

import bct
import networkx as nx



def node_info(region):
    """This function takes as input the desired region ('cort', 'cb', 'bg', 'thal'),
    returns the shen indices and the network each node is assigned to."""
    # shen parcels region assignment - brain areas from Andrew Gerlach
    df_net_assign = pd.read_csv(opj(main_dir, dep_path, 'shen_268_parcellation_networklabels_mod.csv'))
    ci = df_net_assign.iloc[:,1].values

    if region == 'cort':
        # cortical nodes only
        assign = np.array(list(filter(lambda i: i[1] != 4, df_net_assign.values)))
        reg_idx = [i[0] for i in assign]
        ci_reg = ci[reg_idx]
        return(list(reg_idx), list(ci_reg))
    elif region == 'cb':
        # cb nodes only
        df_net_assign_reg = df_net_assign[(df_net_assign['anat_reg'].str.contains('Cerebellum')) \
                                         | (df_net_assign['anat_others'].str.contains('Cerebellum')) \
                                         | (df_net_assign['anat_covered'].str.contains('Cerebellum'))]
        reg_idx = df_net_assign_reg.iloc[:, 0].values
        ci_reg = ci[reg_idx]
        return(list(reg_idx), list(ci_reg))
    elif region =='bg':
        # bg nodes only
        # df_net_assign_exc = df_net_assign.drop(df_net_assign.\
        #                         iloc[list(itertools.chain(c_idx, cb_idx))].index)
        df_net_assign_reg = df_net_assign[(df_net_assign['anat_reg'].str.contains('Caudate')) \
                                         | (df_net_assign['anat_reg'].str.contains('Putamen')) \
                                         | (df_net_assign['anat_reg'].str.contains('SN')) \
                                         | (df_net_assign['anat_reg'].str.contains('N_Acc'))]  # no pallidum ?
        reg_idx = df_net_assign_reg.iloc[:, 0].values
        ci_reg = ci[reg_idx]
        return(list(reg_idx), list(ci_reg))
    elif region == 'thal':
        # thal nodes only
        df_net_assign_reg = df_net_assign[(df_net_assign['anat_reg'].str.contains('Thal'))]
        reg_idx = df_net_assign_reg.iloc[:, 0].values
        ci_reg = ci[reg_idx]
        return(list(reg_idx), list(ci_reg))


def save_mod_idx(subj_lst, task):
    """
    This function takes as input the subject list and the task
    ('stroop', 'msit', 'rest') and saves ouput files with modularity at
    all timepoints for all subjects. Only for cortex modularity.
    """
    out_name = 'subjs_all_net_cort_q_'
    c_idx = node_info('cort')[0]
    q_subj_lst = []
    for subj in subj_lst:
        efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)  # all networks
        efc_mat_cort = np.array(list(map(lambda frame: \
                            np.array(list(map(lambda col: col[c_idx], \
                                frame[c_idx]))), efc_mat)))  # cortex only

        mod_all_frs = np.array(list(map(lambda mat: bct.community_louvain(mat, \
                                                    gamma=1.5, B='negative_sym'),
                                        efc_mat_cort)), dtype=object)
        ci_all_frs = np.array(list(map(lambda x: x[0], mod_all_frs)))
        q_all_frs = np.array(list(map(lambda x: x[1], mod_all_frs)))
        q_subj_lst.append(q_all_frs)
    np.save(f'{main_dir}IntermediateData/{task}/{out_name}{task}.npy', q_subj_lst)
    return


def mat_adj_cort(efc_mat, reg_nodes, thres=0, cort_nodes=node_info('cort')[0]):
    """Takes a matrix as input and a set of region nodes and returns a
    matrix of that region nodes only with connections to cortex. thres=float between 0,1"""
    nodes_cort_connec = target_nodes(efc_mat, reg_nodes, thres, cort_nodes)
    efc_mat_lim = [list(map(lambda col: col[c_idx], frame[c_idx])) \
                   for frame, c_idx in zip(efc_mat, nodes_cort_connec)]
    efc_mat_lim = np.array(list(map(lambda x: np.array(x), efc_mat_lim)))
    return(efc_mat_lim)


def target_nodes(efc_mat, reg_nodes, thres=0, cort_nodes=node_info('cort')[0]):
    """Gets subcortical region nodes that are connected to cortex."""
    adj_mat = list(map(lambda i: np.where(np.abs(i) > thres, 1, 0), efc_mat)) # 0.2, 0.6
    adj_mat_reg = list(map(lambda i: i[reg_nodes], adj_mat))

    cort_bool = list(map(lambda ii: list(map(lambda i: any(i[cort_nodes] == 1), ii)), adj_mat_reg))
    nodes_cort_connec = list(map(lambda i: list(itertools.compress(reg_nodes, i)), cort_bool))
    return(nodes_cort_connec)


def cross_corr_vec(x,y, lags=16, pval=False):
    """
    This function shifts vectors x and y to determine cross correlation.
    max lags = len(lst)-2
    """
    lags = list(range(-lags,lags+1))
    lag_corr = []
    lag_pval = []
    for i in lags:
        if i < 0:
            corr = stats.pearsonr(x[:i], y[np.abs(i):])[0]
            ttest = stats.ttest_ind(x[:i], y[np.abs(i):])[1]
            lag_corr.append(corr)
            lag_pval.append(ttest)
        elif i > 0:
            corr = stats.pearsonr(x[i:], y[:-i])[0]
            ttest = stats.ttest_ind(x[i:], y[:-i])[1]
            lag_corr.append(corr)
            lag_pval.append(ttest)
        elif i == 0:
            corr = stats.pearsonr(x, y)[0]
            ttest = stats.ttest_ind(x, y)[1]
            lag_corr.append(corr)
            lag_pval.append(ttest)
    if pval == False:
        return(lag_corr, lags)
    else:
        return(lag_pval)



def var_allsub(cort_mod, cb_eigen, bg_eigen, lag=1):
    """
    This function takes cortical modularity, cb and bg eigen vector centrality
    for all subjects. then implements vector autoregrssion and returns coefficients,
    std error for each equation and overall correlations (respectively in output).
    returns:
    coeff = mod, cb, bg per subject
    corr = mod x cb, mod x bg, cb x bg per subject
    """
    coeff_lst = []
    corr_lst = []
    for sub_mod, sub_cb, sub_bg in zip(cort_mod, cb_eigen, bg_eigen):
        df_avg_blks = pd.DataFrame(sub_mod, columns=['mod_idx'])
        df_avg_blks['eigen_cb'] = sub_cb
        df_avg_blks['eigen_bg'] = sub_bg

        model = VAR(df_avg_blks)
        results = model.fit(lag)

        # get VAR model output matrix
        mdl_coeff_mat = results.params.values[-3:]
        mdl_corr_mat = results.resid_corr

        # get output for each equation
        mdl_coeff = np.diagonal(mdl_coeff_mat)
        mdl_corr = mdl_corr_mat[np.tril_indices(mdl_corr_mat.shape[0], k=-1)]

        # append to lsts
        coeff_lst.append(mdl_coeff)
        corr_lst.append(mdl_corr)

    return(np.array([coeff_lst, corr_lst]))


def impy_params(param_lst, imp_num=1, epoch=10000):
    """
    This function takes a list of params for all subjects from var_allsub.
    does imputation using EM algorithm over 1000 epochs to stabilise and
    returns single value params representative of the subjects set.
    """
    impy_mdl_param = []
    for ii in range(len(param_lst)):
        # set param for imputation and fill with nan
        param = param_lst[ii]
        fill_nan = np.array(list(map(lambda i: np.pad(i, (0, imp_num), \
                            mode='constant', constant_values=np.nan), param.T)))

        # perform EM on imputations to stabilise
        em_lst = []
        for i in range(epoch):
            em = impy.em(fill_nan.T, loops=1000)
            em_lst.append(em[-imp_num])
        stab_em = np.mean(np.array(em_lst), axis=0)
        impy_mdl_param.append(stab_em)
    return (np.array(impy_mdl_param))


def save_eigen_cen(subj_lst, task, region):
    """
    This function takes as input the subj_lst, task and brain region and computes
    eigenvector centrality for the indicated region with connections to cortex.
    Multiple nodes are averaged together, returns eigen values for the timescales.
    """
    out_name = 'eigen_cen_'
    eigen_lst = []
    print(task, region)
    for subj in subj_lst:
        print(subj)
        efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)

        evec_cen = list(map(lambda mat: nx.eigenvector_centrality(\
                    nx.from_numpy_matrix(mat), weight='weight', max_iter=1000), efc_mat))

        targets = target_nodes(efc_mat, node_info(f'{region}')[0])
        eigen = list(map(lambda i: list(map(lambda ii : i[1][ii], i[0])) , zip(targets, evec_cen)))

        avg_tr = np.average(eigen, axis=1)
        eigen_lst.append(avg_tr)

    np.save(f'{main_dir}IntermediateData/{task}/eigen_cen_{region}_allsub_{task}.npy', eigen_lst)
    return


def bayes_fact(null, alt):
    """function takes in two datasets and returns BIC values
    for both datasets and bayes factor value comparing the two.
    based on: Wagenmakers, 2007 - A practical solution to the pervasive problems of p values"""
    # calculate bic for stroop
    data_0 = null
    m,s = norm.fit(data_0)
    log_ll = np.log(np.product(norm.pdf(data_0,m,s)))
    bic_0 = np.log(-2*log_ll) + np.log(len(data_0))
    bic_0 = round(bic_0, 2)

    # calculate bic for msit
    data_1 = alt
    m, s = norm.fit(data_1)
    log_ll = np.log(np.product(norm.pdf(data_1, m, s)))
    bic_1 = np.log(-2 * log_ll) + np.log(len(data_1))
    bic_1 = round(bic_1, 2)

    # calculate bayes factor, 01
    b_fac = math.exp((bic_1 - bic_0)/2)
    b_fac = round(b_fac, 2)

    return(bic_0, bic_1, b_fac)


"""This function gets the listed statistic for all lags run on the statsmodel 
'grangercausalitytests' funtion. output: stat, pval, df (df_denom, df_num)"""
get_gc_stat = lambda gc_dict, stat: [gc_dict[ii][0][stat] for ii in list(map(lambda i:i, gc_dict))]
