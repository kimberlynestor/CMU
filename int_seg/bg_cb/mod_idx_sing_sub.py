"""
Name: Kimberly Nestor
Date: 03/2022
Project: int_seg, bg_cb
Description: This program

Participation coefficient (signed): https://tinyurl.com/2p89y43a
Modularity maximization: https://tinyurl.com/4pazk7bz
Mod Max e.g.: https://tinyurl.com/3jkf2zjj
"""

import os
from os.path import join as opj
import sys

import bct
import netneurotools
from netneurotools import modularity as mod

from nilearn.image import load_img
# import tensorflow as tf # used pip instead of conda
# import torch # used conda

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jr_funcs as jr

# np.set_printoptions(threshold=sys.maxsize)


# data path and subjects
dir = '/home/kimberlynestor/gitrepo/int_seg/data/'
d_path = 'pip_edge_ts/shen/task-msit/'
subj_file = 'sub-4285_ses-01_task-msit_space-MNI152NLin2009cAsym_desc-edges_bold.nii.gz'

# get efc for single subject
efc_vec = jr.extract_edge_ts(opj(dir, d_path, subj_file))
efc_mat = np.squeeze(load_img(opj(dir, d_path, subj_file)).get_fdata())
efc_mat_t = efc_mat.T

# shen atlas
# shen_comm = load_img(opj(dir, 'atlas/shen_2mm_268_parcellation.nii.gz')).get_fdata()

# mod max, single timepoint, single subj
# mod = mod.consensus_modularity(efc_mat.T[0]) # does not allow neg weights
ci, Q = bct.community_louvain(efc_mat_t[0], gamma=1.5, B='negative_sym') ## timepoint 1, 268x268# 1.5, 'negative_asym'

# mod max, all timepoint, single subj
# try with B = average resting state matrix
mod_max_all = list(map(lambda x: bct.community_louvain(x, gamma=1.5, B='negative_sym'), \
                    efc_mat_t))
ci_all = list(map(lambda xx:xx[0], mod_max_all))
q_all = list(map(lambda xx:xx[1], mod_max_all))


# plot mod max, all timepoint
plt.plot(np.arange(0, 280*2, 2), q_all, linewidth=1)
plt.xticks(np.arange(0, 280*2, 60))
plt.xlabel("Time (s)", size=11, fontname="serif")
plt.ylabel("Modularity (Q)", size=11, fontname="serif")
plt.savefig('sing_sub_mod_qall.png', dpi=300)
plt.show()

sys.exit()




#######
# part coeff, single subj one timepoint, ci from mod max
p_coef = bct.participation_coef_sign(efc_mat_t[50], ci) # this words for ci in timepoint 1
# p_coef = bct.participation_coef_sign(efc_mat_t[80], ci_all[80]) # but appropriate ci for timepoint doesn't work

# part coeff, single subject all timepoint
p_coef_all = list(map( lambda x: np.average(bct.participation_coef_sign(x, ci)), efc_mat_t))
# p_coef_all = list(map( lambda x: bct.participation_coef_sign(x[0], x[1]), zip(efc_mat_t, ci_all))) # all timepoint doesn't work


# plot part coeff, all timepoint
plt.plot(np.arange(0, 280*2, 2), p_coef_all, linewidth=1)
plt.xticks(np.arange(0, 280*2, 60))
plt.xlabel("Time (s)", size=11, fontname="serif")
plt.ylabel("Participation coefficient (integration)", size=11, fontname="serif")
plt.savefig('sing_sub_p_coef_all.png', dpi=300)
plt.show()

