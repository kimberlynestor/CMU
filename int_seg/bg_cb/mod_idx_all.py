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
import jr_funcs as jr

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import math
import matplotlib.pyplot as plt


from nilearn.image import load_img
# import tensorflow as tf # used pip instead of conda
import torch # used conda
import zarr

# np.set_printoptions(threshold=sys.maxsize)
# pd.set_option('display.max_rows', None)


# data path and subjects
main_dir = '/home/kimberlynestor/gitrepo/int_seg/data/'
d_path = 'pip_edge_ts/shen/'
dep_path = 'depend/'

# get subject list and task conditions
subj_lst = np.loadtxt(opj(main_dir, dep_path, 'subjects_intersect_motion_035.txt'))
df_task_events = pd.read_csv(opj(main_dir, dep_path, 'task-stroop_events.tsv'), sep="\t")

"""
# get efc mat for all subjects
for task in ['stroop', 'msit', 'rest']: # mem issue with 100% data
    efc_mat_allsub = jr.get_efc_trans(opj(main_dir, d_path), task, \
                                      subj_lst [0:math.floor(len(subj_lst)*0.7)] ) # 70% data
    np.save(f'subjs_efc_mat_{task}.npy', efc_mat_allsub)
"""

# cause of mem had to save tasks individually
task = 'stroop'
# efc_mat_allsub = jr.get_efc_trans(opj(main_dir, d_path), task, \
#                                   subj_lst[0:math.floor(len(subj_lst)*0.75)])
# np.save(f'subjs_efc_mat_{task}.npy', efc_mat_allsub)


# load saved npy with task coalesce matrices
mat_stroop_all = np.load('subjs_efc_mat_stroop.npy')

# do mod max on all subjs
# mod_max_stroop_all = np.array(list(map(lambda subj: np.array(list(map(lambda mat: \
#                         bct.community_louvain(mat, gamma=1.5, B='negative_sym'), \
#                             subj)), dtype=object), mat_stroop_all)), dtype=object)
# np.save(f'subjs10_mod_max_{task}.npy', mod_max_stroop_all)
# np.save(f'subjs_all_mod_max_{task}.npy', mod_max_stroop_all)


# load mod max for all subjs
# mod_max_stroop_all = np.load('subjs10_mod_max_stroop.npy', allow_pickle=True)
mod_max_stroop_all = np.load('subjs_all_mod_max_stroop.npy', allow_pickle=True)

# separate out communities and index for all subjs
# ci_allsub = np.array(list(map(lambda subj: np.array(list(map(lambda x:x[0], subj))), \
#                                                 mod_max_stroop_all)))
q_allsub = np.array(list(map(lambda subj: np.array(list(map(lambda x:x[1], subj))), \
                                                mod_max_stroop_all)))

# avg q across subjs
q_avg = np.average(q_allsub, axis=0)

# smooth q using moving average
# df_q = pd.DataFrame(q_avg, columns=['Q_avg'])
# df_q['Q_smooth'] = df_q['Q_avg'].rolling(2).sum()
# q_smooth = np.nan_to_num( df_q['Q_smooth'].to_numpy() )

# smooth q using gaussian filter
q_smooth = gaussian_filter(q_avg, sigma=2.5)


# plot mod max, all timepoint, all subjs
plt.plot(np.arange(0, 280*2, 2), q_avg, linewidth=1)
plt.xticks(np.arange(0, 280*2, 60))
plt.xlabel("Time (s)", size=11, fontname="serif")
plt.ylabel("Modularity (Q)", size=11, fontname="serif")
plt.savefig('allsub_mod_qall.png', dpi=300)
plt.show()

# plot mod max, all timepoint, all subjs - smooth
plt.plot(np.arange(0, 280*2, 2), q_smooth, linewidth=1)
plt.xticks(np.arange(0, 280*2, 60))
plt.xlabel("Time (s)", size=11, fontname="serif")
plt.ylabel("Modularity (Q)", size=11, fontname="serif")
plt.savefig('allsub_mod_qall_smooth_sig2p5.png', dpi=300)
plt.show()