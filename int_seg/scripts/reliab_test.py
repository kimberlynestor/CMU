"""


"""

import sys
import os


import jr_funcs as jr
from bg_cb_funcs import *
from task_blocks import *
from config import *

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt


# lagged vector comparison
task = 'stroop'
cort_bg_diff_stroop = np.load(f'{main_dir}IntermediateData/{task}/cross_corr_cort_bg_peak_diff_{task}.npy')
cort_cb_diff_stroop = np.load(f'{main_dir}IntermediateData/{task}/cross_corr_cort_cb_peak_diff_{task}.npy')
cb_bg_diff_stroop = np.load(f'{main_dir}IntermediateData/{task}/cross_corr_cb_bg_peak_diff_{task}.npy')

task = 'msit'
cort_bg_diff_msit = np.load(f'{main_dir}IntermediateData/{task}/cross_corr_cort_bg_peak_diff_{task}.npy')
cort_cb_diff_msit = np.load(f'{main_dir}IntermediateData/{task}/cross_corr_cort_cb_peak_diff_{task}.npy')
cb_bg_diff_msit = np.load(f'{main_dir}IntermediateData/{task}/cross_corr_cb_bg_peak_diff_{task}.npy')



cort_bg_test = np.array(list(map(lambda tup: stats.spearmanr(tup[0], tup[1]).statistic, zip(cort_bg_diff_stroop, cort_bg_diff_msit))))
cort_cb_test = np.array(list(map(lambda tup: stats.spearmanr(tup[0], tup[1]).statistic, zip(cort_cb_diff_stroop, cort_cb_diff_msit))))
cb_bg_test = np.array(list(map(lambda tup: stats.spearmanr(tup[0], tup[1]).statistic, zip(cb_bg_diff_stroop, cb_bg_diff_msit))))



plt.scatter(np.ones(len(cort_bg_test)), cort_bg_test)
plt.scatter(np.ones(len(cort_bg_test))*2, cort_cb_test)
plt.scatter(np.ones(len(cort_bg_test))*3, cb_bg_test)
plt.axhline(0)
plt.show()


# vector to vector comparison
task = 'stroop'
q_allsub_stroop = np.load(f'{main_dir}IntermediateData/{task}/subjs_all_net_cort_q_{task}.npy')
eigen_cb_allsub_stroop = np.load(f'{main_dir}IntermediateData/{task}/eigen_cen_cb_allsub_{task}.npy')
eigen_bg_allsub_stroop = np.load(f'{main_dir}IntermediateData/{task}/eigen_cen_bg_allsub_{task}.npy')

task = 'msit'
q_allsub_msit = np.load(f'{main_dir}IntermediateData/{task}/subjs_all_net_cort_q_{task}.npy')
eigen_cb_allsub_msit = np.load(f'{main_dir}IntermediateData/{task}/eigen_cen_cb_allsub_{task}.npy')
eigen_bg_allsub_msit = np.load(f'{main_dir}IntermediateData/{task}/eigen_cen_bg_allsub_{task}.npy')


cort_test = np.array(list(map(lambda tup: stats.spearmanr(tup[0], tup[1]).statistic, zip(q_allsub_stroop, q_allsub_msit))))
cb_test = np.array(list(map(lambda tup: stats.spearmanr(tup[0], tup[1]).statistic, zip(eigen_cb_allsub_stroop, eigen_cb_allsub_msit))))
bg_test = np.array(list(map(lambda tup: stats.spearmanr(tup[0], tup[1]).statistic, zip(eigen_bg_allsub_stroop, eigen_bg_allsub_msit))))



plt.scatter(np.ones(len(cort_test)), cort_test)
plt.scatter(np.ones(len(cb_test))*2, cb_test)
plt.scatter(np.ones(len(bg_test))*3, bg_test)
plt.axhline(0)
plt.show()
