"""

"""

import sys
import os
from pathlib import Path

# set shared scripts dir for both msit and stroop up one level
curr_dir = os.getcwd()
pars = Path(curr_dir).parents
par_dir = pars[0]
sys.path.insert(0, str(par_dir))

import jr_funcs as jr
from bg_cb_funcs import *
from config import *
from task_blocks import *


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


task = 'msit'


# calculate eigenvec cen for all subjects, save
# save_eigen_cen(subj_lst, task=task, region='cb') # run this line once
# save_eigen_cen(subj_lst, task=task, region='bg') # run this line once


eigen_cen_cb_allsub = np.load(f'{main_dir}{inter_path}{task}/eigen_cen_cb_allsub_{task}.npy')
eigen_cen_bg_allsub = np.load(f'{main_dir}{inter_path}{task}/eigen_cen_bg_allsub_{task}.npy')

eigen_cen_cb_allsub_smooth = np.array(list(map(lambda i: gaussian_filter(i, sigma=1), eigen_cen_cb_allsub)))
eigen_cen_cb_allsub_smooth_z = np.array(list(map(lambda i: stats.zscore(i), eigen_cen_cb_allsub_smooth)))
eigen_cen_bg_allsub_smooth = np.array(list(map(lambda i: gaussian_filter(i, sigma=1), eigen_cen_bg_allsub)))
eigen_cen_bg_allsub_smooth_z = np.array(list(map(lambda i: stats.zscore(i), eigen_cen_bg_allsub_smooth)))

eigen_cen_cb_avg = np.average(eigen_cen_cb_allsub, axis=0)
eigen_cen_bg_avg = np.average(eigen_cen_bg_allsub, axis=0)
eigen_cen_cb_avg_smooth = gaussian_filter(eigen_cen_cb_avg, sigma=1)
eigen_cen_bg_avg_smooth = gaussian_filter(eigen_cen_bg_avg, sigma=1)

eigen_cen_cb_avg_smooth_z = stats.zscore(eigen_cen_cb_avg_smooth)
eigen_cen_bg_avg_smooth_z = stats.zscore(eigen_cen_bg_avg_smooth)
np.save(f'{main_dir}{inter_path}{task}/eigen_cen_cb_avg_smooth_z_{task}.npy', eigen_cen_cb_avg_smooth_z)
np.save(f'{main_dir}{inter_path}{task}/eigen_cen_bg_avg_smooth_z_{task}.npy', eigen_cen_bg_avg_smooth_z)


print(f'\nEigenvector centrality corr_coef, BG and CB: ', \
            np.corrcoef(eigen_cen_cb_avg_smooth_z, eigen_cen_bg_avg_smooth_z)[0][1], '\n') # [1][0]

