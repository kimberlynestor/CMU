"""
Name: Kimberly Nestor
Class: CMU 10-733 NeuroAI
Date: 2/28/24

Description: Makes encoding model from fMRI story data and RoBERTa model embeddings.

Disclosures: I used the cross_validated_ridge.py module for training ridge regression,
             provided by Prof. Leila Wehbe.
"""

import os
import sys
import itertools

import numpy as np
import matplotlib.pyplot as plt

import torch
import cortex
import hcp_utils as hcp
from cross_validated_ridge import SimpleCVRidgeDiffLambda
from nilearn.image import load_img
from nilearn import plotting
import h5py

from fmri_funcs import *

import warnings
warnings.filterwarnings('ignore')


# dir info for data
curr_dir = os.getcwd()
dir_depth = '/recordings/ds003020/derivative'
story_save_path = curr_dir + '/clean_stories/'
output_figs_dir = curr_dir + '/output_figs/' # UTS01, UTS02, UTS03
new_embed_save_path = curr_dir + '/resize_norm_embeddings/'
new_mat_save_path = curr_dir + '/resize_fmri_mats/'
mask_dir = '/pycortex-db/UTS01/transforms/UTS01_auto/'
mdl_output_dir = curr_dir + '/mdl_train_output/'
pred_output_dir = curr_dir + '/fmri_pred_output/'

story_names = sorted(os.listdir(story_save_path))

# mask = load_img(curr_dir + dir_depth + mask_dir + 'mask_cortical.nii.gz').get_fdata()


#### 1.3 Make encoding model
# load embeddings
all_mdl_embeds = [np.load(f'{new_embed_save_path}{i}') for i in os.listdir(new_embed_save_path)]
train_mdl_embeds = all_mdl_embeds[:-1]
test_mdl_embeds = all_mdl_embeds[-1]

# load fmri mats
all_fmri_mats = [np.load(f'{new_mat_save_path}{i}') for i in os.listdir(new_mat_save_path)]
train_fmri_mats = all_fmri_mats[:-1]
test_fmri_mats = all_fmri_mats[-1]


# do cross validated ridge regression training, selected optimizer = sag
# where X is input features (nTRs*features)
# Y is output (nTRs*voxels) if you're doing an encoding model
CV_ridge = SimpleCVRidgeDiffLambda(solver = 'sag', n_splits=10) # init solver = 'auto'

# train and save mdl weights and lambdas
"""
story_param_lst = []
for story in range(len((train_mdl_embeds))):
    # print(f'story = {story}')
    subj_param_lst = []
    for subj in range(train_mdl_embeds[story].shape[0]):
        # print(subj)
        ridge_train = CV_ridge.fit(train_mdl_embeds[story][subj], train_fmri_mats[story][subj]) # X, Y
        subj_param_lst.append(ridge_train)
        # np.save(f'{mdl_output_dir}{story_names[story]}_coeffs_{subj+1}.npy', ridge_train[0])
        # np.save(f'{mdl_output_dir}{story_names[story]}_lambda_{subj+1}.npy', ridge_train[1])
    story_param_lst.append(subj_param_lst)
"""

# predict fmri data from test mdl embed, save for each subj
"""
for i in range(len(test_mdl_embeds)):
    predict_fmri = CV_ridge.predict(test_mdl_embeds[i])
    # np.save(f'{pred_output_dir}{story_names[-1]}_pred{i+1}.npy', predict_fmri)
"""


sys.exit()

#### make brain plots in pycortex
## plot 1 = correlation of encoding prediction and regularization coefficients - for each subj test (3), for each story (8)
# compute_correlation(Test_Y, Pred)
# cor = np.corrcoef(predict_fmri, predict_fmri)

## plot 2 = 1 subj, 1 ft space (story), brain plot of optimal regularization parameter (lambda)




# lz resize to large n voxels
fmri_data_dir = curr_dir + dir_depth + '/preprocessed_data/UTS01/adollshouse.hf5' # UTS01, UTS02, UTS03
fmri_dict = h5py.File(fmri_data_dir, 'r')
fmri_mat = np.array(fmri_dict['data'])
vox_sz = fmri_mat.shape[-1]


# 2d cortex flat map
plotting.plot_surf(hcp.mesh.flat, hcp.cortex_data(fmri_mat[0]), colorbar=True, cmap='magma')
plt.xlabel('pred and coeff\n (correlation)')
plt.show()

# inflated cortex plot
# plotting.plot_surf(hcp.mesh.inflated)
# plotting.plot_surf(hcp.mesh.inflated, hcp.cortex_data(fmri_mat[0]), bg_map=hcp.mesh.sulc, colorbar=True, cmap='magma')
# plt.show()



sys.exit()


# plot random flat map in pycortex
np.random.seed(11)
volume = cortex.Volume.random(subject='S1', xfmname='retinotopy', cmap='magma')
cortex.quickflat.make_figure(volume, with_rois=False)
# plt.savefig(f'{output_figs_dir}pycortex_random_flatmap.png')
plt.tight_layout()
# plt.show()
plt.close()

# cmap - gradient options = magma, plasma, viridis, summer, cool_r
# cmap - two opposites = PiYG, PRGn, RdGy

# worked initially in pycortex but now errors out
# cortex.quickflat.make_figure(predict_fmri[0])
# plt.show()
