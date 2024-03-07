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
all_mdl_embeds = [np.load(f'{new_embed_save_path}{i}') for i in sorted(os.listdir(new_embed_save_path))]
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
story_param_lst = []
for story in range(len((train_mdl_embeds))):
    print(f'story = {story}')
    
    subj_param_lst = []
    for subj in range(train_mdl_embeds[story].shape[0]):
        print(subj)
        ridge_train = CV_ridge.fit(train_mdl_embeds[story][subj], train_fmri_mats[story][subj]) # X, Y
        subj_param_lst.append(ridge_train)
        np.save(f'{mdl_output_dir}{story_names[story]}_coeffs_{subj+1}.npy', ridge_train[0])
        np.save(f'{mdl_output_dir}{story_names[story]}_lambda_{subj+1}.npy', ridge_train[1])
    story_param_lst.append(subj_param_lst)


# predict fmri data from test mdl embed, save for each subj
print(test_mdl_embeds.shape)
for i in range(len(test_mdl_embeds)):
    predict_fmri = CV_ridge.predict(test_mdl_embeds[i])
    print('test shape', test_mdl_embeds[i].shape)
    print('pred shape', predict_fmri.shape)
    np.save(f'{pred_output_dir}{story_names[-1]}_pred{i+1}.npy', predict_fmri)


# training data names
train_data_names = sorted(os.listdir(mdl_output_dir))
train_data_names_coeffs = [i for i in train_data_names if 'coeffs' in i]
train_data_names_lam = [i for i in train_data_names if 'lambda' in i]

# fmri predictions names
fmri_pred_names = sorted(os.listdir(pred_output_dir))


#### make brain plots in nilearn
## plot 1 = correlation of encoding prediction and regularization coefficients - for each subj test data (3), for each story coeffs(8*3)
for i in range(len(fmri_pred_names)):
    # load fmri test predictions
    fmri_pred = np.load(f'{pred_output_dir}{fmri_pred_names[i]}')
    for ii in range(len(train_data_names_coeffs)):
        # load training weights
        mdl_coeffs = np.load(f'{mdl_output_dir}{train_data_names_coeffs[ii]}')
        m_save_name = train_data_names_coeffs[ii].split('.')[0]
        # compute correlation
        cor = np.corrcoef(fmri_pred, mdl_coeffs)

        # plot 2d cortex flat map - in nilearn, save fig
        plotting.plot_surf(hcp.mesh.flat, hcp.cortex_data(np.ravel(cor)), colorbar=True, cmap='magma', vmin=-0.27, vmax=0.7) #
        plt.xlabel('pred and coeff\n (correlation)', fontsize=10)
        plt.title(f'{m_save_name}\npred{i+1}', fontsize=6)
        plt.savefig(f'{output_figs_dir}{m_save_name}_pred{i+1}.png', dpi=500)
        # plt.show()
        plt.close()


## plot 2 = 1 subj, 1 ft space (story), brain plot of optimal regularization parameter (lambda)
mdl_lambda = np.load(f'{mdl_output_dir}{train_data_names_lam[0]}')
l_save_name = train_data_names_lam[0].split('.')[0]

# np.set_printoptions(threshold=sys.maxsize)

# plot 2d cortex flat map - in nilearn, save fig
# plotting.plot_surf(hcp.mesh.flat, hcp.cortex_data(np.repeat(mdl_lambda, mdl_lambda.shape[0], axis=0)), colorbar=True, cmap='viridis') # , vmin=-0.27, vmax=0.7
plotting.plot_surf(hcp.mesh.flat, hcp.cortex_data(np.array(list(mdl_lambda)*mdl_lambda.shape[0])), colorbar=True, cmap='viridis') # , vmin=-0.27, vmax=0.7
plt.xlabel('optimal\n lambda', fontsize=10)
plt.title(f'{l_save_name}', fontsize=7)
plt.savefig(f'{output_figs_dir}{l_save_name}.png', dpi=500)
# plt.show()
plt.close()




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
