"""
Name: Kimberly Nestor
Class: CMU 10-733 NeuroAI
Date: 2/28/24

Description: Compares RSA from fMRI brain data to three other feature spaces:
            RoBERTa model embeddings, ridge regression coefficients and optimized lambda values.

Disclosures: I've computed RDM and RSA values before for the NeuroMatch project summer 2022 and
referred to those files to complete this portion of the HW. Links below show code and project details.
https://drive.google.com/file/d/1yhMggax9L0c9G4piqHAKxgRZhYd_J4u0/view?usp=sharing
https://docs.google.com/document/d/1JqQcTRTu61fV4oDCtXjoasS2X78F2imO/edit?usp=sharing&ouid=105543551381154944670&rtpof=true&sd=true
https://drive.google.com/file/d/1xEWJfoZKs9xgl4vEYDG0LrbOt7g7jHLf/view?usp=sharing
https://drive.google.com/file/d/1YnlRrlVmg-j3ufd-20Rlh_g3gGkY193C/view?usp=sharing
"""

import os
import sys

import numpy as np
from rsa_funcs import *

import matplotlib.pyplot as plt


# dir info for data
curr_dir = os.getcwd()
new_mat_save_path = curr_dir + '/resize_fmri_mats/'
new_embed_save_path = curr_dir + '/resize_norm_embeddings/'
pred_output_dir = curr_dir + '/fmri_pred_output/'
output_figs_dir = curr_dir + '/output_figs/' # UTS01, UTS02, UTS03


story = 'wheretheressmoke'

# load resize normalized fmri matrice
fmri_mat_sub1 = np.load(new_mat_save_path + story + '.npy')[0] # UTS01

# load resize normalized mode embeds - last three layers
mdl_embeds = np.load(new_embed_save_path + story + '.npy')

# load predicted fmri output for UTS01
pred_fmri_mat_sub1 = np.load(pred_output_dir + story + '_pred1.npy') # UTS01


# get rsa to compare fmri data to last three layers of RoBERTa and fmri encoding prediction
rsa_lst = []
for ll_em in mdl_embeds:
    rsa_lst.append(rsa_mats(fmri_mat_sub1, ll_em))
rsa_lst.append(rsa_mats(fmri_mat_sub1, pred_fmri_mat_sub1))

# print(rsa_lst)

# make barplot of rsa values
fig, ax = plt.subplots()
ax.bar(range(len(rsa_lst)), rsa_lst, color='cyan')
ax.set_xticks(range(len(rsa_lst)), labels = ['mdl_layer_11', 'mdl_layer_12', 'mdl_layer_13', 'fmri_prediction'])
for i, j in enumerate(rsa_lst):
    ax.text(i, j+0.02, str(round(j, 2)), c='k', fontweight='bold', verticalalignment='center')
plt.xlabel('Comparison matrix', fontweight='bold')
plt.ylabel('RSA value', fontweight='bold')
plt.title('RSA of wheretheressmoke fMRI data to \nRoBERTa layers and fMRI prediction')
plt.tight_layout()
plt.savefig(f'{output_figs_dir}rsa_bar_fmri_mdlft_pred.png', dpi=500)
plt.show()


