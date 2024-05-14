"""
Name: Kimberly Nestor
Class: CMU 10-733 NeuroAI Project
Date: 5/12/24

Description: Used embeddings from ResNet18 for original, foreground only and background only images to MEG data.

Resources:
https://elifesciences.org/articles/82580
https://things-initiative.org/
https://things-initiative.org/projects/things-images/
https://plus.figshare.com/collections/THINGS-data_A_multimodal_collection_of_large-scale_datasets_for_investigating_object_representations_in_brain_and_behavior/6161151
"""


import os
import sys
from tqdm import tqdm

from PIL import Image
import torch
import mne
import h5py
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dir info for data
curr_dir = os.getcwd()
stim_img_path = curr_dir + '/images/'
nb_img_path = curr_dir + '/images_nb/'
bg_img_path = curr_dir + '/images_bg/'

nb_mask_path = curr_dir + '/nb_mask/'
nf_img_path = curr_dir + '/images_nf/'

meg_path = curr_dir + '/THINGS-MEG/preprocessed/'
embed_path = curr_dir + '/embeddings/'

output_path = curr_dir + '/output/'

# load embeddings
org_img_embed = np.load(f'{embed_path}org_img_embed.npy')
nb_img_embed = np.load(f'{embed_path}nb_img_embed.npy')
nf_img_embed = np.load(f'{embed_path}nf_img_embed.npy')

embed_shape = org_img_embed.shape # [2048, 392]

# get meg data info
meg_stim_info = [i for i in sorted(os.listdir(meg_path)) if 'eyes' in i]
meg_info_pd = pd.read_csv(meg_path + meg_stim_info[0])

# timing of stimulus
meg_timing = meg_info_pd['time'].values[:1681]


# load meg data, resize to embeddings
# meg_mat = h5py.File(meg_path + 'P4_cosmofile-004.mat')
# print(meg_mat.keys())
# meg_vec = np.array(meg_mat['ds']['samples'])
# meg_vec_resize = cv2.resize(meg_vec, embed_shape[::-1], interpolation=cv2.INTER_LANCZOS4)  # row, col reversed
# np.save(f'{embed_path}meg_vec_resize.npy', meg_vec_resize)

meg_vec_resize = np.load(f'{embed_path}meg_vec_resize.npy')


# print(cosine_similarity(meg_vec_resize, org_img_embed))

# get the euclidean distance for embeddings to meg data
euc_dist_meg_org = np.log(np.linalg.norm(meg_vec_resize - org_img_embed))  # l2 norm
euc_dist_meg_nb = np.log(np.linalg.norm(meg_vec_resize - nb_img_embed))
euc_dist_meg_nf = np.log(np.linalg.norm(meg_vec_resize - nf_img_embed))


print(euc_dist_meg_org)
print(euc_dist_meg_nb)
print(euc_dist_meg_nf)


# make barplot of euc values
fig, ax = plt.subplots()
ax.bar(range(3), [euc_dist_meg_org, euc_dist_meg_nb, euc_dist_meg_nb]) # , color='cyan'
ax.set_xticks(range(3), labels = ['meg_org_img', 'meg_nb_img', 'meg_nf_img'])
# for i, j in enumerate(rsa_lst):
#    ax.text(i, j+0.02, str(round(j, 2)), c='k', fontweight='bold', verticalalignment='center')
plt.xlabel(f'{euc_dist_meg_org}            {euc_dist_meg_nb}               {euc_dist_meg_nf}')
plt.ylabel('Euclidean distance', fontweight='bold')
plt.title('Comparison of model embeddings to MEG data', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_path}dist_bar_meg_mdl_embed.png', dpi=500)
plt.show()

