"""
Name: Kimberly Nestor
Class: CMU 10-733 NeuroAI
Date: 2/28/24

Description: Resample model embeddings to match fMRI timescale in TR, shift to account for HRF lag, normalize.
Resample fMRI data for timescale and match cortical voxel to dimension of model layer.

Disclosures: I used the following link for help with the fMRI shift/ FIR model part of this script.
https://github.com/HuthLab/speechmodeltutorial/blob/master/SpeechModelTutorial.ipynb
"""

import os
import itertools

import h5py # acts ac dict
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from fmri_funcs import *

import torch


# dir info for data
curr_dir = os.getcwd()
dir_depth = '/recordings/ds003020/derivative'
text_data_dir = curr_dir + dir_depth + '/TextGrids/' # 8 storeis for train, wheretheressmoke for test
story_save_path = curr_dir + '/clean_stories/'
embed_save_path = curr_dir + '/embeddings/'
fmri_data_dir = curr_dir + dir_depth + '/preprocessed_data/' # UTS01, UTS02, UTS03
output_figs_dir = curr_dir + '/output_figs/' # UTS01, UTS02, UTS03
new_embed_save_path = curr_dir + '/resize_norm_embeddings/'
new_mat_save_path = curr_dir + '/resize_fmri_mats/'


#### 1.2 Align model embeddings with neural data
##### RoBERTa story embeddings
# story names with time to read in fmri tr
story_names = sorted(os.listdir(story_save_path))
story_times_tr = [story_time(text_data_dir, i+'.TextGrid') for i in story_names]


# load model embeddings
model_embeds = []
for i in range(len(story_names)):
    embed = torch.load(embed_save_path+story_names[i]).detach().numpy()
    model_embeds.append(embed)


# plot model embed before padding, interpolation - sample story adollshouse, final layer
sample_story = model_embeds[0][-1][:,1]
plt.plot(np.linspace(0, story_times_tr[0], len(sample_story)), sample_story)
plt.xlabel('Timescale (TR)', weight='bold')
plt.ylabel('Activation', weight='bold')
plt.title('RoBERTa model activations (before, Feature2, adollshouse)', weight='bold')
# plt.savefig(f'{output_figs_dir}mdl_feat_before_resample.png', dpi=500)
# plt.show()
plt.close()


# pad stimulus embeddings to match delayed fMRI response, shift=4
shift_embeds = list(map(hrf_shift, model_embeds))

# resize stimuli time dimension to that of fMRI, Lanczos sampling
resize_embeds = []
for i in range(len(shift_embeds)):
    curr_shape = list(shift_embeds[i].shape)
    curr_shape[1] = story_times_tr[i]
    new_shape = curr_shape
    r_embed = lz_resize(shift_embeds[i], new_shape)
    resize_embeds.append(r_embed)
    # print(f'{r_embed.shape} = {story_names[i]}')


# plot model embed after padding, interpolation - sample story adollshouse, final layer
sample_story = resize_embeds[0][-1][:,1]
plt.plot(sample_story)
plt.xlabel('Timescale (TR)', weight='bold')
plt.ylabel('Activation', weight='bold')
plt.title('RoBERTa model activations (after, Feature2, adollshouse)', weight='bold')
# plt.savefig(f'{output_figs_dir}mdl_feat_after_resample.png', dpi=500)
# plt.show()
plt.close()


# normalize model embed stimuli
resize_norm_embeds = list(map(lambda story: np.array(list(map(lambda layer: stats.zscore(layer), story))), resize_embeds))

"""
# loop to save new embeds as npy
for i in range(len(resize_norm_embeds)):
    np.save(f'{new_embed_save_path}{story_names[i]}.npy', resize_norm_embeds[i])
"""

# final shapes of model embeds after match fmri timescale
mdl_fin_shapes = [i.shape for i in resize_norm_embeds]


##### fmri data
# preprocessed already trimmed by 10secs beginning and end for noise
# already masked using thick mask from pycortex = only cortical voxels, already normalized
# decided not to shift fmri data by FIR model because data has been trimmed considerably
all_subjs_id = [i for i in sorted(os.listdir(fmri_data_dir)) if 'UTS' in i]
en_subjs_id = all_subjs_id[:3] # use three subjects

all_fmri_story_names = sorted(os.listdir(fmri_data_dir + en_subjs_id[0]))
fmri_story_names = list(itertools.chain(all_fmri_story_names[0:8], [all_fmri_story_names[-2]]))

all_story_mats = get_fmri_mats(fmri_data_dir, en_subjs_id, fmri_story_names)

# resample fmri data to match mdl data, fmri timescale and nodes in each mdl layer, Lanczos sampling
resize_mats = []
for i in range(len(all_story_mats)):
    # print(f'{all_story_mats[i].shape}, fmri before = {story_names[i]}')
    new_shape = list(mdl_fin_shapes[i])
    r_mat = lz_resize(all_story_mats[i], new_shape)
    resize_mats.append(r_mat)
    # print(f'{r_mat.shape}, fmri after  = {story_names[i]}')
    # np.save(f'{new_mat_save_path}{story_names[i]}.npy', resize_norm_embeds[i])

