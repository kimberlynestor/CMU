"""
Name: Kimberly Nestor
Class: CMU 10-733 NeuroAI HW2
Date: 4/22/24

Description: Uses RSA to compare 25 Taskonomy models.

Disclosures: Used links below to examine figure code from (Wang et al., 2019).
https://github.com/ariaaay/NeuralTaskonomy/blob/master/code/make_task_matrix.py
https://github.com/ariaaay/NeuralTaskonomy/blob/master/code/make_task_tree.py
"""

import sys
import os
from pathlib import Path

import torch

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# set shared scripts dir for both msit and stroop up one level
curr_dir = os.getcwd()
pars = Path(curr_dir).parents
par_dir = pars[0]
sys.path.insert(0, str(par_dir) +'/'+ os.listdir(par_dir)[-1])

from rsa_funcs import *


# dir info for data
curr_dir = os.getcwd()
stim_img_path = curr_dir + '/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/'
task_mdls_path = curr_dir + '/task_mdls/'
rsa_output_path = curr_dir + '/rsa_output/'

# stim_rep_names = sorted(os.listdir(task_mdls_path +'/'+ mdl_names[0]))
img_names_lst = open('stim_img_names.csv', 'r').readlines()[0].split(',')[:-1]

# taskonomy mdl list
mdl_names = ['autoencoding', 'depth_euclidean', 'jigsaw', 'reshading', 'edge_occlusion', 'keypoints2d', \
             'room_layout', 'curvature', 'edge_texture', 'keypoints3d', 'segment_unsup2d', 'class_object', 'egomotion', \
             'nonfixated_pose', 'segment_unsup25d', 'class_scene', 'fixated_pose', 'normal', 'segment_semantic', 'denoising', \
             'inpainting', 'point_matching', 'vanishing_point']


#### 1.1 Compute and plot RSA matrix
# get pairwise RSA for all models, using stimulus mats

rep_rsa_mat = []
for i in mdl_names:
    print(i)
    rep_mdl1 = np.array(
        [torch.load(task_mdls_path + i + '/' + img + '.pt').detach().numpy() for img in img_names_lst])

    rep_rsa_lst = []
    for ii in mdl_names:
        rep_mdl2 = np.array(
            [torch.load(task_mdls_path + ii + '/' + img + '.pt').detach().numpy() for img in img_names_lst])

        rep_mdl_rsa = rsa_mats(rep_mdl1, rep_mdl2)
        rep_rsa_lst.append(rep_mdl_rsa)

    # print(rep_rsa_lst)
    rep_rsa_mat.append(np.array(rep_rsa_lst))

rep_rsa_mat = np.array(rep_rsa_mat)
np.save(f'{rsa_output_path}rep_mdls_rsa_mat.npy', rep_rsa_mat)


# plot RSA mat
rep_mdls_rsa_mat = np.load(f'{rsa_output_path}rep_mdls_rsa_mat.npy')

sns.heatmap(rep_mdls_rsa_mat, cmap=sns.cubehelix_palette(as_cmap=True), xticklabels=mdl_names, yticklabels=mdl_names)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.subplots_adjust(bottom=0.35, left=0.3)
plt.title('RSA matrix of 23 Taskonomy models', fontweight='bold')
plt.savefig(f'{rsa_output_path}rep_mdls_rsa_mat.png', dpi=500)
plt.show()
# plt.close()


#### 1.2 Derive task tree from RSA matrix
# plot dendrogram from RSA matrix, showing clustering of similarity among rep mdls
links = linkage(rep_mdls_rsa_mat, 'average')
plt.figure(figsize=(12, 4))
ax = plt.subplot(1, 1, 1)
dn = dendrogram(links, labels=mdl_names, leaf_font_size=15, color_threshold=0,
                          above_threshold_color='gray')

color_list = ['blue']*5 + ['green']*3 + ['purple']*7 + ['magenta']*8
[j.set_color(i) for (i, j) in zip(color_list, ax.xaxis.get_ticklabels())]

plt.xticks(rotation="vertical", fontsize=11)
plt.subplots_adjust(bottom=0.42)
plt.title('Dendrogram showing similarity groupings of Taskonomy models', fontweight='bold')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig(f'{rsa_output_path}rep_mdls_rsa_tree_avg.png', dpi=500)
plt.show()
# plt.close()


#### 1.3 Repeat RSA matrix and dendrogram tree with only scene images
# get pairwise RSA for all models, using only scene stimulus mats
scene_img_lst = [i.split('.')[0] for i in sorted(os.listdir(stim_img_path + 'Scene'))][1:]

scene_idxs = sorted([i for i, val in enumerate(img_names_lst) if val in set(scene_img_lst)])
scene_img_names_lst = img_names_lst[scene_idxs[0]:scene_idxs[-1]+1]


scene_rep_rsa_mat = []
for i in mdl_names:
    rep_mdl1 = np.array(
        [torch.load(task_mdls_path + i + '/' + img + '.pt').detach().numpy() for img in scene_img_lst])

    scene_rep_rsa_lst = []
    for ii in mdl_names:
        rep_mdl2 = np.array(
            [torch.load(task_mdls_path + ii + '/' + img + '.pt').detach().numpy() for img in scene_img_lst])

        scene_rep_mdl_rsa = rsa_mats(rep_mdl1, rep_mdl2)
        scene_rep_rsa_lst.append(scene_rep_mdl_rsa)

    # print(rep_rsa_lst)
    scene_rep_rsa_mat.append(np.array(scene_rep_rsa_lst))

scene_rep_rsa_mat = np.array(scene_rep_rsa_mat)
np.save(f'{rsa_output_path}scene_rep_mdls_rsa_mat.npy', scene_rep_rsa_mat)


# plot scene RSA mat
scene_rep_mdls_rsa_mat = np.load(f'{rsa_output_path}scene_rep_mdls_rsa_mat.npy')

sns.heatmap(scene_rep_mdls_rsa_mat, cmap=sns.cubehelix_palette(as_cmap=True), xticklabels=mdl_names, yticklabels=mdl_names)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.subplots_adjust(bottom=0.35, left=0.3)
plt.title('RSA matrix of 23 Taskonomy models - only scene images', fontweight='bold')
plt.savefig(f'{rsa_output_path}scene_rep_mdls_rsa_mat.png', dpi=500)
plt.show()
# plt.close()


# plot dendrogram from RSA matrix, showing clustering of similarity among rep mdls - only scene images
links = linkage(scene_rep_mdls_rsa_mat, 'average')
plt.figure(figsize=(12, 4))
ax = plt.subplot(1, 1, 1)
dn = dendrogram(links, labels=mdl_names, leaf_font_size=15, color_threshold=0,
                          above_threshold_color='gray')

color_list = ['blue']*5 + ['green']*8 + ['magenta']*10
[j.set_color(i) for (i, j) in zip(color_list, ax.xaxis.get_ticklabels())]

plt.xticks(rotation="vertical", fontsize=11)
plt.subplots_adjust(bottom=0.42)
plt.title('Dendrogram showing similarity groupings of Taskonomy models - only scene images', fontweight='bold')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig(f'{rsa_output_path}scene_rep_mdls_rsa_tree_avg.png', dpi=500)
plt.show()
# plt.close()