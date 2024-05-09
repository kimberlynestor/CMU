"""
Name: Kimberly Nestor
Class: CMU 10-733 NeuroAI HW2
Date: 4/22/24

Description: Uses

Disclosures:
"""

import os
import sys
# from pathlib import Path
import itertools # chain, combinations
from tqdm import tqdm
import json


# dir info for data
curr_dir = os.getcwd()
stim_img_path = curr_dir + '/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/'
stim_org_img_path = curr_dir + '/BOLD5000_Stimuli/Stimuli_Presentation_Lists/CSI1/'
mask_path = curr_dir + '/sub-CSI1/anatomicals/'

rsa_output_path = curr_dir + '/rsa_output/'
asym_output_path = curr_dir + '/asym_output/'

task_mdls_path = curr_dir + '/task_mdls/'
task_mdls_stim_org_path = curr_dir + '/task_mdls_stim_org/'
rz_task_mdls_stim_org_path = curr_dir + '/resize_task_mdls_stim_org/'

fmri_path = curr_dir + '/CSI1_TYPED/'
masked_fmri_path = curr_dir + '/masked_fmri_data/'
rz_masked_fmri_path = curr_dir + '/resize_mask_fmri_data/'

pred_output_dir = curr_dir + '/fmri_pred_output/'

# taskonomy mdl list
mdl_names = ['autoencoding', 'depth_euclidean', 'jigsaw', 'reshading', 'edge_occlusion', 'keypoints2d', \
             'room_layout', 'curvature', 'edge_texture', 'keypoints3d', 'segment_unsup2d', 'class_object', 'egomotion', \
             'nonfixated_pose', 'segment_unsup25d', 'class_scene', 'fixated_pose', 'normal', 'segment_semantic', 'denoising', \
             'inpainting', 'point_matching', 'vanishing_point']



#### 2.4 Assymmetric comparison - coefficient of determinant (R^2) for mdl pairs used in encoding


#### 2.5 Relatice unique R^2 - from combined encoding mdls

mdl_r2_dict_file = open(f'{asym_output_path}sing_mdl_r2_score.json', 'r')
sing_mdl_r2_dict = json.load(mdl_r2_dict_file)

print(sing_mdl_r2_dict)