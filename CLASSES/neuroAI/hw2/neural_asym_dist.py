"""
Name: Kimberly Nestor
Class: CMU 10-733 NeuroAI HW2
Date: 4/22/24

Description: Uses

Disclosures:
"""

import os
import sys
from pathlib import Path
import itertools # chain, combinations
from tqdm import tqdm
import json

# set shared scripts dir for both msit and stroop up one level
curr_dir = os.getcwd()
pars = Path(curr_dir).parents
par_dir = pars[0]
sys.path.insert(0, str(par_dir) +'/'+ os.listdir(par_dir)[-1])

from cross_validated_ridge import SimpleCVRidgeDiffLambda
from sklearn.metrics import r2_score

from sklearn.metrics import r2_score
import numpy as np

import warnings
warnings.filterwarnings('ignore')

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

# fmri sess names
sess_names = sorted(os.listdir(fmri_path))


#### 2.4 Assymmetric comparison - coefficient of determinant (R^2) for mdl pairs used in encoding
# load resize masked fmri data
rz_masked_sess_names = sorted(os.listdir(rz_masked_fmri_path))
rz_masked_fmri_data = np.array([np.load(f'{rz_masked_fmri_path}{i}') for i in rz_masked_sess_names])
rz_masked_fmri_data_train = rz_masked_fmri_data[:-1]
rz_masked_fmri_data_test = rz_masked_fmri_data[-1]


for i in tqdm(range(len(mdl_names))):
    print(mdl_names[i])
    # train encoding mdl on sess 1-14
    for ii in range(len(rz_masked_fmri_data_train)):
        print(f'sess {ii+1}')

        # load resize rep mdl org by stim img
        # rz_rep_mdl_names = sorted(os.listdir(rz_task_mdls_stim_org_path + mdl_names[i] +'/'))
        rz_rep_mdl_stim_org = np.load(f'{rz_task_mdls_stim_org_path}{mdl_names[i]}/{mdl_names[i]}_rz_stim_org_{sess_names[ii][-10:-4]}.npy')


        print(rz_rep_mdl_stim_org)
        break

sys.exit()



#### 2.5 Relative unique R^2 - from combined encoding mdls

mdl_r2_dict_file = open(f'{asym_output_path}sing_mdl_r2_score.json', 'r')
sing_mdl_r2_dict = json.load(mdl_r2_dict_file)

print(sing_mdl_r2_dict)