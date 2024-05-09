"""
Name: Kimberly Nestor
Class: CMU 10-733 NeuroAI HW2
Date: 4/22/24

Description: Uses taskonomy representation models to train encoding model to predict fmri response.
             Drop non-significant pvals, compute correlations and use dot product to calculate cosine similarity
             across model fMRI predictions.

Disclosures: Used the following links for reference.
https://neurostars.org/t/extracting-voxels-within-a-mask-for-glm-in-nilearn/23286
https://nilearn.github.io/dev/modules/generated/nilearn.masking.apply_mask.html
https://nilearn.github.io/dev/manipulating_images/masker_objects.html
https://neurostars.org/t/how-to-create-a-niftimage-object/22719
https://neurostars.org/t/using-a-probabilistic-mask-with-nilearns-niftimasker-function/19120
https://nilearn.github.io/dev/auto_examples/06_manipulating_images/plot_nifti_simple.html
"""

import os
import sys
from pathlib import Path
import itertools
from tqdm import tqdm
import json

# set shared scripts dir for both msit and stroop up one level
curr_dir = os.getcwd()
pars = Path(curr_dir).parents
par_dir = pars[0]
sys.path.insert(0, str(par_dir) +'/'+ os.listdir(par_dir)[-1])

from cross_validated_ridge import SimpleCVRidgeDiffLambda
from rsa_funcs import *
from permutation import *

import torch
import cv2

from sklearn.metrics import r2_score
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from nilearn.image import load_img
from nilearn.masking import apply_mask
from nilearn import image, maskers, plotting
import nibabel as nib
from nilearn.mass_univariate import permuted_ols
from nilearn.glm import fdr_threshold

import matplotlib.pyplot as plt
import seaborn as sns

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


#### 2.1 Construct encoding model for CSI1 - use taskonomy mdls to predict fmri response, compute similarity mats

# whole brain mask
brain_mask = load_img( mask_path + sorted(os.listdir(mask_path))[1] ) #.get_fdata()

mask_affine = brain_mask.affine
mask_shape = brain_mask.shape

brain_mask_bin = (np.array(brain_mask.get_fdata()) >0).astype(int)
brain_mask_bin = nib.Nifti1Image(brain_mask_bin, mask_affine, dtype='int64')


# load CSI1 data, 15 sessions - mask, save as npy
sess_names = sorted(os.listdir(fmri_path))
"""
for f_name in sess_names:
    sess = load_img(fmri_path + f_name)
    sess_mean_img = image.mean_img(sess)

    # took forever because of nan and then killed
    ## masker = maskers.NiftiMasker(brain_mask_bin, target_affine=brain_mask.affine, target_shape=brain_mask.shape)
    ## masker.fit(sess1)
    ## masked_data = masker.transform(sess1)

    masker = maskers.NiftiMasker(mask_strategy='epi', target_affine=brain_mask.affine, target_shape=brain_mask.shape)
    masker.fit(sess)
    masked_data = masker.transform(sess) # output 2D mat, (n_voxels x n_tr)

    np.save(f'{masked_fmri_path}CSI1_masked_fmri_{f_name[-10:-4]}.npy', masked_data)

    # plot masked data to sanity check
    # plotting.plot_roi(masker.mask_img_, sess1_mean_img, title='Masked CSI1 data', cut_coords=[50, 35, 50])
    # plt.savefig(f'{asym_output_path}masked_csi1_func_data.png', dpi=500)
    # plt.show()
    # plt.close()
"""

rz_shape = [370, 2048] # from fmri TR and mdl vec size

# load masked fmri data
masked_sess_names = sorted(os.listdir(masked_fmri_path))
masked_fmri_data = [np.load(f'{masked_fmri_path}{i}') for i in masked_sess_names]

# resize fmri brain masks
"""
for i in range(len(sess_names)):
    new_fmri = cv2.resize(masked_fmri_data[i], rz_shape[::-1], interpolation=cv2.INTER_LANCZOS4) # row, col reversed
    print(new_fmri.shape)
    np.save(f'{rz_masked_fmri_path}CSI1_rz_masked_fmri_{sess_names[i][-10:-4]}.npy', new_fmri)
"""

# load resize masked fmri data
rz_masked_sess_names = sorted(os.listdir(rz_masked_fmri_path))
rz_masked_fmri_data = [np.load(f'{rz_masked_fmri_path}{i}') for i in rz_masked_sess_names]


# session dir names
sess_dir = sorted(os.listdir(stim_org_img_path))[1:-1]

# get stim img ord for each session
full_sess_img_ord = []
for i in range(len(sess_dir)):
    sess_run_docs = [i for i in sorted(os.listdir(stim_org_img_path +'/'+ sess_dir[i])) if i.endswith('txt')]
    # for each run in session get img ord
    sess_img_ord = []
    for ii in range(len(sess_run_docs)):
        sess_run_img_ord = open(stim_org_img_path + '/' + sess_dir[i] + '/' + sess_run_docs[ii], 'r').readlines()
        sess_img_ord.append(sess_run_img_ord)
    sess_img_ord = [i.replace('.jpg\n', '').replace('.JPEG\n', '').replace('.jpeg\n', '').replace('rep_', '') \
                    for i in list(itertools.chain.from_iterable(sess_img_ord))]
    full_sess_img_ord.append(sess_img_ord)

# split imaging sessions into train and test - sess 15 = test
full_sess_img_ord_train = full_sess_img_ord[:-1]
full_sess_img_ord_test = full_sess_img_ord[-1]

# use rep from all taskonomy mdls, predict fmri output
"""
for i in tqdm(range(len(mdl_names))):
    print(mdl_names[i])
    # train encoding mdl on sess 1-14
    for ii in range(len(full_sess_img_ord_train)):
        print(f'sess {ii+1}')

        # load rep mdl org by stim img presentation for fmri sess, save
        rep_mdl_stim_org = np.array([torch.load(task_mdls_path + mdl_names[i] + '/' + img + '.pt').detach().numpy() for img in full_sess_img_ord_train[ii]])
        if not os.path.isdir(task_mdls_stim_org_path + mdl_names[i]):
            os.makedirs(task_mdls_stim_org_path + mdl_names[i])
        np.save(f'{task_mdls_stim_org_path}{mdl_names[i]}/{mdl_names[i]}_stim_org_{sess_names[ii][-10:-4]}.npy', rep_mdl_stim_org)

        # resize mdl shape to match fmri data, save
        rz_rep_mdl_stim_org = cv2.resize(rep_mdl_stim_org, rz_shape[::-1], interpolation=cv2.INTER_LANCZOS4)  # row, col reversed
        if not os.path.isdir(rz_task_mdls_stim_org_path + mdl_names[i]):
            os.makedirs(rz_task_mdls_stim_org_path + mdl_names[i])
        np.save(f'{rz_task_mdls_stim_org_path}{mdl_names[i]}/{mdl_names[i]}_rz_stim_org_{sess_names[ii][-10:-4]}.npy', rz_rep_mdl_stim_org)

        # train encoding mdl
        CV_ridge = SimpleCVRidgeDiffLambda(solver = 'sag', n_splits=10) # init solver = 'auto'
        ridge_train = CV_ridge.fit(rz_rep_mdl_stim_org, rz_masked_fmri_data[ii])  # X, Y

    # predict on sess 15 as test, resize mdl
    rep_mdl_stim_org_test = np.array([torch.load(task_mdls_path + mdl_names[i] + '/' + img + '.pt').detach().numpy() for img in full_sess_img_ord_test])
    np.save(f'{task_mdls_stim_org_path}{mdl_names[i]}/{mdl_names[i]}_stim_org_ses-15.npy', rep_mdl_stim_org_test)

    rz_rep_mdl_stim_org_test = cv2.resize(rep_mdl_stim_org_test, rz_shape[::-1], interpolation=cv2.INTER_LANCZOS4)  # row, col reversed
    np.save(f'{rz_task_mdls_stim_org_path}{mdl_names[i]}/{mdl_names[i]}_rz_stim_org_ses-15.npy', rz_rep_mdl_stim_org_test)

    predict_fmri = CV_ridge.predict(rz_rep_mdl_stim_org_test)
    np.save(f'{pred_output_dir}{mdl_names[i]}_pred_fmri.npy', predict_fmri)
"""

pred_fmri_rep_mdls = np.array([np.load(f'{pred_output_dir}{i}_pred_fmri.npy') for i in mdl_names])

"""
rep_sim_mat = []
for i in tqdm(range(len(pred_fmri_rep_mdls))):
    # compute pvals, do permutation test - returns voxel pvals, score_orig_data, and h0_fmax
    permut_ps = permuted_ols(rz_masked_fmri_data[-1],  pred_fmri_rep_mdls[i], n_perm=1000, random_state=11)[0]
    permut_ps_nolog = np.mean(10**np.log10(permut_ps) * -1, axis=0)
    fdr_permut_ps_nolog = fdrcorrection(permut_ps_nolog)[-1]
    sig_pval_idx = sorted(fdr_corrected_significant_ps(fdr_permut_ps_nolog, 0.05))

    # get mdl corr
    mdl1_pred_corr = lower_t(np.corrcoef(pred_fmri_rep_mdls[i]))

    rep_sim_lst = []
    for ii in range(len(pred_fmri_rep_mdls)):
        # get mdl corr
        mdl2_pred_corr = lower_t(np.corrcoef(pred_fmri_rep_mdls[ii]))

        # calculate cosine similarity using dot product
        dot_sim = np.dot(mdl1_pred_corr, mdl2_pred_corr)
        mag_mdl1 = np.linalg.norm(mdl1_pred_corr)
        mag_mdl2 = np.linalg.norm(mdl2_pred_corr)

        cos_sim = dot_sim / (mag_mdl1 * mag_mdl2)
        rep_sim_lst.append(cos_sim)

    rep_sim_mat.append(np.array(rep_sim_lst))

rep_sim_mat = np.array(rep_sim_mat)
np.save(f'{asym_output_path}rep_mdls_fmri_pred_sim_mat.npy', rep_sim_mat)
"""

# plot sim mat - comparing model fmri predictions
rep_mdls_sim_mat = np.load(f'{asym_output_path}rep_mdls_fmri_pred_sim_mat.npy')

sns.heatmap(rep_mdls_sim_mat, cmap=sns.cubehelix_palette(as_cmap=True), xticklabels=mdl_names, yticklabels=mdl_names)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.subplots_adjust(bottom=0.35, left=0.3)
plt.title('Similarity matrix of fMRI predictions from 23 Taskonomy models', fontweight='bold')
# plt.savefig(f'{asym_output_path}rep_mdls_fmri_pred_sim_mat.png', dpi=500)
# plt.show()
plt.close()


#### 2.3 Calculate coefficient of determinant (R^2) for 5 model predictions
# look for r^2, compare mdl pred to true fmri responses
mdl_r2_dict = {}
for i in range(5):
    mdl_r2 = r2_score(rz_masked_fmri_data[-1], pred_fmri_rep_mdls[i])
    mdl_r2_dict[mdl_names[i]] = mdl_r2

print(mdl_r2_dict)

mdl_r2_dict_file = open(f'{asym_output_path}sing_mdl_r2_score.json', 'w')
json.dump(mdl_r2_dict, mdl_r2_dict_file)

