"""
Name: Kimberly Nestor
Class: CMU 10-733 NeuroAI
Date: 2/28/24

Description: Makes encoding model from fMRI story data and RoBERTa model embeddings.

Disclosures: I used the cross_validated_ridge.py module for training ridge regression,
             which was provided by Prof. Leila Wehbe.
"""
import os

from cross_validated_ridge import SimpleCVRidgeDiffLambda



# dir info for data
curr_dir = os.getcwd()
dir_depth = '/recordings/ds003020/derivative'
story_save_path = curr_dir + '/clean_stories/'
output_figs_dir = curr_dir + '/output_figs/' # UTS01, UTS02, UTS03
new_embed_save_path = curr_dir + '/resize_norm_embeddings/'
new_mat_save_path = curr_dir + '/resize_fmri_mats/'

# os.listdir(new_embed_save_path)

CV_ridge = SimpleCVRidgeDiffLambda(solver = 'auto') # set up the sklearn solver here
CV_ridge.fit(X, Y) # where X is input features (nTRs*features) and Y is output (nTRs*voxels) if you're doing an encoding model

