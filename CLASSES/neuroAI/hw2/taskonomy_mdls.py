"""
Name: Kimberly Nestor
Class: CMU 10-733 NeuroAI HW2
Date: 4/22/24

Description: Uses Bold5000 stimuli images to derive 25 Taskonomy models.

Resources:
https://bold5000-dataset.github.io/website/download.html
https://github.com/alexsax/midlevel-reps?files=1

Disclosures:
"""

import os
import sys
from itertools import chain

from PIL import Image
import cv2
import visualpriors

import torch.utils.model_zoo
import torch
import torchvision.transforms.functional as TF

from sklearn.preprocessing import MinMaxScaler
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

# dir info for data
curr_dir = os.getcwd()
stim_img_path = curr_dir + '/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/'
task_mdls_path = curr_dir + '/task_mdls/'


# stimulus img info
img_source_lst = sorted(os.listdir(stim_img_path))[1:]

# get all stim imgs, as tensor - resize [3, 256, 256], normalize [-1, 1]
all_img_lst = []
img_names_lst = []
for i in range(len(img_source_lst)):
    sources_imgs = sorted(os.listdir(stim_img_path + img_source_lst[i])) #[1:]
    img_names_lst.append(sources_imgs)
    for ii in range(len(sources_imgs)):
        img = Image.open(stim_img_path + img_source_lst[i] +'/'+ sources_imgs[ii])
        to_tensor = TF.to_tensor(TF.resize(img, [256, 256])) * 2 - 1
        to_tensor = to_tensor.unsqueeze_(0)
        # img.show()
        all_img_lst.append(to_tensor)


# get all image names, remove jpg, save in csv
img_names_lst = [i.split('.')[0] for i in sum(img_names_lst, [])]
stim_names_file = open('stim_img_names.csv', 'w')
stim_names_file = open('stim_img_names.csv', 'a')
for i in img_names_lst:
    stim_names_file.write(f'{i},')


# taskonomy mdl list
mdl_names = ['autoencoding', 'depth_euclidean', 'jigsaw', 'reshading', 'colorization', 'edge_occlusion', 'keypoints2d', \
             'room_layout', 'curvature', 'edge_texture', 'keypoints3d', 'segment_unsup2d', 'class_object', 'egomotion', \
             'nonfixated_pose', 'segment_unsup25d', 'class_scene', 'fixated_pose', 'normal', 'segment_semantic', 'denoising', \
             'inpainting', 'point_matching', 'vanishing_point']


# get taskonomy mdls, using visualpriors -- really takes shape [1, 3, 256, 256] and output [1, 8, 16, 16]
for mdl in mdl_names:
        print('\n', mdl)
        if mdl == 'colorization':  # weird checkpoint shape mismatch error
        # if mdl in ['autoencoding', 'depth_euclidean', 'jigsaw', 'reshading', 'colorization']:
            continue
        else:
            for i in range(len(img_names_lst)):
                print(img_names_lst[i])
                representation = visualpriors.representation_transform(all_img_lst[i], mdl, device='cpu')
                representation = torch.ravel(representation)

                if not os.path.isdir(task_mdls_path + mdl):
                    os.makedirs(task_mdls_path + mdl)
                torch.save(representation, f'{task_mdls_path + mdl}/{img_names_lst[i]}.pt')

