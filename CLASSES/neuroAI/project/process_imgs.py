"""
Name: Kimberly Nestor
Class: CMU 10-733 NeuroAI Project
Date: 5/12/24

Description: Preprocess Things images into foreground only images and background only images.
             Fill in missing spaces in background images with similar landscape.
             These along with original images will be used as input to ResNEt18 to collect embeddings.

Resources:
https://elifesciences.org/articles/82580
https://things-initiative.org/
https://things-initiative.org/projects/things-images/
https://plus.figshare.com/collections/THINGS-data_A_multimodal_collection_of_large-scale_datasets_for_investigating_object_representations_in_brain_and_behavior/6161151
"""

import os
import sys
from tqdm import tqdm

import rembg
from PIL import Image
import cv2

import numpy as np
import matplotlib.pyplot as plt

import multiprocessing


# dir info for data
curr_dir = os.getcwd()
stim_img_path = curr_dir + '/images/'
nb_img_path = curr_dir + '/images_nb/'
bg_img_path = curr_dir + '/images_bg/'

nb_mask_path = curr_dir + '/nb_mask/'
nf_img_path = curr_dir + '/images_nf/'

meg_path = curr_dir + '/THINGS-MEG/preprocessed/'


def strip_imgs(stim_img_cats):
    """This function takes as input the stimulus image categories, removes foreground, background and saves as separate images."""
    # loop to process images - get only meg images, remove background, remove foreground, save as npy
    for i in tqdm(range(len(stim_img_cats))):
        print(stim_img_cats[i])

        # select only first 12 images from each category
        meg_stim_imgs = sorted(os.listdir(f'{stim_img_path}{stim_img_cats[i]}/'))[:12]

        # loop to load imgs, remove background, remove foreground
        for ii in range(len(meg_stim_imgs)):
            # load stim img
            stim_img = np.array(Image.open(f'{stim_img_path}{stim_img_cats[i]}/{meg_stim_imgs[ii]}'))

            # foreground only image - remove background, leave only main object
            nb_img_vec = rembg.remove(stim_img)
            nb_img = Image.fromarray(nb_img_vec)

            # Save the foreground only image
            if not os.path.isdir(nb_img_path + stim_img_cats[i]):
                os.makedirs(nb_img_path + stim_img_cats[i])
            nb_img.save(f'{nb_img_path}{stim_img_cats[i]}/{meg_stim_imgs[ii][:-3]}png')


            # background only image - remove foreground object
            stim_img_rgb = stim_img[:, :, 0:3]  # get only RGB channels
            alpha_chan_nb = nb_img_vec[:, :, 3] # get alpha channel from nb img
            alpha_nb = np.dstack((alpha_chan_nb, alpha_chan_nb, alpha_chan_nb))  # 3 channels - foreground mask
            alpha_nb_img = Image.fromarray(alpha_nb)

            # Save the foreground only mask
            if not os.path.isdir(nb_mask_path + stim_img_cats[i]):
                os.makedirs(nb_mask_path + stim_img_cats[i])
            alpha_nb_img.save(f'{nb_mask_path}{stim_img_cats[i]}/{meg_stim_imgs[ii][:-3]}png')

            # remove foreground from the image
            nf_img = stim_img_rgb.astype(float) * (1 - alpha_nb.astype(float) / 255)  # Multiply by 1-alpha
            nf_img = nf_img.astype(np.uint8)  # Convert back to uint8
            nf_img_pic = Image.fromarray(nf_img)

            # Save the images with blank foreground
            if not os.path.isdir(nf_img_path + stim_img_cats[i]):
                os.makedirs(nf_img_path + stim_img_cats[i])
            nf_img_pic.save(f'{nf_img_path}{stim_img_cats[i]}/{meg_stim_imgs[ii][:-3]}png')

            # inpaint missing foreground to get only background img
            in_img = cv2.imread(f'{nf_img_path}{stim_img_cats[i]}/{meg_stim_imgs[ii][:-3]}png')
            in_mask = cv2.imread(f'{nb_mask_path}{stim_img_cats[i]}/{meg_stim_imgs[ii][:-3]}png', cv2.IMREAD_GRAYSCALE)

            bg_only_img = cv2.inpaint(in_img, in_mask, 10, cv2.INPAINT_NS)

            # Save the background only image
            if not os.path.isdir(bg_img_path + stim_img_cats[i]):
                os.makedirs(bg_img_path + stim_img_cats[i])
            cv2.imwrite(f'{bg_img_path}{stim_img_cats[i]}/{meg_stim_imgs[ii][:-3]}png', bg_only_img)


if __name__ == '__main__':

    # get categories of stim images - MEG study presented first 12 egs of each
    stim_img_cats = sorted(os.listdir(stim_img_path))[2:]

    # pool = multiprocessing.Pool() # use 12 cores
    # pool.map(strip_imgs, ['aardvark', 'abacus', 'accordion'])
    # pool.map(strip_imgs, stim_img_cats[59:])
    # pool.map_async(strip_imgs, ['aardvark', 'abacus', 'accordion'])

    man_pool_lst = list(zip(np.linspace(210, len(stim_img_cats), 5).astype(int), np.linspace(210, len(stim_img_cats), 5).astype(int)[1:]))

    # strip_imgs(stim_img_cats[man_pool_lst[0][0]+6+39+73:man_pool_lst[0][1]])
    # strip_imgs(stim_img_cats[man_pool_lst[1][0]+5+40+73:man_pool_lst[1][1]])
    # strip_imgs(stim_img_cats[man_pool_lst[2][0]+5+39+73:man_pool_lst[2][1]])
    # strip_imgs(stim_img_cats[man_pool_lst[3][0]+5+24+13+73:man_pool_lst[3][1]])




