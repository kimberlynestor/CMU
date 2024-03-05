"""
Name: Kimberly Nestor
Class: CMU 10-733 NeuroAI
Date: 2/28/24
Description: Helper functions to align model stimuli data with fMRI lag and match model and fMRI dimensions.
"""

import sys
import itertools

import nltk
from nltk.corpus import words

import math
import numpy as np

import cv2
import h5py # acts ac dict


tr = 2 # from json file

def story_time(path, story_name):
    """function that takes a path to a TextGrids file determines
    how long it took to read the story in fmri tr."""
    story_path = path + story_name

    # open data, find story time, convert to int
    story = open(story_path, 'r').readlines()
    xmax_str = story[4]
    xmax_int = math.floor(float(xmax_str.split(' ')[-2])/tr)
    return(xmax_int)


def hrf_shift(embed, shift=4):
    """Func uses np.zeros to pad beginning of embeddings to
    account for fMRI delay for all layers"""
    em_shape = list(embed.shape)
    em_shape[1] = em_shape[1] + shift
    pad_embed = np.zeros(em_shape)
    pad_embed[:, shift : em_shape[1], :] = embed
    return(pad_embed)


def lz_resize(embed, new_shape):
    """Func to use Lanczos sampling to resize embedding to match fMRI timescale"""
    new_shape_2d = new_shape[1:][::-1]
    new_embed = map(lambda layer : cv2.resize(layer, new_shape_2d, interpolation=cv2.INTER_LANCZOS4), embed) # row, col reversed
    new_embed = np.array(list(new_embed))

    return(new_embed)


def get_fmri_mats(data_path, id_lst, fmri_names):
    """func takes in path to fmri data, list of ids and name of fmri files and returns data."""
    # load fmri mats
    all_story_mats = []
    for story in range(len(fmri_names)):
        # print(fmri_story_names[story])
        all_sub_mats = []
        for sub in range(len(id_lst)):
            fmri_dict = h5py.File(data_path + id_lst[sub] + '/' + fmri_names[story], 'r')
            fmri_mat = np.array(fmri_dict['data'])
            all_sub_mats.append(fmri_mat)

            # print(f'{en_subjs_id[sub]} = {fmri_mat.shape}')
        all_story_mats.append(all_sub_mats)
    return(all_story_mats)

if __name__ == '__main__':
    pass
    # print(hrf_shift(model_embeds[0], shift=2)  [:, 0:6, :])
