"""
Name: Kimberly Nestor
Class: CMU 10-733 NeuroAI
Date: 2/28/24
Description: Helper functions to align fMRI data with words from story stimuli during scanning.
"""

import sys
import itertools

import nltk
from nltk.corpus import words

import math
import numpy as np

import cv2


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



if __name__ == '__main__':
    print(hrf_shift(model_embeds[0], shift=2)  [:, 0:6, :])
