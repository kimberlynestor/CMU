"""
Name: Kimberly Nestor
Class: CMU 10-733 NeuroAI
Date: 2/28/24

Description: Averages regression weights for one instance over time,
compare this vector to see which words from a story are most similar to coefficient vector.

Disclosures:
"""

import sys
import os

import numpy as np
from numpy.linalg import norm
import torch

from fmri_funcs import *
from mdl_funcs import *


# dir info for data
curr_dir = os.getcwd()
story_save_path = curr_dir + '/clean_stories/'
mdl_output_dir = curr_dir + '/mdl_train_output/'
pred_output_dir = curr_dir + '/fmri_pred_output/'
embed_save_path = curr_dir + '/embeddings/'


#### 1.4 Interpreting encoding model
# subj UTS01, story avatar, ridge regression weights, avg time
mdl_coeffs = np.load(mdl_output_dir + 'avatar_coeffs_1.npy')
mdl_coeffs_avg = np.mean(mdl_coeffs, axis=0)
# print(mdl_coeffs_avg, '\n')

# test prediction of wherethereissmoke, subj UTS01, avg time
pred_output = np.load(pred_output_dir + 'wheretheressmoke_pred1.npy')
pred_output_avg = np.mean(pred_output, axis=0)
# print(pred_output_avg, '\n')

# resize coeffs from square
coeff_output_rz = lz_resize(mdl_coeffs, pred_output.shape)
# print( pred_output.shape)


# get org RoBERTa mdl embeddings
mdl_embed = torch.load(embed_save_path + 'avatar').detach().numpy()[-1]

# find top 3 words that best match ridge regression coeffs output vector (averaged)
# best_match = np.array(list(map(lambda word: np.dot(word,mdl_coeffs_avg)/(norm(word)*norm(mdl_coeffs_avg)), mdl_embed)))
best_match = list(map(lambda word: np.dot(word,mdl_coeffs_avg), mdl_embed))
idx_words = sorted([best_match.index(i) for i in sorted(best_match)[:3]]) # , reverse=True

# print(f'\nidx top 3 match words (coeffs, embed) = {idx_words}\n')
# idx top 3 match words (coeffs, embed) = [6, 34, 330] # idx of mdl embedding vector (truncated)

# load clean avatar story
story = open(story_save_path + 'avatar', 'r').read()
story_lst = story.split(' ')


# init RoBERTa model
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', use_fast=True, skip_special_tokens=True, truncation=True)
model = RobertaModel.from_pretrained('roberta-base')

# loop to pass each token from the story, avg > 1 token, compare token using correlation to truncated mdl embed vector best_match words
best_match_str = []
best_match_idx = []
for idx in range(len(idx_words)):
    corr_lst = []
    for word in range(len(story_lst)):
        vec_input = tokenizer(story_lst[word], return_tensors='pt', add_special_tokens=False, truncation=True)  # word2vec, encoded
        mdl_output = model(**vec_input)

        mdl_last_embed = mdl_output.last_hidden_state #.detach().numpy() # .ravel()

        if mdl_last_embed.shape[1] > 1:
            mdl_last_embed = mdl_last_embed.mean(1)
            # print(mdl_last_embed.shape)

        mdl_last_embed = mdl_last_embed.detach().numpy().ravel()

        cor = np.corrcoef(mdl_last_embed, mdl_embed[idx_words[idx]])[0][1]
        corr_lst.append(cor)

    max_corr = corr_lst.index(max(corr_lst))
    next_max_corr = corr_lst.index(sorted(corr_lst, reverse=True)[1])
    # print(f'best match = {max_corr}')
    # print(f'next best match = {next_max_corr}')
    best_match_str.append(story_lst[max_corr])
    best_match_idx.append(max_corr)

# print best match words and next best match words
print(f'\nidx of top 3 best match words (coeffs, embed) = {best_match_idx}\n')
print(f'\ntop 3 best match words (coeffs, embed) = {best_match_str}\n')


# wrd_lst = [878, 995, 994, 437, 25]
# two_last_lst = [23, 994, 437, 995, 86, 25]
# print(f'last words = {[story_lst[i] for i in wrd_lst]}')

