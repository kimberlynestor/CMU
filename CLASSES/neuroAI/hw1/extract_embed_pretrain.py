"""
Name: Kimberly Nestor
Class: CMU 10-733 NeuroAI
Date: 2/28/24

Description: Use hugging face pretrain LLM RoBERTa model to extract embeddings after running stories through the model.

Disclosures: I used the following links for help in this hw.
https://huggingface.co/FacebookAI/roberta-base
https://huggingface.co/docs/transformers/model_doc/roberta
https://huggingface.co/transformers/v3.2.0/model_doc/roberta.html
https://discuss.huggingface.co/t/generate-raw-word-embeddings-using-transformer-models-like-bert-for-downstream-process/2958
https://www.reddit.com/r/machinelearningmemes/comments/tfupuo/how_to_combine_between_embedding/
"""

import sys
import os, glob
from pathlib import Path

import pandas as pd
import numpy as np


# import cortex
import torch
from transformers import RobertaTokenizer, RobertaModel, RobertaTokenizerFast

from mdl_funcs import *


# dir info for data
curr_dir = os.getcwd()
dir_depth = '/recordings/ds003020/derivative'
text_data_dir = curr_dir + dir_depth + '/TextGrids/' # 8 storeis for train, wheretheressmoke for test
story_save_path = curr_dir + '/clean_stories/'
embed_save_path = curr_dir + '/embeddings/'
test_word_save_path = curr_dir + '/test_word_embed/'

# get all story file names
story_files = sorted(os.listdir(text_data_dir))

# train and test names
train_lst = story_files[0:8]
test_name = story_files[-2] # 1 item

# clean story data and save to text file
# get_story(text_data_dir, test_name, story_save_path)
# [get_story(text_data_dir, i, story_save_path) for i in train_lst]

# load clean stories - run train, get hidden embed
clean_story_names = sorted(os.listdir(story_save_path))
all_stories = [open(story_save_path+i, 'r').read() for i in clean_story_names]
train_stories = [open(story_save_path+i, 'r').read() for i in clean_story_names[0:-1]]
test_story = open(story_save_path+clean_story_names[-1], 'r').readlines()


#### 1.1 Extract stimuli featurs from a pretrained deep network (RoBERTa)
# extract final three layers for each story and torch save
"""
for i in range(len(all_stories)):
    all_embed = mdl_embed(word=None, sentence=all_stories[i])
    last_three = all_embed[-3:, :]
    # print(last_three.shape)
    torch.save(last_three, embed_save_path+clean_story_names[i])
"""

# torch load story embeddings
"""
for i in range(len(clean_story_names)):
    embed = torch.load(embed_save_path+clean_story_names[i])
    print(f'{embed.shape} = {clean_story_names[i]}')
"""

# check L2 norm for model layers for test words
test_words = ['tiger', 'lion', 'saturn']
test_sentence = 'can you see the big beautiful '

word_embed_tiger = mdl_embed(word=test_words[0], sentence=None)
word_embed_lion = mdl_embed(word=test_words[1], sentence=None)
word_embed_saturn = mdl_embed(word=test_words[2], sentence=None)

# torch.save(word_embed_tiger, test_word_save_path + 'tiger')
# torch.save(word_embed_lion, test_word_save_path + 'lion')
# torch.save(word_embed_saturn, test_word_save_path + 'saturn')

# text_input = "Replace me by any text you'd like."
# all_embed_context = mdl_embed('any', text_input)


# tiger and lion
print(f'\nTiger and Lion Euclidean distance (L2 norm)')
for l_a, l_b, i in zip(word_embed_tiger.detach().numpy(), word_embed_lion.detach().numpy(), range(1, len(word_embed_tiger))):
    euc_dist = np.linalg.norm(l_a - l_b) # l2 norm
    print(f' layer{i} = {euc_dist}')

# tiger and saturn
print(f'\nTiger and Saturn Euclidean distance (L2 norm)')
for l_a, l_b, i in zip(word_embed_tiger.detach().numpy(), word_embed_saturn.detach().numpy(), range(1, len(word_embed_tiger))):
    euc_dist = np.linalg.norm(l_a - l_b) # l2 norm
    print(f' layer{i} = {euc_dist}')


