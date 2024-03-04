"""
Name: Kimberly Nestor
Class: CMU 10-733 NeuroAI
Date: 2/28/24
Description: Helper functions to extract embeddings from RoBERTa LLM layers after story input.
"""

import sys
import os, glob
from pathlib import Path
import itertools

import numpy as np
import math

import torch
import h5py
from transformers import RobertaTokenizer, RobertaModel, RobertaTokenizerFast

import nltk
from nltk.corpus import words
# nltk.download('words')

#### init Roberta model
# split words into tokens using spaces, first word == no space before, treated differently
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', use_fast=True, skip_special_tokens=True, truncation=True)
model = RobertaModel.from_pretrained('roberta-base')
config = model.config
config.output_hidden_states=True
n_hidden = config.num_hidden_layers

mdl_max = tokenizer.model_max_length # , model_max_length=math.inf
# sys.exit()


#### helper functions
# function to split string sentence into list of strings, split on space
s_split = lambda sentence : sentence.split()

# function takes in a sentence and a word and returns the idx of word in sentence
word_idx = lambda s_lst, word : s_lst.index(word)

# function to determine how llm will tokenize words, return dict
token_split = lambda s_lst  : {word : tokenizer.encode(word, add_special_tokens=False, truncation=True) for word in s_lst}


def splits_idx(splits):
    """Takes as input values from token_split dict and returns paired list with
    indexes of those values i.e. embeddings to convatenate."""
    splits = list(splits)
    t_idx_lst = []
    count = 0
    for i in splits:
        if count < mdl_max-2:
            t_len = len(i)
            if t_len > 1:
                ap_lst = list(range(count, count+t_len))
                t_idx_lst.append(ap_lst)
                count = ap_lst[-1]+1
            else:
                t_idx_lst.append([count])
                count+=1
        else:
            break
    return(t_idx_lst)


def mdl_embed(word, sentence):
    """
    Function that either takes a word as input only or a word with a context sentence.
    Returns embeddings of Roberta LLM model for the word by number of LLM layers.
    Combines split word tokens by averaging the embeddings from one word.
    Return if word only (sentence == None): model hidden embedding for just a word
    Return if sentence only (word == None): model hidden embedding for the entire sentence

    Return if word and context sentence (else): model hidden embedding for word only, hidden embedding for full sentence
    """
    if sentence == None: # word only
        # get embedding for single word, no context
        vec_input = tokenizer(word, return_tensors='pt', add_special_tokens=False, truncation=True)  # word2vec, encoded
        mdl_output = model(**vec_input)  # full mdl output, no logit
        mdl_hidden_embed = torch.cat(mdl_output.hidden_states)

        mdl_shape = list(mdl_hidden_embed.shape)
        # for each layer average the split word tokens
        avg_emb_lst = []
        if mdl_shape[1] > 1:
            for layer in mdl_hidden_embed:
                avg_emb = torch.mean(layer, dim=0)
                avg_emb_lst.append(avg_emb)

            mdl_shape[1] = -1  # infer no. words
            avg_mdl_hidden_embed = torch.cat(avg_emb_lst)
            avg_mdl_hidden_embed = torch.reshape(avg_mdl_hidden_embed, mdl_shape)
            return(avg_mdl_hidden_embed)
        else:
            return(mdl_hidden_embed)

    elif word == None: # sentence/ story only
        # input sentence into model and get full context embeddings for sentence
        vec_input = tokenizer(sentence, return_tensors='pt', add_special_tokens=False, truncation=True)  # word2vec, encoded
        mdl_output = model(**vec_input)  # full mdl output, no logit
        mdl_hidden_embed = mdl_output.hidden_states

        # get indxs for the tokens of words in the sentence
        splt_idx_lst = splits_idx(token_split(s_split(sentence)).values())
        # print(splt_idx_lst)
        # for each layer average the split word tokens
        avg_emb_lst = []
        for layer in mdl_hidden_embed:
            # this line taken from this link: https://discuss.pytorch.org/t/constructing-a-tensor-by-taking-mean-over-index-list-of-another-tensor/140193
            avg_emb = torch.stack([layer[:, i].mean(1) for i in splt_idx_lst], dim=1)
            avg_emb_lst.append(avg_emb)
        avg_mdl_hidden_embed = torch.cat(avg_emb_lst)
        return(avg_mdl_hidden_embed)

    else: # word embed with sentence context
        # input sentence into model and get full context embeddings for sentence
        vec_input = tokenizer(sentence, return_tensors='pt', add_special_tokens=False, truncation=True) # word2vec, encoded
        mdl_output = model(**vec_input)  # full mdl output, no logit
        mdl_hidden_embed = mdl_output.hidden_states

        # get indxs for the tokens of words in the sentence
        splt_idx_lst = splits_idx(token_split(s_split(sentence)).values())

        # for each layer average the split word tokens
        avg_emb_lst = []
        for layer in mdl_hidden_embed:
            # this line taken from this link: https://discuss.pytorch.org/t/constructing-a-tensor-by-taking-mean-over-index-list-of-another-tensor/140193
            avg_emb = torch.stack([layer[:, i].mean(1) for i in splt_idx_lst], dim=1)
            avg_emb_lst.append(avg_emb)
        avg_mdl_hidden_embed = torch.cat(avg_emb_lst)

        # word idx and model shape info
        w_idx = word_idx(s_split(sentence), word)
        mdl_shape = list(avg_mdl_hidden_embed.shape)
        mdl_shape[1] = -1 # infer no. words

        # get context embedding for input word
        word_embed = torch.cat([ll[w_idx] for ll in avg_mdl_hidden_embed])
        word_embed = torch.reshape(word_embed, mdl_shape)
        return(word_embed, avg_mdl_hidden_embed)


def get_story(path, story_name, save_path):
    """function that takes a path to a TextGrids file remove story words, 
    cleans data and returns a single strong string."""
    story_path = path + story_name
    # open data
    story = open(story_path, 'r').readlines()
    story_lst = [iii for iii in list(itertools.chain.from_iterable([ii.split('"') for ii in \
                                        [i for i in story if 'text' in i]])) if ' ' not in iii]
    story_lst = [ii for ii in [i.lower() for i in story_lst if 'sp' not in i] if ii in words.words()]

    # remove random beginning letters
    f_wrd = 0
    for i in range(len(story_lst)):
        if len(story_lst[i]) > 2:
            f_wrd = i
            break
    story_lst = story_lst[f_wrd:]
    story_str = ' '.join(story_lst)

    save_name = story_name.replace('.TextGrid', '')
    story_file = open(save_path+save_name, 'w')
    story_file.write(story_str)


# test funcs
if __name__ == '__main__':
    text_input = "Replace me by any text you'd like."
    test_words = ['tiger', 'lion', 'saturn']

    # all_embed_context = mdl_embed('by', text_input)
    # print(all_embed_context, all_embed_context[0].shape, all_embed_context[1].shape)

    # word_embed_only = mdl_embed(word=test_words[0], sentence=None)
    # print(word_embed_only, word_embed_only.shape)

    story_embed_only = mdl_embed(word=None, sentence=text_input)
    # print(story_embed_only, story_embed_only.shape)

    print(token_split(s_split(text_input)))


