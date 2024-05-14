"""
Name: Kimberly Nestor
Class: CMU 10-733 NeuroAI Project
Date: 5/12/24

Description: Used reprocessed Things images into foreground only images and background only images along with original
             images as input to ResNet18. Get final layers embeddings.

Resources:
https://elifesciences.org/articles/82580
https://things-initiative.org/
https://things-initiative.org/projects/things-images/
https://plus.figshare.com/collections/THINGS-data_A_multimodal_collection_of_large-scale_datasets_for_investigating_object_representations_in_brain_and_behavior/6161151
"""

import os
import sys
from tqdm import tqdm

from PIL import Image
import torch
from torchvision import transforms
from transformers import ResNetModel

import numpy as np
import pandas as pd

# pd.set_option('display.max_rows', 2000)
# pd.set_option('display.max_columns', 50)
# pd.set_option('display.width', 1000)
# np.set_printoptions(threshold=sys.maxsize)


# dir info for data
curr_dir = os.getcwd()
stim_img_path = curr_dir + '/images/'
nb_img_path = curr_dir + '/images_nb/'
bg_img_path = curr_dir + '/images_bg/'

nb_mask_path = curr_dir + '/nb_mask/'
nf_img_path = curr_dir + '/images_nf/'

meg_path = curr_dir + '/THINGS-MEG/preprocessed/'
embed_path = curr_dir + '/embeddings/'


# get meg data info
meg_stim_info = [i for i in sorted(os.listdir(meg_path)) if 'eyes' in i]
meg_info_pd = pd.read_csv(meg_path + meg_stim_info[0])

# only have epoch 0-4 data, first test is in epoch 14
meg_stim_order = [i[11:] for i in meg_info_pd[meg_info_pd['epoch'].isin([0,1,2,3])]['image_path'].values]

# meg_info_pd_test = meg_info_pd[meg_info_pd['image_path'].str.contains('test')]
# meg_test_epochs = np.unique(meg_info_pd_test['epoch'].values)

# timing of stimulus
meg_timing = meg_info_pd['time'].values[:1681]


# original images
stim_img_cats = sorted(os.listdir(stim_img_path))[2:]


# loop to get original images embeddings
for i in tqdm(range(len(stim_img_cats))):
    print(stim_img_cats[i])

    # select only first 12 images from each category
    meg_stim_imgs = sorted(os.listdir(f'{stim_img_path}{stim_img_cats[i]}/'))[:12]

    for ii in range(len(meg_stim_imgs)):

        # select only first 12 images from each category
        meg_stim_imgs = sorted(os.listdir(f'{stim_img_path}{stim_img_cats[i]}/'))[:12]
        input_image = Image.open(f'{stim_img_path}{stim_img_cats[i]}/{meg_stim_imgs[ii]}')

        # init model, get output
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model = ResNetModel.from_pretrained("microsoft/resnet-50")

        # resize and normalize images
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch, output_hidden_states=True)

# get last hidden state
second_hidden = torch.reshape(output.hidden_states[1][0], [2048,-1])
second_hidden = second_hidden.detach().numpy()
np.save(f'{embed_path}org_img_embed.npy', second_hidden)



# nb background images
stim_img_cats = sorted(os.listdir(nb_img_path)) [1:]

# loop to get img no background images embeddings
for i in tqdm(range(len(stim_img_cats))):
    print(stim_img_cats[i])

    # select only first 12 images from each category
    meg_stim_imgs = sorted(os.listdir(f'{stim_img_path}{stim_img_cats[i]}/'))[:12]

    for ii in range(len(meg_stim_imgs)):

        # select only first 12 images from each category
        meg_stim_imgs = sorted(os.listdir(f'{stim_img_path}{stim_img_cats[i]}/'))[:12]
        input_image = Image.open(f'{stim_img_path}{stim_img_cats[i]}/{meg_stim_imgs[ii]}')

        # init model, get output
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model = ResNetModel.from_pretrained("microsoft/resnet-50")

        # resize and normalize images
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch, output_hidden_states=True)

# get last hidden state
second_hidden = torch.reshape(output.hidden_states[1][0], [2048,-1])
second_hidden = second_hidden.detach().numpy()
np.save(f'{embed_path}nb_img_embed.npy', second_hidden)


# nf background images
stim_img_cats = sorted(os.listdir(nf_img_path)) [1:]

# loop to get img no background images embeddings
for i in tqdm(range(len(stim_img_cats))):
    print(stim_img_cats[i])

    # select only first 12 images from each category
    meg_stim_imgs = sorted(os.listdir(f'{stim_img_path}{stim_img_cats[i]}/'))[:12]

    for ii in range(len(meg_stim_imgs)):

        # select only first 12 images from each category
        meg_stim_imgs = sorted(os.listdir(f'{stim_img_path}{stim_img_cats[i]}/'))[:12]
        input_image = Image.open(f'{stim_img_path}{stim_img_cats[i]}/{meg_stim_imgs[ii]}')

        # init model, get output
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model = ResNetModel.from_pretrained("microsoft/resnet-50")

        # resize and normalize images
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch, output_hidden_states=True)

# get last hidden state
second_hidden = torch.reshape(output.hidden_states[1][0], [2048,-1])
second_hidden = second_hidden.detach().numpy()
np.save(f'{embed_path}nf_img_embed.npy', second_hidden)
