"""
Name: Kimberly Nestor
AdvML Homework 1
Date:
Description: Train a perception to classify MNIST 7 and 9 images.
"""


import numpy as np
import matplotlib.pyplot as plt
from perceptron import *

import sys
import os
from os.path import join as opj
from pathlib import Path

# add sample scripts to path
curr_dir = os.getcwd()
sys.path.insert(0, opj(curr_dir, 'hw1_starter_code'))

from data import *
import urllib.request

np.set_printoptions(threshold=sys.maxsize)


# download data from online
mnist_lst = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
# training set images, training set labels, test set images, test set labels

get_alldata = lambda lst : [maybe_download(i) for i in lst]
get_alldata(mnist_lst)

# get train, test data
mnist_train = extract_data(mnist_lst[0])
mnist_test = extract_data(mnist_lst[2])

# plt.imshow(mnist_train[2])
# plt.show()

# get data labels
mnist_train_lbl = extract_labels(mnist_lst[1])
mnist_test_lbl = extract_labels(mnist_lst[3])

# filter only 7, and 9
get_fil = lambda check_lst, lst: list(map(check_lst.__contains__, lst))

train_fil = get_fil([7,9], mnist_train_lbl)
test_fil = get_fil([7,9], mnist_test_lbl)


# get train, test data - filtered, reduced
mnist_train_fil = mnist_train[train_fil][:500]
mnist_test_fil = mnist_test[test_fil][:500]

# get data labels - filtered, reduced
mnist_train_lbl_fil = mnist_train_lbl[train_fil][:500]
mnist_test_lbl_fil = mnist_test_lbl[test_fil][:500]

# plot MNIST - single
# for img, lbl in zip(mnist_train_fil, mnist_train_lbl_fil):
#     print(lbl)
#     plt.imshow(img)
#     plt.show()

# plot MNIST - grid, 25 train
fig, ax = plt.subplots(5, 5)
ax = ax.flatten()

for i, ax in zip(range(25), ax):
    ax.imshow(mnist_train_fil[i])
    ax.axis('off')
    if not os.path.exists('./output'):
        os.mkdir('./output')
fig.suptitle('MNIST train first 25 images (7 and 9 only)')
plt.savefig('output/mnist_grid_img.png', dpi=300)
# plt.show()
plt.close()


# flatten imgs to vectors len 784
mnist_train_fil_flat = np.array(list(map(lambda i: np.ravel(i), mnist_train_fil)))
mnist_test_fil_flat = np.array(list(map(lambda i: np.ravel(i), mnist_test_fil)))
dim = mnist_train_fil_flat.shape[-1]

# plot hist: 7, 9
plt.hist(mnist_train_lbl_fil, bins=2)
plt.xticks([7,9])
plt.xlabel('MNIST train image label', fontweight='bold')
plt.ylabel('Quantity', fontweight='bold')
plt.title('Histogram of training labels')
plt.savefig('output/mnist_hist.png', dpi=300)
# plt.show()
plt.close()


# transform labels:  7 = -1,   9 = +1
mnist_train_lbl_fil_tranf = np.array([-1 if i==7 else 1 for i in mnist_train_lbl_fil])
mnist_test_lbl_fil_tranf = np.array([-1 if i==7 else 1 for i in mnist_test_lbl_fil])

per = Perceptron(dim)
update = per.update(mnist_train_fil_flat[0], mnist_train_lbl_fil_tranf[0])
pred = per.predict(mnist_train_fil_flat)
train = per.train(mnist_train_fil_flat, mnist_train_lbl_fil_tranf, mnist_test_fil_flat, mnist_test_lbl_fil_tranf, 2000)
fin_train_acc = train['train'][-1]
fin_test_acc = train['test'][-1]

# plot training trajectories
plt.plot(train['train'])
plt.xlabel('Iterations', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.title('Perceptron training trajectories')
plt.savefig('output/training_trajs.png', dpi=300)
plt.show()

# plot testing trajectories
plt.plot(train['test'])
plt.xlabel('Iterations', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.title('Perceptron testing trajectories')
plt.savefig('output/testing_trajs.png', dpi=300)
plt.show()

print(f'\nFinal train accuracy: {fin_train_acc}')
print(f'Final test accuracy: {fin_test_acc}\n')