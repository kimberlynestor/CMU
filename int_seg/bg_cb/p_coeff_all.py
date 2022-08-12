"""


"""

from os.path import join as opj
import sys
import itertools

import bct
import jr_funcs as jr

import numpy as np
import pandas as pd
import math
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.lines import Line2D

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_columns', None)


p_dict = {'Incongruent':'#c6b5ff', 'Congruent':'#a5cae7', 'Fixation':'#ffbcd9', \
          'Difference':'#8A2D1C', 'Incongruent Fixation':'#c6b5ff', 'Congruent Fixation':'#a5cae7'}

# data path and subjects
main_dir = '/home/kimberlynestor/gitrepo/int_seg/data/'
d_path = 'pip_edge_ts/shen/'
dep_path = 'depend/'

# get subject list and task conditions
subj_lst = np.loadtxt(opj(main_dir, dep_path, 'subjects_intersect_motion_035.txt'))
df_task_events = pd.read_csv(opj(main_dir, dep_path, 'task-stroop_events.tsv'), sep="\t")

# shen parcels region assignment - brain areas from Andrew Gerlach
net_assign = pd.read_csv(opj(main_dir, dep_path, 'shen_268_parcellation_networklabels_mod.csv')) .iloc[:,1] .values
# cort_assign = np.array(list(filter(lambda i: i[1] != 4, net_assign)))
# c_idx = [i[0] for i in cort_assign]


task = 'stroop'
efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj_lst[0])  # all networks, sing sub

# sys.exit()

#######
# part coeff, single subj one timepoint, ci from shen atlas node info
p_coef = bct.participation_coef_sign(efc_mat[0], net_assign)

# print(p_coef)
# print(efc_mat[0])
# sys.exit()

# part coeff, single subject all timepoint
p_coef_all = list(map( lambda x: np.average(bct.participation_coef_sign(x, net_assign)), efc_mat))
# p_coef_all = list(map( lambda x: gaussian_filter(np.average(bct.participation_coef_sign(x, net_assign)), sigma=1), efc_mat))


# plot part coeff, all timepoint
# plt.plot(np.arange(0, 280*2, 2), p_coef_all, linewidth=1)
plt.plot(np.arange(0, 280*2, 2), gaussian_filter(p_coef_all, sigma=1), linewidth=1)

plt.xticks(np.arange(0, 280*2, 60))
plt.xlabel("Time (s)", size=11, fontname="serif")
plt.ylabel("Participation coefficient", size=11, fontname="serif") # integration
# plt.savefig('sing_sub_p_coef_all.png', dpi=300)
plt.show()


