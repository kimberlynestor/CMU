"""
Name: Kimberly Nestor
Date: 03/2022
Project: int_seg, bg_cb
Dataset: fMRI PIP
Description: This program creates an image of subject cofluctuations during tasks
coupled with registration to glm task conditions.
Based on Javi's code:
https://github.com/CoAxLab/cofluctuating-task-connectivity/blob/main/notebooks/03-analyse_RSS.ipynb
"""

import os
from os.path import join as opj
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as colors

import nibabel as nib
from nilearn import plotting
from nilearn.image import mean_img
# from nilearn import datasets
from nilearn.glm.first_level import compute_regressor

import jr_funcs as jr

# data path and subjects
main_dir = '/home/kimberlynestor/gitrepo/int_seg/data/'
d_path = 'pip_edge_ts/shen/'
dep_path = 'depend/'

d_path_ss = 'task-msit/'
subj_file = 'sub-4285_ses-01_task-msit_space-MNI152NLin2009cAsym_desc-edges_bold.nii.gz'

#plot data in browser
# plotting.view_img(mean_img(opj(dir, d_path, subj_file)), threshold=None)\
#     .open_in_browser()


#get efc for single subject
efc_vec = jr.extract_edge_ts(opj(main_dir, d_path, d_path_ss, subj_file))

# get subject list and task conditions
subj_lst = np.loadtxt(opj(main_dir, dep_path, 'subjects_intersect_motion_035.txt'))
df_task_events = pd.read_csv(opj(main_dir, dep_path, 'task-stroop_events.tsv'), sep="\t")


## HRF PREDICT
# timing for inter task rest
rest = df_task_events.iloc[0,0]

# make list of onset times for incongruent and congruent tasks
inc_cond = df_task_events[df_task_events.loc[:,"trial_type"]=="Incongruent"].to_numpy().T
inc_cond[2,:] = np.ones(len(inc_cond[0]))

con_cond = df_task_events[df_task_events.loc[:,"trial_type"]=="Congruent"].to_numpy().T
con_cond[2,:] = np.ones(len(con_cond[0]))

# onset times for inter trial 10sec rest
fix_cond = np.array( [df_task_events['onset'].to_numpy() - rest] + \
           [np.full(len(df_task_events['onset']), inc_cond[0][0])] + \
           [np.ones(len(df_task_events['onset']))] )
# print(fix_cond)

# timepoints of image capture
frame_times = np.arange(efc_vec.shape[0])*2.0

# create hrf prediction model of tasks
inc_regressor = np.squeeze(compute_regressor(inc_cond, hrf_model = "glover", \
                                              frame_times=frame_times)[0] )
con_regressor = np.squeeze(compute_regressor(con_cond, hrf_model = "glover", \
                                              frame_times=frame_times)[0] )
fix_regressor = np.squeeze(compute_regressor(fix_cond, hrf_model = "glover", \
                                              frame_times=frame_times)[0] )

# plot hrf graph
plt.plot(frame_times, inc_regressor, label="Incongruent")
plt.plot(frame_times, con_regressor, label="Congruent")
plt.plot(frame_times, fix_regressor, label="Fixation")
plt.xticks(np.arange(min(frame_times), max(frame_times), 100)) # 50
plt.legend()
plt.savefig('hrf_tasks.png', dpi=300)
plt.show()


## RSS
# make all subjects rss lst for tasks, save to npy
# for task in ["stroop", "msit", "rest"]:
#     rss_task_lst = jr.load_rss_task(opj(main_dir, d_path), task, subj_lst)
#     np.save(f'subjs_rss_{task}.npy', rss_task_lst)

# load saved npy with task rss
rss_stroop_lst = np.load('subjs_rss_stroop.npy')
rss_msit_lst = np.load('subjs_rss_msit.npy')
rss_rest_lst = np.load('subjs_rss_rest.npy')


## SIM GRAPH
fig  = plt.figure(figsize=(10, 7))
gs = gridspec.GridSpec(4, 1, height_ratios=[1.5, 2, 2.25, 9])

ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])
ax3 = fig.add_subplot(gs[3])

## plot task hrf
ax0.plot(con_regressor, label="Congruent")
ax0.plot(inc_regressor, label="Incongruent")
ax0.set_yticklabels([])
ax0.set_xticklabels([])
ax0.tick_params(size=0)
ax0.set_xlim([0,280])
ax0.legend(loc=4)


## plot single sine wave - estimate for integration, participation coeff
# plot x (time) and y (amplitude)
time = np.arange(0, 35, 0.00001)
amp = np.sin(time*0.69) # divide smaller have less amp
ax1.set_xticks([])
# ax1.set_yticks([])
ax1.plot(time, amp, '#b53737', label='Integration', linewidth=1.5)
ax1.legend(loc=4)


## plot destructive sinisoidal wave
# sampling rate
sr = 100.0
# sampling interval
ts = 1.0/sr
t = np.arange(0,1,ts)

# frequency of the signal
freq = 3.8 # chage no. amps - 1.5

# plot first wave
y = np.sin(2*np.pi*freq*t) #*5
ax2.plot(t, y, 'm', label='Basal ganglia')

# plot second wave
# y = 10*np.sin(2*np.pi*freq*t + 10)
y = np.sin(2*np.pi*freq*t + 9.5) #*5

ax2.plot(t, y, 'b', label='Cerebellum')
ax2.set_xticks([])
# ax3.set_yticks([])
ax2.legend(loc=4, prop={'size': 10})



## plot subjects heatmap
im3 = ax3.pcolormesh(np.array(rss_stroop_lst), cmap=plt.cm.RdYlBu_r, \
                     norm=colors.TwoSlopeNorm(vmin=50, vcenter=200, vmax=350))

cbar = fig.colorbar(im3, ax=ax3, orientation='horizontal', fraction=0.08, pad=0.3)
cbar.ax.tick_params(labelsize=11)
cbar.ax.set_ylabel("RSS", size=13, fontname="serif", rotation=90)

ax3.set_xticks(np.arange(30, 280, 30))
ax3.set_xticklabels((np.arange(30, 280, 30)*2.0).astype(int), size=18)
ax3.set_yticklabels([])
ax3.set_xlabel("Time (s)", size=25, fontname="serif")
ax3.set_ylabel("Subjects", size=25, fontname="serif")
ax3.set_xlim([0,280])
ax3.tick_params(labelsize=18)
# plt.suptitle("STROOP", size=40)

plt.tight_layout()
plt.savefig('sim_int_seg_graph_ax4_cbar.png', dpi=300)
plt.show()