"""
Name: Kimberly Nestor
Date: 03/2022
Project: int_seg, bg_cb
Description: This program

Participation coefficient (signed): https://tinyurl.com/2p89y43a
Modularity maximization: https://tinyurl.com/4pazk7bz
Mod Max e.g.: https://tinyurl.com/3jkf2zjj
"""

import os
from os.path import join as opj
import sys
from operator import itemgetter

import bct
import netneurotools
from netneurotools import modularity as mod
from netneurotools import plotting
from netneurotools import cluster

import jr_funcs as jr

import numpy as np
import pandas as pd
import math
from statsmodels.tsa.api import VAR

import ruptures as rpt
from scipy.ndimage import gaussian_filter
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
# from scipy.stats import zscore
from scipy import stats


from nilearn.image import load_img
from nilearn.glm.first_level import compute_regressor

import networkx as nx
from networkx import path_graph, random_layout
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import ptitprince as pt

# import tensorflow as tf # used pip
import torch # used conda
import zarr # think conda

# np.set_printoptions(threshold=sys.maxsize)
# pd.set_option('display.max_rows', None)

from scipy import stats
rng = np.random.default_rng()
rvs = stats.norm.rvs(loc=5, scale=10, size=(50, 2), random_state=rng)


# data path and subjects
main_dir = '/home/kimberlynestor/gitrepo/int_seg/data/'
d_path = 'pip_edge_ts/shen/'
dep_path = 'depend/'

rt = 2
frs = 280

# get subject list and task conditions
subj_lst = np.loadtxt(opj(main_dir, dep_path, 'subjects_intersect_motion_035.txt'))
df_task_events = pd.read_csv(opj(main_dir, dep_path, 'task-stroop_events.tsv'), sep="\t")

"""
# get efc mat for all subjects
for task in ['stroop', 'msit', 'rest']: # mem issue with 100% data
    efc_mat_allsub = jr.get_efc_trans(opj(main_dir, d_path), task, \
                                      subj_lst [0:math.floor(len(subj_lst)*0.7)] ) # 70% data
    np.save(f'subjs_efc_mat_{task}.npy', efc_mat_allsub)
"""

########## - stroop
# cause of mem had to save tasks individually
task = 'rest' # stroop
# efc_mat_allsub = jr.get_efc_trans(opj(main_dir, d_path), task, \
#                                   subj_lst[0:math.floor(len(subj_lst)*0.75)])
# np.save(f'subjs_efc_mat_{task}.npy', efc_mat_allsub)


# load saved npy with task coalesce matrices
mat_stroop_all = np.load('subjs_efc_mat_stroop.npy')

# do mod max on all subjs
# mod_max_stroop_all = np.array(list(map(lambda subj: np.array(list(map(lambda mat: \
#                         bct.community_louvain(mat, gamma=1.5, B='negative_sym'), \
#                             subj)), dtype=object), mat_stroop_all)), dtype=object)
# np.save(f'subjs10_mod_max_{task}.npy', mod_max_stroop_all)
# np.save(f'subjs_all_mod_max_{task}.npy', mod_max_stroop_all)


# load mod max for all subjs
# mod_max_stroop_all = np.load('subjs10_mod_max_stroop.npy', allow_pickle=True)
mod_max_stroop_all = np.load('subjs_all_mod_max_stroop.npy', allow_pickle=True)

# separate out communities and index for all subjs
ci_allsub = np.array(list(map(lambda subj: np.array(list(map(lambda x:x[0], subj))), \
                                                mod_max_stroop_all)))
q_allsub = np.array(list(map(lambda subj: np.array(list(map(lambda x:x[1], subj))), \
                                                mod_max_stroop_all)))

ci_avg_sub = list(map(lambda time: list(map(lambda x:math.floor(x), time)), \
                      np.average(ci_allsub, axis=0)))

# avg q across subjs
q_avg = np.average(q_allsub, axis=0)
ci_avg = np.average(ci_allsub, axis=0)
frames = q_avg.shape[0]


# smooth q using moving average
# df_q = pd.DataFrame(q_avg, columns=['Q_avg'])
# df_q['Q_smooth'] = df_q['Q_avg'].rolling(2).sum()
# q_smooth = np.nan_to_num( df_q['Q_smooth'].to_numpy() )

# smooth q using gaussian filter, mask first few vals
q_smooth = gaussian_filter(q_avg, sigma=2.5)
q_smooth_mask = np.ma.masked_where(q_smooth < 0.446, q_smooth)
q_smooth_mask_init = np.ma.array(q_smooth, mask=np.pad(np.ones(5), (0,frs-5)))


##############
# plot mod max, all timepoint, all subjs
plt.plot(np.arange(0, 280*rt, 2), q_avg, linewidth=1)
plt.xticks(np.arange(0, 280*rt, 60))
plt.xlabel("Time (s)", size=11, fontname="serif")
plt.ylabel("Modularity (Q)", size=11, fontname="serif")
# plt.savefig('allsub_mod_qall.png', dpi=300)
plt.show()

# plot mod max, all timepoint, all subjs - smooth
plt.plot(np.arange(0, 280*rt, 2), q_smooth, linewidth=1)
plt.xticks(np.arange(0, 280*rt, 60))
plt.xlabel("Time (s)", size=11, fontname="serif")
plt.ylabel("Modularity (Q)", size=11, fontname="serif")
# plt.savefig('allsub_mod_qall_smooth_sig2p5.png', dpi=300)
plt.show()
##############


"#############"
## change point detection - binary segmentation algorithm
model = 'l1'  #'l1', 'l2', 'rbf'
algo = rpt.Binseg(model=model, min_size=8, jump=5).fit(q_avg) # Pelt, Dynp, Window
q_bkps = algo.predict(n_bkps=15)

# plot breakpoints
rpt.display(q_smooth, q_bkps, figsize=(10, 6))
# plt.title('Change Point Detection: Binary Segmentation Search Method', \
#               fontdict={'fontsize' : 10})
# plt.savefig('bkps_binseg.png', dpi=300)
plt.show()


# get break sections
frame_bkps = np.insert( np.array(q_bkps), 0, 0 )
bk_secs = list(zip(frame_bkps[:-1], frame_bkps[1:]))

rest_secs = bk_secs[::2]
inc_secs = bk_secs[1::2][::2]
con_secs = bk_secs[1::2][1::2]

# separate out q for task blocks
q_inc = list(map(lambda x: q_avg[x[0]:x[1]], inc_secs))
q_con = list(map(lambda x: q_avg[x[0]:x[1]], con_secs))

q_inc_all = [ii for i in q_inc for ii in i]
q_con_all = [ii for i in q_con for ii in i]

# ttest for inc and con mod idx
ttest = stats.ttest_ind(q_inc_all, q_con_all)
# print(f'\nIndependent paired-t test for Q incongruent vs congruent:\n   {ttest}\n')


# plot mod max, all timepoint, all subjs - smooth, with breakpoints
plt.plot(np.arange(0, 280*rt, 2), q_smooth_mask, linewidth=1)
plt.xticks(np.arange(0, 280*rt, 60))
plt.xlabel("Time (s)", size=15, fontname="serif")
plt.ylabel("Modularity (Q)", size=15, fontname="serif")

# colour plot background with breakpoints
for i in range(len(bk_secs)):
    if i in range(len(bk_secs))[::2]:  # rest
        # plt.axvspan(bk_secs[i][0]*rt, bk_secs[i][1]*rt, facecolor='tab:green', alpha=0.2)
        pass
    elif i in range(len(bk_secs))[1::2][::2]:  # incongruent
        plt.axvspan(bk_secs[i][0]*rt, bk_secs[i][1]*rt, facecolor='tab:orange', alpha=0.22)
    else:  # congruent
        plt.axvspan(bk_secs[i][0]*rt, bk_secs[i][1]*rt, facecolor='tab:blue', alpha=0.2)
# plt.xlim(10)
plt.ylim(0.436, 0.466)
plt.tight_layout()
# plt.savefig('allsub_mod_qall_smooth_sig2p5_bkps.png', dpi=300)
plt.show()
"#############"


## task block onset
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

# task timing blocks - modularity
inc_block = list(zip(inc_cond[0], inc_cond[0]+inc_cond[1][0]))
con_block = list(zip(con_cond[0], con_cond[0]+con_cond[1][0]))


##############
# legend patches
inc_patch = mpatches.Patch(color='tab:orange', label='Incongruent', alpha=0.35)
con_patch = mpatches.Patch(color='tab:blue', label='Congruent', alpha=0.35)

## plot mod max, all timepoint, all subjs - smooth, with task blocks
plt.plot(np.arange(0, 280*rt, 2), q_smooth_mask_init, linewidth=1)
plt.xticks(np.arange(0, 280*rt, 60))
plt.xlabel("Time (s)", size=15, fontname="serif")
plt.ylabel("Modularity (Q)", size=15, fontname="serif")

# colour plot background with breakpoints
for i in range(len(inc_block)):
    # incongruent
    plt.axvspan(inc_block[i][0], inc_block[i][1], facecolor='tab:orange', alpha=0.22)
    # congruent
    plt.axvspan(con_block[i][0], con_block[i][1], facecolor='tab:blue', alpha=0.2)
plt.ylim(0.436, 0.466) # mask
# plt.ylim(0.423, 0.466) # no mask
plt.legend(handles=[inc_patch, con_patch], loc=4)
plt.tight_layout()
plt.savefig('allsub_mod_qall_smooth_sig2p5_blocks.png', dpi=300)
plt.show()
##############


# timepoints of image capture
frame_times = np.arange(frames)*2.0

# create hrf prediction model of tasks
inc_regressor = np.squeeze(compute_regressor(inc_cond, hrf_model = "glover", \
                                              frame_times=frame_times)[0] )
con_regressor = np.squeeze(compute_regressor(con_cond, hrf_model = "glover", \
                                              frame_times=frame_times)[0] )
fix_regressor = np.squeeze(compute_regressor(fix_cond, hrf_model = "glover", \
                                              frame_times=frame_times)[0] )

# make regressor df
df_mod_reg = pd.DataFrame(np.column_stack([q_avg, inc_regressor, fix_regressor, con_regressor]),  \
                          columns=['mod_avg', 'inc_reg', 'fix_reg', 'con_reg'])


"""
# fit multiple linear regression model to q for all subs
mlr_coeffs_lst = []
for i in q_allsub:
    # lin reg model
    mlr = LinearRegression()
    # mlr.fit(df_mod_reg[['inc_reg', 'fix_reg', 'con_reg']], df_mod_reg['mod_avg'])
    mlr.fit(df_mod_reg[['inc_reg', 'fix_reg', 'con_reg']], i)

    # slope
    mlr_slope = mlr.intercept_
    # coeffs, best fit line
    mlr_coeffs = mlr.coef_

    mlr_coeffs_lst.append(mlr_coeffs)
# print(mlr_coeffs_lst)
"""


# plot hist with kde - all subj for each timepoint
# for i in range(len(q_allsub.T)):
#     # original
#     plt.figure(figsize=(8, 8))
#     sns.histplot(q_allsub.T[i], kde=True)
#     plt.title(f'{i}')
#     plt.show()
#
#     # transform
#     plt.figure(figsize=(8, 8))
#     sns.histplot(stats.boxcox(q_allsub.T[i])[0], kde=True) # displot or histplot
#     plt.title(f'{i}')
#     plt.show()

mod_max_rest_all = np.load('subjs_all_mod_max_rest.npy', allow_pickle=True)

# resting state data modularity
q_allsub_rest = np.array(list(map(lambda subj: np.array(list(map(lambda x:x[1], subj))), \
                                                mod_max_rest_all)))


# normalize data - z score, boxcox transform
q_allsub_znorm = np.array(list(map(lambda x: stats.zscore(x), q_allsub)))
q_allsub_boxcox = np.array(list(map(lambda x: stats.boxcox(x)[0], q_allsub.T))).T

q_allsub_znorm_rest = np.array(list(map(lambda x: stats.zscore(x), q_allsub_rest)))


# make data frame with all subjects and timepoints
df_q_all = pd.DataFrame(q_allsub) # q_allsub_znorm
df_q_all_c = df_q_all.copy(deep=True)
df_q_all_c.insert(0, 'subj_ID', [int(i) for i in subj_lst[0:math.floor(len(subj_lst)*0.75)]])
df_q_all_c = pd.melt(df_q_all_c, id_vars='subj_ID', var_name='frame', value_name='mod_idx')

# merge all sub all frame mod_idx with regressors
df_q_all_reg = df_q_all_c.merge(df_mod_reg, how='inner', left_on='frame', \
                                right_index=True, sort=False)
df_q_all_reg = df_q_all_reg.drop(columns=['mod_avg', 'fix_reg'])

### between subject test
# autoregression model
ar_model = AutoReg(df_q_all_reg['mod_idx'], exog=df_q_all_reg[['inc_reg', 'con_reg']], lags=frs).fit()
ar_weights = ar_model.params[1:-2]
ar_coeffs = ar_model.params[-2:]

rest_mean = np.average(q_allsub_znorm_rest)
ttest_1samp_between = stats.ttest_1samp(ar_weights, rest_mean)

print(f'\nResting state mean: {rest_mean}')
print(f'One samp t-test: {ttest_1samp_between} \n')
print("Model coefficients:")
print(round(ar_coeffs, 4), "\n")

print(ar_model.summary())
sys.exit()



### save and load mlr coeffs
# np.save('mlr_coeffs_allsub.npy', mlr_coeffs_lst)
mlr_coeffs_allsub = np.load('mlr_coeffs_allsub.npy', allow_pickle=True)
df_coeffs = pd.DataFrame(mlr_coeffs_allsub, columns=['inc_coeff', 'fix_coeff', 'con_coeff'])

### within subject test
# con inc diff vec, mean, std err
diff_con_inc = np.squeeze(list(map(lambda x:[x[2] - x[0]], mlr_coeffs_allsub)))
mean_diff = np.mean(diff_con_inc)
std_err_diff = stats.sem(diff_con_inc)

ttest_1samp_within = stats.ttest_1samp(diff_con_inc, mean_diff)
# print('congruent and incongruent', ttest_1samp, '\n')


# independent paired ttest
# ttest_ind = stats.ttest_ind(df_coeffs['inc_coeff'].values, df_coeffs['con_coeff'].values)
# print(ttest_1samp.pvalue, ttest_1samp.statistic)


##########
## boxplots
df_coeffs.boxplot(column=['inc_coeff', 'con_coeff'])
plt.ylabel("Beta coefficients", size=15, fontname="serif")
plt.savefig('boxplots_bcoeff.png', dpi=300)
plt.show()


######## matrix visualization - incongruent
inc_block_frames = np.array(inc_block)/2
# inc_frames_idx = [ii for i in list(map(lambda x: list(range(int(x[0]), \
#                                 int(x[1]))), inc_block_frames)) for ii in i]
inc_frames_idx = list(range(int(inc_block_frames[2][0]), int(inc_block_frames[2][1])))

# get efc matrices for inc timepoints
inc_mat_alltime_allsub = list(map(lambda subj: itemgetter(*inc_frames_idx)(subj), \
                                  mat_stroop_all ))
inc_mat_avgtime_allsub = list(map(lambda subj:np.average(subj, axis=0), inc_mat_alltime_allsub))

inc_mat_avg_time_sub = np.average(inc_mat_avgtime_allsub, axis=0)

# ci community map for averaged corr time subj matrix
ci_inc_avg = bct.community_louvain(inc_mat_avg_time_sub, gamma=1.5, B='negative_sym')
# ci_inc_avg = [math.floor(i) for i in np.average(itemgetter(*inc_frames_idx)(ci_avg_sub), axis=0)]
ci_inc = itemgetter(*inc_frames_idx)(ci_avg_sub)


# plot inc corr matrix
fig, ax = plt.subplots(1, 1)
coll = ax.imshow(inc_mat_avg_time_sub, vmin=-1, vmax=1, cmap='viridis') # cmap=plt.cm.RdYlBu_r
# ax.set(xticklabels=[], yticklabels=[])
fig.colorbar(coll)
plt.title('Incongruent', fontdict={'fontsize':12, 'fontname':'serif', \
                                                 'fontweight':'bold'})
plt.tight_layout()
# plt.savefig('inc_corr_heatmap_avg_subj_time.png', dpi=300)
plt.savefig('inc_corr_heatmap_avg_subj_time_blk3.png', dpi=300)
plt.show()

# plot inc corr matrix - sorted by community
plotting.plot_mod_heatmap(inc_mat_avg_time_sub, ci_inc_avg[0], vmin=-1, vmax=1, \
                          cmap='viridis')
plt.title('Incongruent', fontdict={'fontsize':12, 'fontname':'serif', \
                                                 'fontweight':'bold'})
plt.tight_layout()
# plt.savefig('inc_corr_heatmap_avg_subj_time_ci_sort.png', dpi=300)
plt.savefig('inc_corr_heatmap_avg_subj_time_ci_sort_blk3.png', dpi=300)
plt.show()

# plot inc corr matrix - consensus map
consensus = cluster.find_consensus(np.column_stack(ci_inc), seed=0)
plotting.plot_mod_heatmap(inc_mat_avg_time_sub, consensus, cmap='viridis')
plt.title('Incongruent', fontdict={'fontsize':12, 'fontname':'serif', \
                                                 'fontweight':'bold'})
plt.tight_layout()
# plt.savefig('inc_corr_heatmap_avg_subj_time_con_map.png', dpi=300)
plt.savefig('inc_corr_heatmap_avg_subj_time_con_map_blk3.png', dpi=300)
plt.show()

# incongruent - graphical form
G = nx.from_numpy_matrix(inc_mat_avg_time_sub)
G.remove_edges_from([(n1, n2) for n1, n2, w in G.edges(data="weight") if -0.2 <= w <= 0.2])
pos = nx.spring_layout(G, seed=0, k=1)
nx.draw(G, node_color='tab:orange', node_size=350, edge_color='#767676', \
        linewidths=1, font_size=15, pos=pos, alpha=0.75)
plt.tight_layout()
plt.show()


######## matrix visualization - congruent
con_block_frames = np.array(con_block)/2
# con_frames_idx = [ii for i in list(map(lambda x: list(range(int(x[0]), \
#                                 int(x[1]))), con_block_frames[2])) for ii in i]
con_frames_idx = list(range(int(con_block_frames[2][0]), int(con_block_frames[2][1])))

# get efc matrices for inc timepoints
con_mat_alltime_allsub = list(map(lambda subj: itemgetter(*con_frames_idx)(subj), \
                                  mat_stroop_all ))
con_mat_avgtime_allsub = list(map(lambda subj:np.average(subj, axis=0), con_mat_alltime_allsub))

con_mat_avg_time_sub = np.average(con_mat_avgtime_allsub, axis=0)

# ci community map for averaged corr time subj matrix
ci_con_avg = bct.community_louvain(con_mat_avg_time_sub, gamma=1.5, B='negative_sym')
# ci_con_avg = [math.floor(i) for i in np.average(itemgetter(*con_frames_idx)(ci_avg_sub), axis=0)]
ci_con = itemgetter(*con_frames_idx)(ci_avg_sub)


# plot inc corr matrix
fig, ax = plt.subplots(1, 1)
coll = ax.imshow(con_mat_avg_time_sub, vmin=-1, vmax=1, cmap='viridis') # cmap=plt.cm.RdYlBu_r
fig.colorbar(coll)
plt.title('Congruent', fontdict={'fontsize':12, 'fontname':'serif', \
                                                 'fontweight':'bold'})
plt.tight_layout()
plt.savefig('con_corr_heatmap_avg_subj_time.png', dpi=300)
plt.savefig('con_corr_heatmap_avg_subj_time_blk3.png', dpi=300)
plt.show()

# plot inc corr matrix - sorted by community
plotting.plot_mod_heatmap(con_mat_avg_time_sub, ci_con_avg[0], vmin=-1, vmax=1, \
                          cmap='viridis')
plt.title('Congruent', fontdict={'fontsize':12, 'fontname':'serif', \
                                                 'fontweight':'bold'})
plt.tight_layout()
# plt.savefig('con_corr_heatmap_avg_subj_time_ci_sort.png', dpi=300)
plt.savefig('con_corr_heatmap_avg_subj_time_ci_sort_blk3.png', dpi=300)
plt.show()


# plot con corr matrix - consensus map
consensus = cluster.find_consensus(np.column_stack(ci_inc), seed=0)
plotting.plot_mod_heatmap(inc_mat_avg_time_sub, consensus, cmap='viridis')
plt.title('Congruent', fontdict={'fontsize':12, 'fontname':'serif', \
                                                 'fontweight':'bold'})
plt.tight_layout()
# plt.savefig('con_corr_heatmap_avg_subj_time_con_map.png', dpi=300)
plt.savefig('con_corr_heatmap_avg_subj_time_con_map_blk3.png', dpi=300)
plt.show()

# congruent - graphical form
G = nx.from_numpy_matrix(con_mat_avg_time_sub)
G.remove_edges_from([(n1, n2) for n1, n2, w in G.edges(data="weight") if -0.2 <= w <= 0.2])
pos = nx.spring_layout(G, seed=0, k=1)
nx.draw(G, node_color='tab:blue', node_size=350, edge_color='#767676', \
            linewidths=1, font_size=15, pos=pos, alpha=0.75)
plt.tight_layout()
plt.show()


sys.exit()



########## - resting state
# task = 'rest'
# efc_mat_allsub = jr.get_efc_trans(opj(main_dir, d_path), task, \
#                                   subj_lst[0:math.floor(len(subj_lst)*0.75)])
# np.save(f'subjs_efc_mat_{task}.npy', efc_mat_allsub)


# load saved npy with task coalesce matrices
mat_rest_all = np.load('subjs_efc_mat_rest.npy')

# do mod max on all subjs
# mod_max_rest_all = np.array(list(map(lambda subj: np.array(list(map(lambda mat: \
#                         bct.community_louvain(mat, gamma=1.5, B='negative_sym'), \
#                             subj)), dtype=object), mat_rest_all)), dtype=object)
# np.save(f'subjs_all_mod_max_{task}.npy', mod_max_rest_all)

mod_max_rest_all = np.load('subjs_all_mod_max_rest.npy', allow_pickle=True)

# separate out communities and index for all subjs
q_allsub_rest = np.array(list(map(lambda subj: np.array(list(map(lambda x:x[1], subj))), \
                                                mod_max_rest_all)))
# avg q across subjs
q_avg_rest = np.average(q_allsub_rest, axis=0)
frames = q_avg.shape[0]

# smooth q using gaussian filter, mask first few vals
q_smooth_rest = gaussian_filter(q_avg_rest, sigma=2.5)

##############
# plot mod max, all timepoint, all subjs
plt.plot(np.arange(0, frames*rt, 2), q_avg_rest, linewidth=1)
plt.xticks(np.arange(0, frames*rt, 60))
plt.xlabel("Time (s)", size=11, fontname="serif")
plt.ylabel("Modularity (Q)", size=11, fontname="serif")
plt.savefig('allsub_mod_qall_rest.png', dpi=300)
plt.show()

# plot mod max, all timepoint, all subjs - smooth
plt.plot(np.arange(0, frames*rt, 2), q_smooth_rest, linewidth=1)
plt.xticks(np.arange(0, frames*rt, 60))
plt.xlabel("Time (s)", size=11, fontname="serif")
plt.ylabel("Modularity (Q)", size=11, fontname="serif")
plt.savefig('allsub_mod_qall_smooth_sig2p5_rest.png', dpi=300)
plt.show()
##############

