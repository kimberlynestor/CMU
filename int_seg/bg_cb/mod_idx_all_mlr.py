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
from time import *
import itertools

from csv import writer
from csv import DictWriter
import csv

import bct
import jr_funcs as jr

import numpy as np
import pandas as pd

import math
from scipy import stats
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression
from mlinsights.mlmodel import IntervalRegressor

from statsmodels.tsa.ar_model import AutoReg
from scipy.stats import zscore
from scipy import stats
import scipy.stats as st

from nilearn.image import load_img
from nilearn.glm.first_level import compute_regressor

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import ptitprince as pt

# np.set_printoptions(threshold=sys.maxsize)
# pd.set_option('display.max_rows', None)


# data path and subjects
main_dir = '/home/kimberlynestor/gitrepo/int_seg/data/'
d_path = 'pip_edge_ts/shen/'
dep_path = 'depend/'

rt = 2
frs = 280

# get subject list and task conditions
subj_lst = np.loadtxt(opj(main_dir, dep_path, 'subjects_intersect_motion_035.txt'))
df_task_events = pd.read_csv(opj(main_dir, dep_path, 'task-stroop_events.tsv'), sep="\t")

# shen parcels region assignment
net_assign = pd.read_csv(opj(main_dir, dep_path, 'shen_268_parcellation_networklabels.csv')).values
cort_assign = np.array(list(filter(lambda i: i[1] != 4, net_assign)))
c_idx = [i[0] for i in cort_assign]

task = 'stroop'
out_name = 'subjs_all_net_cort_q_'
# out_name = 'subjs_all_net_all_q_'
"""
# get efc mat for all subjects, cortical networks only

for task in ['stroop', 'msit', 'rest']:
    #     print(task)
    # mod_file = open(f'subjs_mod_mat_{task}.csv', 'w')
    q_file = open(f'{out_name}{task}.csv', 'w')

    # loop_start = process_time()
    q_subj_lst = []
    for subj in subj_lst:
        efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj) # all networks
        efc_mat_cort = np.array(list(map(lambda frame: \
                        np.array(list(map(lambda col: col[c_idx], \
                            frame[c_idx]))), efc_mat))) # cortex only

        mod_all_frs = np.array(list(map(lambda mat: bct.community_louvain(mat, \
                            gamma=1.5, B='negative_sym'), efc_mat_cort)), dtype=object)

        ci_all_frs = np.array(list(map(lambda x: x[0], mod_all_frs)))
        q_all_frs = np.array(list(map(lambda x: x[1], mod_all_frs)))
        q_subj_lst.append(q_all_frs)

        # save to csv
        with open(f'{out_name}{task}.csv', 'a') as f_obj:
            w_obj = writer(f_obj)
            w_obj.writerow(q_all_frs)
            f_obj.close()
        # break
    # save to npy
    np.save(f'{out_name}{task}.npy', q_subj_lst)
# loop_end = process_time()
# loop_time = loop_end - loop_start
# print(f'loop time for execution: {loop_time}')
"""


# load modularity time series, remove init 5, smooth, z score
q_allsub = np.load(f'{out_name}{task}.npy')
q_allsub_smooth = np.array(list(map(lambda i: gaussian_filter(i[5:], sigma=1), q_allsub)))
q_allsub_smooth_z = np.array(list(map(lambda i: stats.zscore(i), q_allsub_smooth)))

# avg q across subjs
q_avg = np.average(q_allsub, axis=0)
q_avg_smooth = np.average(q_allsub_smooth, axis=0)
q_avg_smooth_z = stats.zscore(q_avg_smooth)

# smooth q using gaussian filter, mask first few vals
# q_smooth = gaussian_filter(q_avg, sigma=1)
# q_smooth_mask_init = np.ma.array(q_smooth, mask=np.pad(np.ones(5), (0,frs-5)))
# init_mask = np.ma.array(np.ones(5), mask=np.ones(5))



## TASK BLOCK ONSET
df_task_events = pd.read_csv(opj(main_dir, dep_path, 'task-stroop_events.tsv'), sep="\t")

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

# make init mask
q_avg_smooth_mask = np.ma.array(np.insert(q_avg_smooth, 0, np.ones(5)), \
                                mask=np.pad(np.ones(5), (0,frs-5)))
q_avg_smooth_z_mask = np.ma.array(np.insert(q_avg_smooth_z, 0, np.ones(5)), \
                                  mask=np.pad(np.ones(5), (0,frs-5)))

## plot mod max, all timepoint, all subjs - smooth, with task blocks
plt.plot(np.arange(0, 280*rt, 2), q_avg_smooth_z_mask, linewidth=1)
plt.xticks(np.arange(0, 280*rt, 60))
plt.xlabel("Time (s)", size=15, fontname="serif")
plt.ylabel("Modularity (Q)", size=15, fontname="serif")

# colour plot background with breakpoints
for i in range(len(inc_block)):
    # incongruent
    plt.axvspan(inc_block[i][0], inc_block[i][1], facecolor='tab:orange', alpha=0.22)
    # congruent
    plt.axvspan(con_block[i][0], con_block[i][1], facecolor='tab:blue', alpha=0.2)
# plt.ylim(0.43, 0.45) # mask mod, no z
plt.ylim(-3.5, 2.25)
plt.legend(handles=[inc_patch, con_patch], loc=4)
plt.tight_layout()
plt.savefig('subjs_all_net_cort/mod/allsub_cortnet_mod_qall_smooth_sig1_blocks_mask_init.png', dpi=300)
plt.show()
##############


## HRF PREDICT
# timepoints of image capture
frame_times = np.arange(frs)*2.0

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


## WITHIN SUBJECT TEST
# fit multiple linear regression model to q for all subs
mlr_coeffs_lst = []
for i in q_allsub_smooth_z:
    # lin reg model
    mlr = LinearRegression()
    mlr.fit(df_mod_reg[['inc_reg', 'con_reg']].iloc[5:,:], i)

    # slope
    mlr_slope = mlr.intercept_
    # coeffs, best fit line
    mlr_coeffs = mlr.coef_
    mlr_coeffs_lst.append(mlr_coeffs)
# print(mlr_coeffs_lst)

"""
# get coeff confidence interval
lin = IntervalRegressor(mlr).fit(df_mod_reg['inc_reg'].values.reshape(-1, 1), q_avg)
sorted_x = np.array(list(sorted(df_mod_reg['inc_reg'].values)))
pred = lin.predict(sorted_x.reshape(-1, 1))
boot_pred = lin.predict_sorted(sorted_x.reshape(-1, 1))
min_pred = boot_pred[:, 0]
max_pred = boot_pred[:, boot_pred.shape[-1]-1]
print(f'inc_interval: {np.mean(min_pred)}, {np.mean(max_pred)}')

import statsmodels.api as sm 
lr_model = sm.OLS(q_avg, df_mod_reg[['inc_reg', 'con_reg']]).fit()
print(lr_model.summary())
sys.exit()
"""

# make diff vec
np.save('mlr_coeffs_subjs_all_net_cort.npy', mlr_coeffs_lst)
df_coeffs = pd.DataFrame(mlr_coeffs_lst, columns=['inc_coeff', 'con_coeff'])
diff_vec = np.squeeze(list(map(lambda x:[x[1] - x[0]], mlr_coeffs_lst)))
df_coeffs.insert(2, 'diff_vec', diff_vec)

## inc and con beta coeff - single
inc_reg = np.mean(df_coeffs['inc_coeff'].values)
con_reg = np.mean(df_coeffs['con_coeff'].values)


"""
mlr.fit(df_mod_reg[['inc_reg', 'con_reg']], q_avg) 
mlr_coeffs = mlr.coef_
print(mlr_coeffs)
"""

# get confidence intervals
# inc_interval = st.t.interval(alpha=0.95, df=len(df_coeffs['inc_coeff'].values)-1, \
#                              loc=inc_reg, scale=st.sem(df_coeffs['inc_coeff'].values))
inc_interval = st.norm.interval(alpha=0.95, loc=inc_reg, scale=st.sem(df_coeffs['inc_coeff'].values))

# con_interval = st.t.interval(alpha=0.95, df=len(df_coeffs['con_coeff'].values)-1, \
#                              loc=con_reg, scale=st.sem(df_coeffs['con_coeff'].values))
con_interval = st.norm.interval(alpha=0.95, loc=con_reg, scale=st.sem(df_coeffs['con_coeff'].values))

"""
# load rest modularity
q_allsub_rest = np.load('subjs_all_net_cort_q_rest.npy', allow_pickle=True)

# z-score rest and get mat mean
q_allsub_znorm_rest = np.array(list(map(lambda x: stats.zscore(x), q_allsub_rest)))
rest_mean = np.average(q_allsub_znorm_rest)
"""

# one samp test - within subj, rest as mean
ttest_1samp_within = stats.ttest_1samp(diff_vec, np.average(diff_vec))

print(f'\nDifference vector mean: {np.average(diff_vec)}')
print(f'One samp t-test: {ttest_1samp_within} \n')
print("Model coefficients:")
# print(f'inc_coeff  {round(inc_reg, 4)}    {inc_interval}')
# print(f'con_coeff  {round(con_reg, 4)}    {con_interval}\n')
print(f'inc_coeff  {inc_reg}    {inc_interval}')
print(f'con_coeff  {con_reg}    {con_interval}\n')

# save model output to csv
m_output = open(f'subjs_all_net_cort/mlr/mod_mlr_output.csv', 'w')
m_output = open(f'subjs_all_net_cort/mlr/mod_mlr_output.csv', 'a')
writer = csv.writer(m_output)

writer.writerow(['Difference vector mean ', np.average(diff_vec)])
writer.writerow(['One samp t-test ', ttest_1samp_within])
writer.writerow(['inc_coeff ', inc_reg, inc_interval])
writer.writerow(['con_coeff ', con_reg, con_interval])


## SWARM PLOT - of beta coeffs
# single
# sns.swarmplot(x=diff_vec, zorder=1)
# sns.pointplot(x=diff_vec, ci=95, join=False, color='black', seed=0)
# plt.show()

# two groups
p_dict = {'Incongruent':'tab:orange', 'Congruent':'tab:blue', 'Difference':'#8A2D1C'}
df_coeffs_melt = pd.DataFrame(list(itertools.chain( \
                    list(map(lambda i: [i[0], 'Incongruent'], mlr_coeffs_lst)), \
                    list(map(lambda i: [i[1], 'Congruent'], mlr_coeffs_lst)))  ), \
                              columns=['coeff', 'task'])
df_coeffs_melt_diff = pd.concat([pd.DataFrame(list(map(lambda i: [i, 'Difference'], \
                            diff_vec)), columns=['coeff', 'task']), df_coeffs_melt])

# plt.figure(figsize=(22,13), dpi=300) # 8, 12
plt.figure(figsize=(15,10.5), dpi=300) # 8, 12  # no smooth
sns.set(font='serif', font_scale=1.8)
sns.swarmplot(x='task', y='coeff', data=df_coeffs_melt_diff, \
              order=['Incongruent', 'Congruent', 'Difference'], zorder=1, \
              palette=p_dict, alpha=0.8, size=6)
sns.pointplot(x='task', y='coeff', data=df_coeffs_melt_diff, \
              order=['Incongruent', 'Congruent', 'Difference'], ci=95, \
              join=False, color='black', scale=1.25, errwidth=3, seed=0)

plt.xlabel('Task block', size=20, fontname="serif")
plt.ylabel('Beta coefficients (Q)', size=20, fontname="serif")
# plt.subplots_adjust(wspace=.2)
plt.tight_layout()
plt.savefig('subjs_all_net_cort/mlr/allsub_cortnet_mod_swarm_diff.png', dpi=300)
plt.show()


## STRIPLOT
sns.set(font='serif', font_scale=.95)
sns.stripplot(x='task', y='coeff', data=df_coeffs_melt_diff, \
              order=['Incongruent', 'Congruent', 'Difference'], zorder=1, \
              palette=p_dict, alpha=0.8, size=3, jitter=0.2) #jitter=0.25 # no smooth = no jitter
sns.pointplot(x='task', y='coeff', data=df_coeffs_melt_diff, \
              order=['Incongruent', 'Congruent', 'Difference'], ci=95, \
              join=False, color='black', scale=0.7, seed=0)
plt.xlabel('Task block', size=10.5, fontname="serif")
plt.ylabel('Beta coefficients (Q)', size=10.5, fontname="serif")
plt.tight_layout()
plt.savefig('subjs_all_net_cort/mlr/allsub_cortnet_stripplot_diff.png', dpi=300)
plt.show()


print(df_coeffs)
# C - I