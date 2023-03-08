"""
Name: Kimberly Nestor
Date: 03/2023
Project: int_seg, bg_cb
Description: This program takes fMRI matrices from Shen atlas and determines
             modularity along the timescale.

Modularity info: https://tinyurl.com/4pazk7bz
Sample code: https://tinyurl.com/3jkf2zjj
"""

import sys
import os
from pathlib import Path

# set shared scripts dir for both msit and stroop up one level
curr_dir = os.getcwd()
pars = Path(curr_dir).parents
par_dir = pars[0]
sys.path.insert(0, str(par_dir))

from os.path import join as opj
from operator import itemgetter
from time import *
import itertools

from csv import writer
from csv import DictWriter
import csv

import bct

import jr_funcs as jr
from bg_cb_funcs import *
from task_blocks import *
from config import *

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
import statsmodels.api as sm
import statsmodels.formula.api as smf

from nilearn.image import load_img
from nilearn.glm.first_level import compute_regressor

import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt

task = 'msit'
figs_dir = '/home/kimberlynestor/gitrepo/int_seg/output/'


# load modularity time series, remove init 5, smooth, z score
# save_mod_idx(subj_lst, task=task) # run this line once


# cortex
q_allsub = np.load(f'{main_dir}IntermediateData/{task}/subjs_all_net_cort_q_{task}.npy')
q_allsub_z = np.array(list(map(lambda i: stats.zscore(i[5:]), q_allsub)))
q_allsub_smooth = np.array(list(map(lambda i: gaussian_filter(i[5:], sigma=1), q_allsub))) # sigma=2
q_allsub_smooth_z = np.array(list(map(lambda i: stats.zscore(i), q_allsub_smooth)))

q_allsub_smooth_z_all = np.array(list(map(lambda i: stats.zscore(i), \
                            np.array(list(map(lambda i: gaussian_filter(i, sigma=1), \
                                              q_allsub))))))

q_allsub_smooth_allpts = np.array(list(map(lambda i: gaussian_filter(i, sigma=1), q_allsub)))
q_allsub_smooth_z_allpts = np.array(list(map(lambda i: stats.zscore(i), q_allsub_smooth_allpts)))

q_avg_allpts = np.average(q_allsub, axis=0)
q_avg_smooth_allpts = gaussian_filter(q_avg_allpts, sigma=1)
q_avg_smooth_z_allpts = stats.zscore(q_avg_smooth_allpts)
np.save(f'{main_dir}IntermediateData/{task}/q_avg_smooth_z_allpts_cort_{task}.npy', q_avg_smooth_z_allpts)

## average task blocks
q_allsub_smooth_z_allpts_pad = np.array(list(map(lambda sub:np.pad(sub, (0,10), \
                                mode='constant', constant_values=np.nan), q_allsub_smooth_z_allpts)))
mod_cort_avg_blocks_allsub = np.array(list(map(lambda sub: np.nanmean(np.array(list(\
                                map(lambda i : sub[int(i[0]):int(i[1])+1], group_blocks))), \
                                    axis=0), q_allsub_smooth_z_allpts_pad)))

# avg q across subjs
q_avg = np.average(q_allsub, axis=0)
q_avg_smooth = np.average(q_allsub_smooth, axis=0)
q_avg_smooth_z = stats.zscore(q_avg_smooth)
# np.save(f'q_avg_smooth_z_cort_stroop.npy', q_avg_smooth_z)


###### MODULARITY LINE GRAPH
# smooth q using gaussian filter, mask first few vals, sigma=2
# q_avg_smooth_mask = np.ma.array(np.insert(q_avg_smooth, 0, np.ones(5)), \
#                                 mask=np.pad(np.ones(5), (0,frs-5)))
q_avg_smooth_z_mask = np.ma.array(np.insert(stats.zscore(np.average(np.array(list( \
                        map(lambda i:gaussian_filter(i[5:], sigma=2), q_allsub))), axis=0)), \
                            0, np.ones(5)), mask=np.pad(np.ones(5), (0,frs-5)))

## SINGLE CORTEX LINE GRAPH - with STD spread
mod_ci_all = list(map(lambda i: st.norm.interval(alpha=0.95, loc=np.mean(i), scale=st.sem(i)), q_allsub_smooth_z_allpts.T))
mod_lb = stats.zscore(np.array(mod_ci_all).T[0])
mod_ub = stats.zscore(np.array(mod_ci_all).T[1])
mod_sterr_allpts = list(map(lambda i: st.sem(i), q_allsub_smooth_z_allpts.T))
mod_std_allpts = list(map(lambda i: np.std(i), q_allsub_smooth_z_allpts.T))

plt.axhline(y=0, c='k', lw=1.2, alpha=0.2, ls='--', dashes=(5, 5))
plt.plot(np.arange(0, frs*rt, 2), q_avg_smooth_z_mask, lw=1, c=p_dict['cort_line_cb']) # 0.2
plt.fill_between(np.arange(0, frs*rt, 2), q_avg_smooth_z_mask-mod_std_allpts, \
                 q_avg_smooth_z_mask+mod_std_allpts, lw=0, color=p_dict['cort_line'], alpha=0.6)

plt.xticks(np.arange(0, frs*rt, 60))
plt.xlabel('Time (s)', size=15, fontname='serif')
plt.ylabel('Modularity (Q/z)', size=15, fontname='serif')
plt.title(task.upper(), size=20, fontname='serif', fontweight='bold')

# colour plot background with breakpoints
for i in range(len(inc_block)):
    # incongruent
    plt.axvspan(inc_block[i][0], inc_block[i][1], facecolor=p_dict['Incongruent_cb'], alpha=0.15) # tab:orange, 0.22 - cc, .35
    # congruent
    plt.axvspan(con_block[i][0], con_block[i][1], facecolor=p_dict['Congruent_cb'], alpha=0.16) # tab:blue, 0.2, #91C1E2
# plt.ylim(-3.5, 2.25)
plt.ylim(-4.5, 3.25)
plt.legend(handles=[inc_patch_cb, con_patch_cb], loc=4)
plt.tight_layout()
plt.savefig(f'{pars[1]}/output/{task}/allsub_cortnet_mod_qall_smooth_sig2_blocks_mask_init_cb_yerr_std_{task}.png', dpi=2000)
plt.show()


##### SEPARATE INC AND CON BLOCKS - ON TASK
# for all subjects get only inc and con mod
q_allsub_inc = np.array(list(map(lambda subj: np.array(list(map(lambda block: \
                                    subj[block] , inc_frames_idx))) ,q_allsub_z)))
q_allsub_con = np.array(list(map(lambda subj: np.array(list(map(lambda block: \
                                    subj[block] , con_frames_idx))) ,q_allsub_z)))
q_allsub_fix = np.array(list(map(lambda subj: np.array(list(map(lambda block: \
                                    subj[block] , fix_frames_idx))) ,q_allsub_z)))


# make dataframe for plotting - average subjects, all block
q_inc_avg = np.mean(q_allsub_inc, axis=0)
q_inc_sep_blocks = np.array([np.array(list(map(lambda x: [x, i[0]], i[1])))  \
                    for i in np.array(list(enumerate(q_inc_avg, 1)) , dtype=object)])
q_inc_sep_blocks = list(map(lambda xxx:[xxx[0], int(xxx[1]), 'Incongruent'], \
                    np.array([np.array(list(map(lambda x:[x[0], x[1]+j], i))) \
                    for i,j in zip(q_inc_sep_blocks, range(4))]).reshape((120, 2)) ))

q_con_avg = np.mean(q_allsub_con, axis=0)
q_con_sep_blocks = list(map(lambda xxx:[xxx[0], int(xxx[1]), 'Congruent'], \
                        np.array([np.array(list(map(lambda x: [x, i[0]], i[1])))  \
                            for i in np.array(list(enumerate(q_con_avg, 1))  , \
                                dtype=object )]).reshape((120, 2))))
q_con_sep_blocks = list(map(lambda x: [x[0], x[1]+x[1], x[2]], q_con_sep_blocks))


df_task = pd.DataFrame(list(itertools.chain(q_inc_sep_blocks, q_con_sep_blocks)), \
                               columns=['mod_idx', 'block', 'task'])


###### POINTPLOT OF MEAN AND CI - on task only
# plt.axhline(y=0, c='k', lw=1.2, alpha=0.2, ls='--', dashes=(5, 5))
plt.axhline(y=0, c='k', lw=1.2, alpha=0.28, ls='--', dashes=(13, 25))

sns.pointplot(x='block', y='mod_idx', hue='task', data=df_task, \
                    join=False, palette=p_dict_cb, errwidth=5, scale=2, alpha=0.9) # capsize=0.5
plt.xlabel('Task Block', size=15, fontname='serif')
plt.ylabel('Modularity (Q/z)', size=15, fontname='serif')
plt.title(task.upper(), size=20, fontname='serif', fontweight='bold')
plt.legend((inc_circ_cb, con_circ_cb), ('Incongruent', 'Congruent'), numpoints=1, loc=1)
plt.tight_layout()
plt.savefig(f'{pars[1]}/output/{task}/allsub_cortnet_mod_qavg_smooth_sig1_pointplot_ci_ontask_cb_{task}.png', dpi=2000)
plt.show()


### prepare melt df with subj_id, task, block and mean mod_idx by block
q_allsub_con_tmean = np.array(list(map(lambda sub: np.mean(sub, axis=1), q_allsub_con)))
df_mlm_con = pd.DataFrame(q_allsub_con_tmean, columns=range(1,5))
df_mlm_con.insert(0, 'subj_ID', [int(i) for i in subj_lst])
df_mlm_con['task'] = 'Congruent'#1
df_mlm_con_melt = pd.melt(df_mlm_con, id_vars=['subj_ID', 'task'], var_name='block', \
                          value_name='mu_mod_idx')

q_allsub_inc_tmean = np.array(list(map(lambda sub: np.mean(sub, axis=1), q_allsub_inc)))
df_mlm_inc = pd.DataFrame(q_allsub_inc_tmean, columns=range(1,5))
df_mlm_inc.insert(0, 'subj_ID', [int(i) for i in subj_lst])
df_mlm_inc['task'] = 'Incongruent'#0
df_mlm_inc_melt = pd.melt(df_mlm_inc, id_vars=['subj_ID', 'task'], var_name='block', \
                          value_name='mu_mod_idx')

# concat con and inc melt df
df_mlm = pd.concat([df_mlm_con_melt, df_mlm_inc_melt], ignore_index=True)
df_mlm.to_csv(f'{main_dir}IntermediateData/{task}/on_task_block_mu_mod_idx_{task}.csv', index=False)


sys.exit()


#### HRF PREDICT
# timepoints of image capture
frame_times = np.arange(frs)*rt

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


## WITHIN SUBJECT TEST - MLR
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
np.save('mlr_coeffs_subjs_all_net_cort_mod.npy', mlr_coeffs_lst)
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
m_output = open(f'subjs_all_net_cort/mlr_mod/mod_mlr_output_cort.csv', 'w')
m_output = open(f'subjs_all_net_cort/mlr_mod/mod_mlr_output_cort.csv', 'a')
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
# plt.savefig('subjs_all_net_cort/mlr/allsub_cortnet_mod_swarm_diff.png', dpi=300)
# plt.show()
plt.close()


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
# plt.savefig('subjs_all_net_cort/mlr/allsub_cortnet_stripplot_diff.png', dpi=300)
# plt.show()
plt.close()


# print(df_coeffs)
# C - I