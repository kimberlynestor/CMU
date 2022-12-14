"""
Name: Kimberly Nestor
Date: 03/2022
Project: int_seg, bg_cb
Description: This program

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


# load edge time series by region
# save_con(subj_lst, task='all', region='cort')
# save_con(subj_lst, task='all', region='cb')
# save_con(subj_lst, task='all', region='bg')
# save_con(subj_lst, task='stroop', region='thal')

"""
# average across nodes, then subjects, smooth, z
# cortical nodes
con_allsub_cort = np.load(f'subjs_all_net_cort_stroop.npy')
con_allsub_cort = np.absolute( con_allsub_cort )
avg_con_cort = np.array(list(map(lambda sub: np.array(list(map(lambda t: \
                                np.average(t) , sub))), con_allsub_cort)))
avg_all_con_cort = np.average(avg_con_cort, axis=0)
con_cort_smooth = gaussian_filter(avg_all_con_cort, sigma=1)
con_cort_smooth_z = stats.zscore(con_cort_smooth)


# cb nodes
con_allsub_cb = np.load(f'subjs_all_net_cb_stroop.npy')
con_allsub_cb = np.absolute( con_allsub_cb )
avg_con_cb = np.array(list(map(lambda sub: np.array(list(map(lambda t: \
                                np.average(t) , sub))), con_allsub_cb)))
avg_all_con_cb = np.average(avg_con_cb, axis=0)
con_cb_smooth = gaussian_filter(avg_all_con_cb, sigma=1)
con_cb_smooth_z = stats.zscore(con_cb_smooth)

# bg nodes
con_allsub_bg = np.load(f'subjs_all_net_bg_stroop.npy')
con_allsub_bg = np.absolute( con_allsub_bg )
avg_con_bg = np.array(list(map(lambda sub: np.array(list(map(lambda t: \
                                np.average(t) , sub))), con_allsub_bg)))
avg_all_con_bg = np.average(avg_con_bg, axis=0)
con_bg_smooth = gaussian_filter(avg_all_con_bg, sigma=1)
con_bg_smooth_z = stats.zscore(con_bg_smooth)

# thal nodes
con_allsub_thal = np.load(f'subjs_all_net_thal_stroop.npy')
con_allsub_thal = np.absolute( con_allsub_thal )
avg_con_thal = np.array(list(map(lambda sub: np.array(list(map(lambda t: \
                                np.average(t) , sub))), con_allsub_thal)))
avg_all_con_thal = np.average(avg_con_thal, axis=0)
con_thal_smooth = gaussian_filter(avg_all_con_thal, sigma=1)
con_thal_smooth_z = stats.zscore(con_thal_smooth)
"""

# load modularity time series, remove init 5, smooth, z score
# save_mod_idx(subj_lst, task='all', region='cort')
# save_mod_idx(subj_lst, task='stroop', region='cb')
# save_mod_idx(subj_lst, task='stroop', region='bg')
# save_mod_idx(subj_lst, task='stroop', region='thal')

# save_mod_idx_adj_cort(subj_lst, task='stroop', region='cb')
# save_mod_idx_adj_cort(subj_lst, task='stroop', region='bg')
# save_mod_idx_adj_cort(subj_lst, task='stroop', region='thal')

# cortex
q_allsub = np.load(f'subjs_all_net_cort_q_stroop.npy')
q_allsub_z = np.array(list(map(lambda i: stats.zscore(i[5:]), q_allsub)))
q_allsub_smooth = np.array(list(map(lambda i: gaussian_filter(i[5:], sigma=2), q_allsub))) # sigma=2
q_allsub_smooth_z = np.array(list(map(lambda i: stats.zscore(i), q_allsub_smooth)))

q_allsub_smooth_z_all = np.array(list(map(lambda i: stats.zscore(i), \
                            np.array(list(map(lambda i: gaussian_filter(i, sigma=1), \
                                              q_allsub))))))

q_allsub_smooth_allpts = np.array(list(map(lambda i: gaussian_filter(i, sigma=1), q_allsub)))
q_allsub_smooth_z_allpts = np.array(list(map(lambda i: stats.zscore(i), q_allsub_smooth_allpts)))

q_avg_allpts = np.average(q_allsub, axis=0)
q_avg_smooth_allpts = gaussian_filter(q_avg_allpts, sigma=1)
q_avg_smooth_z_allpts = stats.zscore(q_avg_smooth_allpts)
np.save(f'q_avg_smooth_z_allpts_cort_stroop.npy', q_avg_smooth_z_allpts)
# sys.exit()

# modularity bg, cb, thal
# q_cb_allsub = np.load(f'subjs_all_net_cb_q_stroop.npy')
q_cb_allsub = np.load(f'subjs_all_net_cb_adj_cort_q_stroop.npy')
q_cb_allsub_z = np.array(list(map(lambda i: stats.zscore(i[5:]), q_cb_allsub)))
q_cb_allsub_smooth = np.array(list(map(lambda i: gaussian_filter(i[5:], sigma=1), q_cb_allsub)))
q_cb_allsub_smooth_z = np.array(list(map(lambda i: stats.zscore(i), q_cb_allsub_smooth)))

# q_bg_allsub = np.load(f'subjs_all_net_bg_q_stroop.npy')
q_bg_allsub = np.load(f'subjs_all_net_bg_adj_cort_q_stroop.npy')
q_bg_allsub_z = np.array(list(map(lambda i: stats.zscore(i[5:]), q_bg_allsub)))
q_bg_allsub_smooth = np.array(list(map(lambda i: gaussian_filter(i[5:], sigma=1), q_bg_allsub)))
q_bg_allsub_smooth_z = np.array(list(map(lambda i: stats.zscore(i), q_bg_allsub_smooth)))

# q_thal_allsub = np.load(f'subjs_all_net_thal_q_stroop.npy')
q_thal_allsub = np.load(f'subjs_all_net_thal_adj_cort_q_stroop.npy')
q_thal_allsub_z = np.array(list(map(lambda i: stats.zscore(i[5:]), q_thal_allsub)))
q_thal_allsub_smooth = np.array(list(map(lambda i: gaussian_filter(i[5:], sigma=1), q_thal_allsub)))
q_thal_allsub_smooth_z = np.array(list(map(lambda i: stats.zscore(i), q_thal_allsub_smooth)))


# avg q across subjs
q_avg = np.average(q_allsub, axis=0)
q_avg_smooth = np.average(q_allsub_smooth, axis=0)
q_avg_smooth_z = stats.zscore(q_avg_smooth)

q_cb_avg = np.average(q_cb_allsub, axis=0)
q_cb_avg_smooth = np.average(q_cb_allsub_smooth, axis=0)
q_cb_avg_smooth_z = stats.zscore(q_cb_avg_smooth)

q_bg_avg = np.average(q_bg_allsub, axis=0)
q_bg_avg_smooth = np.average(q_bg_allsub_smooth, axis=0)
q_bg_avg_smooth_z = stats.zscore(q_bg_avg_smooth)

q_thal_avg = np.average(q_thal_allsub, axis=0)
q_thal_avg_smooth = np.average(q_thal_allsub_smooth, axis=0)
q_thal_avg_smooth_z = stats.zscore(q_thal_avg_smooth)

# smooth q using gaussian filter, mask first few vals
# q_smooth = gaussian_filter(q_avg, sigma=1)
# q_smooth_mask_init = np.ma.array(q_smooth, mask=np.pad(np.ones(5), (0,frs-5)))
# init_mask = np.ma.array(np.ones(5), mask=np.ones(5))


###### MODULARITY LINE GRAPH
# make init mask
q_avg_smooth_mask = np.ma.array(np.insert(q_avg_smooth, 0, np.ones(5)), \
                                mask=np.pad(np.ones(5), (0,frs-5)))
q_avg_smooth_z_mask = np.ma.array(np.insert(q_avg_smooth_z, 0, np.ones(5)), \
                                  mask=np.pad(np.ones(5), (0,frs-5)))
# np.save(f'q_avg_smooth_z_cort_stroop.npy', q_avg_smooth_z)

q_cb_avg_smooth_z_mask = np.ma.array(np.insert(q_cb_avg_smooth_z, 0, np.ones(5)), \
                                  mask=np.pad(np.ones(5), (0,frs-5)))
q_bg_avg_smooth_z_mask = np.ma.array(np.insert(q_bg_avg_smooth_z, 0, np.ones(5)), \
                                  mask=np.pad(np.ones(5), (0,frs-5)))
q_thal_avg_smooth_z_mask = np.ma.array(np.insert(q_thal_avg_smooth_z, 0, np.ones(5)), \
                                  mask=np.pad(np.ones(5), (0,frs-5)))


## SINGLE CORTEX LINE GRAPH
## plot mod max, all timepoint, all subjs - smooth, with task blocks
# plt.axhline(y=0, c='k', lw=1.2, alpha=0.28, ls='--', dashes=(15, 10))
plt.axhline(y=0, c='k', lw=1.2, alpha=0.2, ls='--', dashes=(5, 5))
plt.plot(np.arange(0, frs*rt, 2), q_avg_smooth_z_mask, lw=1, c=p_dict['cort_line']) # dimgrey, gray, #757575

plt.xticks(np.arange(0, frs*rt, 60))
plt.xlabel('Time (s)', size=15, fontname='serif')
plt.ylabel('Modularity (Q/z)', size=15, fontname='serif')

# colour plot background with breakpoints
for i in range(len(inc_block)):
    # incongruent
    plt.axvspan(inc_block[i][0], inc_block[i][1], facecolor=p_dict['Incongruent_cb'], alpha=0.15) # tab:orange, 0.22 - cc, .35
    # congruent
    plt.axvspan(con_block[i][0], con_block[i][1], facecolor=p_dict['Congruent_cb'], alpha=0.16) # tab:blue, 0.2, #91C1E2
# plt.ylim(0.43, 0.45) # mask mod, no z
plt.ylim(-3.5, 2.25)
plt.legend(handles=[inc_patch_cb, con_patch_cb], loc=4)
plt.tight_layout()
plt.savefig('subjs_all_net_cort/mod/allsub_cortnet_mod_qall_smooth_sig1_blocks_mask_init_cb.png', dpi=300)
plt.show()


## SINGLE CORTEX LINE GRAPH - with CI spread
mod_ci_all = list(map(lambda i: st.norm.interval(alpha=0.95, loc=np.mean(i), scale=st.sem(i)), q_allsub_smooth_z_allpts.T))
mod_lb = stats.zscore(np.array(mod_ci_all).T[0])
mod_ub = stats.zscore(np.array(mod_ci_all).T[1])
mod_sterr_allpts = list(map(lambda i: st.sem(i), q_allsub_smooth_z_allpts.T))
mod_std_allpts = list(map(lambda i: np.std(i), q_allsub_smooth_z_allpts.T))

plt.axhline(y=0, c='k', lw=1.2, alpha=0.2, ls='--', dashes=(5, 5))
plt.plot(np.arange(0, frs*rt, 2), q_avg_smooth_z_mask, lw=1, c=p_dict['cort_line_cb']) # 0.2
# plt.plot(np.arange(0, frs*rt, 2), mod_lb, lw=1, c='r')
# plt.plot(np.arange(0, frs*rt, 2), mod_ub, lw=1, c='b')
# plt.fill_between(np.arange(0, frs*rt, 2), q_avg_smooth_z_mask-mod_sterr_allpts, q_avg_smooth_z_mask+mod_sterr_allpts, lw=0, color=p_dict['cort_line'], alpha=0.7)
plt.fill_between(np.arange(0, frs*rt, 2), q_avg_smooth_z_mask-mod_std_allpts, q_avg_smooth_z_mask+mod_std_allpts, lw=0, color=p_dict['cort_line'], alpha=0.6)

plt.xticks(np.arange(0, frs*rt, 60))
plt.xlabel('Time (s)', size=15, fontname='serif')
plt.ylabel('Modularity (Q/z)', size=15, fontname='serif')

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
plt.savefig('subjs_all_net_cort/mod/allsub_cortnet_mod_qall_smooth_sig2_blocks_mask_init_cb_yerr_std.png', dpi=2000)
plt.show()

sys.exit()


#### MOD LINE GRAPH - ALL TOGETHER
# plot part coeff, all timepoint
plt.plot(np.arange(0, frs*rt, 2), q_avg_smooth_z_mask, lw=1, \
                            label='cort_line', c=p_dict['cort_line'])
plt.plot(np.arange(0, frs*rt, 2), q_cb_avg_smooth_z_mask, lw=1, label='cb_line', \
                            c=p_dict['cb_line'], ls='--')
plt.plot(np.arange(0, frs*rt, 2), q_bg_avg_smooth_z_mask, lw=1, label='bg_line', \
                            c=p_dict['bg_line'], ls='--') # flat

plt.xticks(np.arange(0, frs*rt, 60))
plt.xlabel('Time (s)', size=11, fontname='serif')
plt.ylabel('Modularity (Q/z)', size=15, fontname='serif')
# colour plot background with breakpoints
for i in range(len(inc_block)):
    # incongruent
    plt.axvspan(inc_block[i][0], inc_block[i][1], facecolor=p_dict['Incongruent'], alpha=0.35) # tab:orange, 0.22
    # congruent
    plt.axvspan(con_block[i][0], con_block[i][1], facecolor=p_dict['Congruent'], alpha=0.35) # tab:blue, 0.2, #91C1E2
plt.legend(handles=[inc_patch, con_patch, cort_line, bg_line, cb_line], prop={'size':6}, loc=4)
plt.tight_layout()
plt.savefig('subjs_all_net_cort/mod/allsub_cort_cb_bg_mod_smooth_z_cc.png', dpi=300)
plt.show()


# SUBPLOTS WITH BG, CB, THAL
## plot mod max, all timepoint, all subjs - smooth, with task blocks
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12,7), sharex=True)

ax1.plot(np.arange(0, frs*rt, 2), q_avg_smooth_z_mask, lw=1, c=p_dict['cort_line'])
ax1.plot(np.arange(0, frs*rt, 2), q_bg_avg_smooth_z_mask, lw=1.5, c=p_dict['bg_line'], ls=':', alpha=0.7)
ax1.get_xaxis().set_visible(False)

ax2.plot(np.arange(0, frs*rt, 2), q_avg_smooth_z_mask, lw=1, c=p_dict['cort_line'])
ax2.plot(np.arange(0, frs*rt, 2), q_cb_avg_smooth_z_mask, lw=1.5, c=p_dict['cb_line'], ls=':', alpha=0.7)
ax2.get_xaxis().set_visible(False)

ax3.plot(np.arange(0, frs*rt, 2), q_avg_smooth_z_mask, lw=1, c=p_dict['cort_line'])
ax3.plot(np.arange(0, frs*rt, 2), q_thal_avg_smooth_z_mask, lw=1.5, c=p_dict['thal_line'], ls=':', alpha=0.7)
ax3.get_xaxis().set_visible(False)

ax4.plot(np.arange(0, frs*rt, 2), q_bg_avg_smooth_z_mask, lw=1, c=p_dict['bg_line'], ls='-', alpha=0.7)
ax4.plot(np.arange(0, frs*rt, 2), q_cb_avg_smooth_z_mask, lw=1, c=p_dict['cb_line'], ls='-', alpha=0.7)
ax4.plot(np.arange(0, frs*rt, 2), q_thal_avg_smooth_z_mask, lw=1, c=p_dict['thal_line'], ls='-', alpha=0.7)

plt.xticks(np.arange(0, frs*rt, 60))
plt.xlabel('Time (s)', size=13, fontname='serif')
fig.text(0.003, 0.5, 'Modularity (Q/z)', size=11, fontname='serif', va='center', rotation='vertical')

# colour plot background with breakpoints
for i in range(len(inc_block)):
    # blocks incongruent and congruent
    ax1.axvspan(inc_block[i][0], inc_block[i][1], facecolor=p_dict['Incongruent'], alpha=0.35)
    ax1.axvspan(con_block[i][0], con_block[i][1], facecolor=p_dict['Congruent'], alpha=0.35)
    ax2.axvspan(inc_block[i][0], inc_block[i][1], facecolor=p_dict['Incongruent'], alpha=0.35)
    ax2.axvspan(con_block[i][0], con_block[i][1], facecolor=p_dict['Congruent'], alpha=0.35)
    ax3.axvspan(inc_block[i][0], inc_block[i][1], facecolor=p_dict['Incongruent'], alpha=0.35)
    ax3.axvspan(con_block[i][0], con_block[i][1], facecolor=p_dict['Congruent'], alpha=0.35)
    ax4.axvspan(inc_block[i][0], inc_block[i][1], facecolor=p_dict['Incongruent'], alpha=0.35)
    ax4.axvspan(con_block[i][0], con_block[i][1], facecolor=p_dict['Congruent'], alpha=0.35)
# plt.ylim(0.43, 0.45) # mask mod, no z
ax1.set_ylim(-3.5, 3.25)
ax2.set_ylim(-3.5, 3.25)
ax3.set_ylim(-3.5, 3.25)
ax4.set_ylim(-3.5, 3.25)
# plt.legend(handles=[inc_patch, con_patch], loc=4)
plt.legend(handles=[inc_patch, con_patch, cort_line, bg_line, cb_line, thal_line], loc=4, prop={'size':6})
plt.tight_layout()
plt.savefig('subjs_all_net_cort/mod/allsub_cort_cb_bg_thal_mod_smooth_sig1_blocks_mask_init_cc_subplots.png', dpi=300)
plt.show()
##############



##### SEPARATE INC AND CON BLOCKS
# for all subjects get only inc and con mod
q_allsub_inc = np.array(list(map(lambda subj: np.array(list(map(lambda block: \
                                    subj[block] , inc_frames_idx))) ,q_allsub_z))) # q_allsub, q_allsub_z, q_allsub_smooth_z
q_allsub_con = np.array(list(map(lambda subj: np.array(list(map(lambda block: \
                                    subj[block] , con_frames_idx))) ,q_allsub_z)))
q_allsub_fix = np.array(list(map(lambda subj: np.array(list(map(lambda block: \
                                    subj[block] , fix_frames_idx))) ,q_allsub_z)))

# make dataframe for plotting
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


q_fix_avg = np.mean(q_allsub_fix, axis=0)
q_fix_sep_blocks = list(map(lambda xxx:[xxx[0], int(xxx[1]), 'Fixation'], \
                        np.array([np.array(list(map(lambda x: [x, i[0]], i[1])))  \
                            for i in np.array(list(enumerate(q_fix_avg, 9))  , \
                                dtype=object )]).reshape((35, 2))))
q_fix_off_task_sep = list(map(lambda i: [i[0], i[1], 'Incongruent Fixation'] \
                            if i[1] in range(9, 16, 2) else [i[0], i[1], 'Congruent Fixation'], \
                              q_fix_sep_blocks))

df_q_sep_blocks = pd.DataFrame(list(itertools.chain(q_inc_sep_blocks, q_con_sep_blocks, q_fix_sep_blocks)), \
                               columns=['mod_idx', 'block', 'task'])
df_q_sep_blocks.to_csv('df_q_cort_sep_blocks_stroop.csv', index=False)

df_task = df_q_sep_blocks[(df_q_sep_blocks['task'] == 'Incongruent') | \
                          (df_q_sep_blocks['task'] == 'Congruent')]
df_fix = pd.DataFrame(q_fix_off_task_sep, columns=['mod_idx', 'block', 'task'])
# print( [f'F{i}' for i in range(1, 8)] )
# print(df_q_sep_blocks)


###### POINTPLOT OF CI - all in one
# fig, ax = plt.subplots()
sns.pointplot(x='block', y='mod_idx', hue='task', data=df_q_sep_blocks, \
              join=False, palette=p_dict, order=list(sum(zip(range(1,9), \
                                                range(9,16)), ()))+[8] )
plt.xticks(np.arange(0,15), labels=list(sum(zip(range(1,9), ['']*7), ()))+[8])
# plt.xticks(np.arange(0,14), labels=range(1,15))
# ax.set_xticklabels(range(1,15))
plt.xlabel("Task block", size=14, fontname="serif")
plt.ylabel('Modularity (Q/z)', size=14, fontname="serif")
plt.legend((inc_circ, con_circ, fix_circ), ('Incongruent', 'Congruent', 'Fixation'), numpoints=1)
plt.tight_layout()
plt.savefig('subjs_all_net_cort/mod/allsub_cortnet_mod_qavg_smooth_sig1_pointplot_ci_wr_cc.png', dpi=300)
plt.show()


###### POINTPLOT OF CI - two subplots - on and off task
# fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios':[2,3]})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
sns.pointplot(ax=ax1, x='block', y='mod_idx', hue='task', data=df_task, \
                            join=False, palette=p_dict) # capsize=0.5
sns.pointplot(ax=ax2, x='block', y='mod_idx', hue='task', data=df_fix[df_fix['block'] != 15], \
                            join=False, palette=p_dict)
plt.xticks(np.arange(0,6), labels=[f'F{i}' for i in range(1, 7)]) # 7, 8
ax1.set_xlabel('On Task Block', size=12, fontname='serif')
ax2.set_xlabel('Off Task Block', size=12, fontname='serif')
ax1.set_ylabel('Modularity (Q/z)', size=12, fontname='serif')
ax2.set_ylabel('')
ax1.legend((inc_circ, con_circ), ('Incongruent', 'Congruent'), numpoints=1, loc=1)
ax2.legend('', frameon=False)
fig.legend('', frameon=False)
plt.tight_layout()
plt.savefig('subjs_all_net_cort/mod/allsub_cortnet_mod_qavg_smooth_sig1_pointplot_ci_subplots_cc.png', dpi=300)
plt.show()

###### POINTPLOT OF CI - on task only
# fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios':[2,3]})
sns.pointplot(x='block', y='mod_idx', hue='task', data=df_task, \
                    join=False, palette=p_dict, errwidth=5, scale=2) # capsize=0.5
plt.xlabel('On Task Block', size=15, fontname='serif')
plt.ylabel('Modularity (Q/z)', size=15, fontname='serif')
plt.legend((inc_circ, con_circ), ('Incongruent', 'Congruent'), numpoints=1, loc=1)
plt.tight_layout()
plt.savefig('subjs_all_net_cort/mod/allsub_cortnet_mod_qavg_smooth_sig1_pointplot_ci_ontask_cc.png', dpi=300)
plt.show()

##### ON TASK
# prepare melt df with subj_id, task, block and mean mod_idx by block
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
# print(df_mlm)
df_mlm.to_csv('on_task_block_mu_mod_idx.csv', index=False)


# FULL TS
# get data for full timescale
df_mlm_ts = pd.DataFrame(q_allsub_smooth_z_all)
df_mlm_ts.columns +=1 # start column count from 1
df_mlm_ts.insert(0, 'subj_ID', [int(i) for i in subj_lst])
df_mlm_ts_melt = pd.melt(df_mlm_ts, id_vars=['subj_ID'], \
                                 var_name='frame', value_name='mod_idx')
df_mlm_ts_melt.insert(2, 'inc_reg', np.tile(inc_regressor, len(subj_lst)))
df_mlm_ts_melt.insert(3, 'con_reg', np.tile(con_regressor, len(subj_lst)))
df_mlm_ts_melt.to_csv('cort_mod_idx_reg_ts.csv', index=False)

sys.exit()

###### MIXED EFFECTS - con=1, incon=0
# run mixed effects model - on task, inc and con
# vcf = {'task': '0 + C(task)', 'block': '0 + C(block)'} #, vc_formula=vcf)
# md = smf.mixedlm('mu_mod_idx ~ task + block + task:block', df_mlm, \
#                  groups=df_mlm['subj_ID'], re_formula='~task') # task:block
# md = smf.mixedlm('mu_mod_idx ~ C(task) + block + C(task):block', df_mlm, \
#                  groups=df_mlm['subj_ID'], re_formula='~C(task)')
# mdf = md.fit() # method=['powell', 'lbfgs', 'bfgs']
# print(mdf.summary())


##### OFF TASK
"""
q_fix_off_task_sep_blocks_allsub = np.array(list(map(lambda sub: np.array([\
    np.array(list(map(lambda x: [x, i[0]], i[1]))) for i in np.array(\
        list(enumerate(sub, 1)), dtype=object)]).reshape((35, 2)), q_allsub_fix)))
q_fix_off_task_sep_allsub = list(map(lambda sub: list(map(lambda i: [i[0], int(i[1]), \
        'Incongruent Fixation'] if i[1] in range(1, 8, 2) else [i[0], int(i[1]), \
                'Congruent Fixation'], sub)), q_fix_off_task_sep_blocks_allsub))
"""

# MEAN TASK BLOCK
q_allsub_fix_tmean = np.array(list(map(lambda sub: np.mean(sub, axis=1), q_allsub_fix)))

q_allsub_fix_inc_tmean = list(map(lambda sub: sub[list(range(0,6,2))], q_allsub_fix_tmean))
df_mlm_inc_off_task = pd.DataFrame(q_allsub_fix_inc_tmean, columns=range(1,4))
df_mlm_inc_off_task.insert(0, 'subj_ID', [int(i) for i in subj_lst])
df_mlm_inc_off_task['task'] = 'Incongruent'
df_mlm_inc_melt_off = pd.melt(df_mlm_inc_off_task, id_vars=['subj_ID', 'task'], var_name='block', \
                          value_name='mu_mod_idx')

q_allsub_fix_con_tmean = list(map(lambda sub: sub[list(range(1,7,2))], q_allsub_fix_tmean))
df_mlm_con_off_task = pd.DataFrame(q_allsub_fix_con_tmean, columns=range(1,4))
df_mlm_con_off_task.insert(0, 'subj_ID', [int(i) for i in subj_lst])
df_mlm_con_off_task['task'] = 'Congruent'
df_mlm_con_melt_off = pd.melt(df_mlm_con_off_task, id_vars=['subj_ID', 'task'], var_name='block', \
                          value_name='mu_mod_idx')

# concat con and inc melt df
df_mlm_off_task = pd.concat([df_mlm_con_melt_off, df_mlm_inc_melt_off], ignore_index=True)
# print(df_mlm_off_task)
df_mlm_off_task.to_csv('off_task_block_mu_mod_idx.csv', index=False)


# FULL TS
# get data for full timescale of task block
q_allsub_fix_inc_ts = list(map(lambda sub: np.array(list(enumerate(\
                        sub[list(range(0,6,2))], 1)), dtype=object).ravel(), q_allsub_fix))
df_mlm_inc_off_task_ts = pd.DataFrame(q_allsub_fix_inc_ts)

q_allsub_fix_con_ts = list(map(lambda sub: np.array(list(enumerate(\
                        sub[list(range(1,6,2))], 1)), dtype=object).ravel(), q_allsub_fix))
df_mlm_con_off_task_ts = pd.DataFrame(q_allsub_fix_con_ts)

# df for each task, all frames - incongruent
df_off_inc_task1 = pd.DataFrame(df_mlm_inc_off_task_ts[1].values.tolist(), columns=range(1,6))
df_off_inc_task1.insert(0, 'block', df_mlm_inc_off_task_ts[0].values.tolist())
df_off_inc_task1.insert(0, 'subj_ID', [int(i) for i in subj_lst])

df_off_inc_task2 = pd.DataFrame(df_mlm_inc_off_task_ts[3].values.tolist(), columns=range(1,6))
df_off_inc_task2.insert(0, 'block', df_mlm_inc_off_task_ts[2].values.tolist())
df_off_inc_task2.insert(0, 'subj_ID', [int(i) for i in subj_lst])

df_off_inc_task3 = pd.DataFrame(df_mlm_inc_off_task_ts[5].values.tolist(), columns=range(1,6))
df_off_inc_task3.insert(0, 'block', df_mlm_inc_off_task_ts[4].values.tolist())
df_off_inc_task3.insert(0, 'subj_ID', [int(i) for i in subj_lst])

df_mlm_off_ts_inc = pd.concat([df_off_inc_task1, df_off_inc_task2, df_off_inc_task3])
df_mlm_off_ts_inc['task'] = 'Incongruent'
df_mlm_off_ts_inc_melt = pd.melt(df_mlm_off_ts_inc, id_vars=['subj_ID', 'task', 'block'], \
                                 var_name='frame', value_name='mod_idx')

# df for each task, all frames - congruent
df_off_con_task1 = pd.DataFrame(df_mlm_con_off_task_ts[1].values.tolist(), columns=range(1,6))
df_off_con_task1.insert(0, 'block', df_mlm_con_off_task_ts[0].values.tolist())
df_off_con_task1.insert(0, 'subj_ID', [int(i) for i in subj_lst])

df_off_con_task2 = pd.DataFrame(df_mlm_con_off_task_ts[3].values.tolist(), columns=range(1,6))
df_off_con_task2.insert(0, 'block', df_mlm_con_off_task_ts[2].values.tolist())
df_off_con_task2.insert(0, 'subj_ID', [int(i) for i in subj_lst])

df_off_con_task3 = pd.DataFrame(df_mlm_con_off_task_ts[5].values.tolist(), columns=range(1,6))
df_off_con_task3.insert(0, 'block', df_mlm_con_off_task_ts[4].values.tolist())
df_off_con_task3.insert(0, 'subj_ID', [int(i) for i in subj_lst])

df_mlm_off_ts_con = pd.concat([df_off_con_task1, df_off_con_task2, df_off_con_task3])
df_mlm_off_ts_con['task'] = 'Congruent'
df_mlm_off_ts_con_melt = pd.melt(df_mlm_off_ts_con, id_vars=['subj_ID', 'task', 'block'], \
                                 var_name='frame', value_name='mod_idx')

# df off task melt full timescale
df_mlm_off_ts = pd.concat([df_mlm_off_ts_con_melt, df_mlm_off_ts_inc_melt])
df_mlm_off_ts.to_csv('off_task_block_mod_idx_ts.csv', index=False)
# print(df_mlm_off_ts)


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
# plt.savefig('subjs_all_net_cort/mlr/allsub_cortnet_stripplot_diff.png', dpi=300)
plt.show()


# print(df_coeffs)
# C - I