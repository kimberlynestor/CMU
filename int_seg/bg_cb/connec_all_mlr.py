"""


"""


import os
from os.path import join as opj
import sys
from operator import itemgetter
import itertools

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

from scipy.stats import zscore
from scipy import stats
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import seaborn as sns


# load edge time series by region
# save_connec(subj_lst, task='all', region='cort')
# save_connec(subj_lst, task='all', region='cb')
# save_connec(subj_lst, task='all', region='bg')
# save_connec(subj_lst, task='stroop', region='thal')

# save_connec_adj_cort(subj_lst, task='stroop', region='cb') # done
# save_connec_adj_cort(subj_lst, task='stroop', region='bg')
# save_connec_adj_cort(subj_lst, task='stroop', region='thal')


"""
import json
import ast

con_allsub_cort = [list(ast.literal_eval(i.replace(' ', '').replace('"', '').replace('array(', '').replace(')', '').replace('\n', ''))) for i in open(f'subjs_all_net_cb_adj_cort_stroop.csv', 'r').readlines()]

# con_allsub_cort = ast.literal_eval(con_allsub_cort)
# con_allsub_cort = json.loads(con_allsub_cort)
# print(con_allsub_cort[0:2])

print(np.array(con_allsub_cort)[0:2])

# print(   np.array(con_allsub_cort)[0:2]  )
# print(m_output[0:2].replace('"', '').replace('array(', '').replace(')', ''))
# print(  np.array(con_allsub_cort[0:2], dtype='float64')  )
# print(  np.array(list(float(con_allsub_cort[0]))).type()  )

# xx = ast.literal_eval('[[0,0,0], [0,0,1], [1,1,0]]')
# [[0, 0, 0], [0, 0, 1], [1, 1, 0]]
# print(xx[0])
"""


# average across nodes, then subjects, smooth, z
# cortical nodes
con_allsub_cort = np.load(f'subjs_all_net_cort_stroop.npy')
# con_allsub_cort = np.absolute( con_allsub_cort )
avg_con_cort = np.array(list(map(lambda sub: np.array(list(map(lambda t: \
                            np.average(np.absolute(t)) , sub))), con_allsub_cort)))
avg_all_con_cort = np.average(avg_con_cort, axis=0)
con_cort_smooth = gaussian_filter(avg_all_con_cort, sigma=1)
con_cort_smooth_z = stats.zscore(con_cort_smooth)
np.save(f'con_cort_smooth_z_stroop.npy', con_cort_smooth_z)


# cb nodes
# con_allsub_cb = np.load(f'subjs_all_net_cb_stroop.npy')
# con_allsub_cb = np.load(f'subjs_all_net_cb_adj_cort_stroop.npy', allow_pickle=True)
con_allsub_cb = [np.load(f'cb_adj_cort/{i}', allow_pickle=True) for i in \
                            os.listdir('cb_adj_cort')]
avg_con_cb = np.array(list(map(lambda sub: np.array(list(map(lambda t: \
                            np.average(np.absolute(t)) , sub))), con_allsub_cb)))
# avg_all_con_cb = np.average(avg_con_cb, axis=0)
avg_all_con_cb = np.ma.average(np.ma.masked_array(avg_con_cb, \
                                np.isnan(avg_con_cb)), axis=0)
con_cb_smooth = gaussian_filter(avg_all_con_cb, sigma=1)
con_cb_smooth_z = stats.zscore(con_cb_smooth)
np.save(f'con_cb_smooth_z_cort_stroop.npy', con_cb_smooth_z)

# bg nodes
# con_allsub_bg = np.load(f'subjs_all_net_bg_stroop.npy')
# con_allsub_bg = np.load(f'subjs_all_net_bg_adj_cort_stroop.npy', allow_pickle=True)
con_allsub_bg = [np.load(f'bg_adj_cort/{i}', allow_pickle=True) for i in \
                            os.listdir('bg_adj_cort')]
avg_con_bg = np.array(list(map(lambda sub: np.array(list(map(lambda t: \
                            np.average(np.absolute(t)) , sub))), con_allsub_bg)))
# avg_all_con_bg = np.average(avg_con_bg, axis=0)
avg_all_con_bg = np.ma.average(np.ma.masked_array(avg_con_bg, \
                                np.isnan(avg_con_bg)), axis=0)
con_bg_smooth = gaussian_filter(avg_all_con_bg, sigma=1)
con_bg_smooth_z = stats.zscore(con_bg_smooth)
np.save(f'con_bg_smooth_z_cort_stroop.npy', con_bg_smooth_z)

# thal nodes
# con_allsub_thal = np.load(f'subjs_all_net_thal_stroop.npy')
# con_allsub_thal = np.load(f'subjs_all_net_thal_adj_cort_stroop.npy', allow_pickle=True)
con_allsub_thal = [np.load(f'thal_adj_cort/{i}', allow_pickle=True) for i in \
                            os.listdir('thal_adj_cort')]
avg_con_thal = np.array(list(map(lambda sub: np.array(list(map(lambda t: \
                            np.average(np.absolute(t)) , sub))), con_allsub_thal)))
# avg_all_con_thal = np.average(avg_con_thal, axis=0)
avg_all_con_thal = np.ma.average(np.ma.masked_array(avg_con_thal, \
                                np.isnan(avg_con_thal)), axis=0)
con_thal_smooth = gaussian_filter(avg_all_con_thal, sigma=1)
con_thal_smooth_z = stats.zscore(con_thal_smooth)


## SINGLE CORTEX LINE GRAPH
## plot connectivity, all timepoint, all subjs - smooth, with task blocks
plt.plot(np.arange(0, frs*rt, 2), con_cort_smooth_z, lw=1, c=p_dict['cort_line'])

plt.xticks(np.arange(0, frs*rt, 60))
plt.xlabel('Time (s)', size=15, fontname='serif')
plt.ylabel('Connectivity (z-score)', size=15, fontname='serif')

# colour plot background with breakpoints
for i in range(len(inc_block)):
    plt.axvspan(inc_block[i][0], inc_block[i][1], facecolor=p_dict['Incongruent'], alpha=0.35) # tab:orange, 0.22
    plt.axvspan(con_block[i][0], con_block[i][1], facecolor=p_dict['Congruent'], alpha=0.35) # tab:blue, 0.2, #91C1E2
plt.legend(handles=[inc_patch, con_patch], loc=1)
plt.tight_layout()
plt.savefig('subjs_all_net_cort/connec/allsub_cortnet_con_smooth_sig1_blocks_cc_abs.png', dpi=300)
plt.show()


# ALL IN ONE PLOT
## plot connectivity from cort, cb, bg, thal
plt.plot(np.arange(0, frs*rt, 2), con_cort_smooth_z, lw=1, c=p_dict['cort_line'], label='cort_line')
plt.plot(np.arange(0, frs*rt, 2), con_cb_smooth_z, lw=1, c=p_dict['cb_line'], label='cb_line', ls='--')
plt.plot(np.arange(0, frs*rt, 2), con_bg_smooth_z, lw=1, c=p_dict['bg_line'], label='bg_line', ls='--')
# plt.plot(np.arange(0, frs*rt, 2), con_thal_smooth_z, lw=1, c=p_dict['thal_line'], label='thal_line')

plt.xticks(np.arange(0, frs*rt, 60))
plt.xlabel('Time (s)', size=15, fontname='serif')
plt.ylabel('Connectivity (z-score)', size=15, fontname='serif')

# colour plot background with breakpoints
for i in range(len(inc_block)):
    plt.axvspan(inc_block[i][0], inc_block[i][1], facecolor=p_dict['Incongruent'], alpha=0.35)
    plt.axvspan(con_block[i][0], con_block[i][1], facecolor=p_dict['Congruent'], alpha=0.35)
# plt.ylim(0.43, 0.45) # mask mod, no z
# plt.ylim(-3.5, 2.25)
# plt.legend(handles=[inc_patch, con_patch], loc=4)
plt.legend(handles=[inc_patch, con_patch, cort_line, bg_line, cb_line], prop={'size':7.5}, loc=1)
plt.tight_layout()
plt.savefig('subjs_all_net_cort/connec/allsub_cort_cb_bg_con_smooth_sig1_blocks_cc_abs.png', dpi=300)
plt.show()



# line graph of connectivity - SUBPLOTS WITH CORT, BG, CB, THAL
## plot mod max, all timepoint, all subjs - smooth, with task blocks
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12,7), sharex=True)
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(9,7), sharex=True)

ax1.plot(np.arange(0, frs*rt, 2), con_cort_smooth_z, lw=1, c=p_dict['cort_line'])
ax1.plot(np.arange(0, frs*rt, 2), con_bg_smooth_z, lw=1.5, c=p_dict['bg_line'], ls=':', alpha=0.7)
ax1.get_xaxis().set_visible(False)

ax2.plot(np.arange(0, frs*rt, 2), con_cort_smooth_z, lw=1, c=p_dict['cort_line'])
ax2.plot(np.arange(0, frs*rt, 2), con_cb_smooth_z, lw=1.5, c=p_dict['cb_line'], ls=':', alpha=0.7)
ax2.get_xaxis().set_visible(False)

ax3.plot(np.arange(0, frs*rt, 2), con_cort_smooth_z, lw=1, c=p_dict['cort_line'])
ax3.plot(np.arange(0, frs*rt, 2), con_thal_smooth_z, lw=1.5, c=p_dict['thal_line'], ls=':', alpha=0.7)
ax3.get_xaxis().set_visible(False)

ax4.plot(np.arange(0, frs*rt, 2), con_bg_smooth_z, lw=1, c=p_dict['bg_line'], ls='-', alpha=0.7)
ax4.plot(np.arange(0, frs*rt, 2), con_cb_smooth_z, lw=1, c=p_dict['cb_line'], ls='-', alpha=0.7)
ax4.plot(np.arange(0, frs*rt, 2), con_thal_smooth_z, lw=1, c=p_dict['thal_line'], ls='-', alpha=0.7)

plt.xticks(np.arange(0, frs*rt, 60))
plt.xlabel('Time (s)', size=13, fontname='serif')
fig.text(0.003, 0.5, 'Connectivity (z-score)', size=11, fontname='serif', va='center', rotation='vertical')


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
# ax1.set_ylim(-3.5, 3.25)
# ax2.set_ylim(-3.5, 3.25)
# ax3.set_ylim(-3.5, 3.25)
# plt.legend(handles=[inc_patch, con_patch], loc=4)
ax1.legend(handles=[inc_patch, con_patch, cort_line, bg_line, cb_line, thal_line], loc=1, prop={'size':6.75})
plt.tight_layout()
plt.savefig('subjs_all_net_cort/connec/allsub_cort_cb_bg_thal_adj_con_smooth_sig1_blocks_cc_subplots_abs.png', dpi=300)
plt.show()
##############



# absolute connectivity for all subjects - smooth, z by subject
connec_cort_allsub_z = np.array(list(map(lambda i: stats.zscore(i), avg_con_cort)))
connec_cort_allsub_smooth = np.array(list(map(lambda i: gaussian_filter(i[5:], sigma=1), avg_con_cort)))
connec_cort_allsub_smooth_z = np.array(list(map(lambda i: stats.zscore(i), avg_con_cort)))

connec_cb_allsub_z = np.array(list(map(lambda i: stats.zscore(i), avg_con_cb)))
connec_cb_allsub_smooth = np.array(list(map(lambda i: gaussian_filter(i[5:], sigma=1), avg_con_cb)))
connec_cb_allsub_smooth_z = np.array(list(map(lambda i: stats.zscore(i), avg_con_cb)))

connec_bg_allsub_z = np.array(list(map(lambda i: stats.zscore(i), avg_con_bg)))
connec_bg_allsub_smooth = np.array(list(map(lambda i: gaussian_filter(i[5:], sigma=1), avg_con_bg)))
connec_bg_allsub_smooth_z = np.array(list(map(lambda i: stats.zscore(i), avg_con_bg)))

connec_thal_allsub_z = np.array(list(map(lambda i: stats.zscore(i), avg_con_thal)))
connec_thal_allsub_smooth = np.array(list(map(lambda i: gaussian_filter(i[5:], sigma=1), avg_con_thal)))
connec_thal_allsub_smooth_z = np.array(list(map(lambda i: stats.zscore(i), avg_con_thal)))

##### SEPARATE INC AND CON BLOCKS
# for all subjects get only inc and con connec
connec_cort_allsub_inc = np.array(list(map(lambda subj: np.array(list(map(lambda block: \
                                subj[block], inc_frames_idx))), connec_cort_allsub_z)))
connec_cort_allsub_con = np.array(list(map(lambda subj: np.array(list(map(lambda block: \
                                subj[block], con_frames_idx))), connec_cort_allsub_z)))
connec_cort_allsub_fix = np.array(list(map(lambda subj: np.array(list(map(lambda block: \
                                    subj[block] , fix_frames_idx))), connec_cort_allsub_z)))

connec_cb_allsub_inc = np.array(list(map(lambda subj: np.array(list(map(lambda block: \
                                subj[block], inc_frames_idx))), connec_cb_allsub_z)))
connec_cb_allsub_con = np.array(list(map(lambda subj: np.array(list(map(lambda block: \
                                subj[block], con_frames_idx))), connec_cb_allsub_z)))

connec_bg_allsub_inc = np.array(list(map(lambda subj: np.array(list(map(lambda block: \
                                subj[block], inc_frames_idx))), connec_bg_allsub_z)))
connec_bg_allsub_con = np.array(list(map(lambda subj: np.array(list(map(lambda block: \
                                subj[block], con_frames_idx))), connec_bg_allsub_z)))

# make dataframe for plotting
## CORTEX
connec_cort_inc_avg = np.mean(connec_cort_allsub_inc, axis=0)
connec_cort_inc_sep_blocks = np.array([np.array(list(map(lambda x: [x, i[0]], i[1])))  \
                    for i in np.array(list(enumerate(connec_cort_inc_avg, 1)) , dtype=object)])
connec_cort_inc_sep_blocks = list(map(lambda xxx:[xxx[0], int(xxx[1]), 'Incongruent'], \
                    np.array([np.array(list(map(lambda x:[x[0], x[1]+j], i))) \
                    for i,j in zip(connec_cort_inc_sep_blocks, range(4))]).reshape((120, 2)) ))


connec_cort_con_avg = np.mean(connec_cort_allsub_con, axis=0)
connec_cort_con_sep_blocks = list(map(lambda xxx:[xxx[0], int(xxx[1]), 'Congruent'], \
                        np.array([np.array(list(map(lambda x: [x, i[0]], i[1])))  \
                            for i in np.array(list(enumerate(connec_cort_con_avg, 1))  , \
                                dtype=object )]).reshape((120, 2))))
connec_cort_con_sep_blocks = list(map(lambda x: [x[0], x[1]+x[1], x[2]], connec_cort_con_sep_blocks))


connec_cort_fix_avg = np.mean(connec_cort_allsub_fix, axis=0)
connec_cort_fix_sep_blocks = list(map(lambda xxx:[xxx[0], int(xxx[1]), 'Fixation'], \
                        np.array([np.array(list(map(lambda x: [x, i[0]], i[1])))  \
                            for i in np.array(list(enumerate(connec_cort_fix_avg, 9))  , \
                                dtype=object )]).reshape((35, 2))))
connec_cort_fix_off_task_sep = list(map(lambda i: [i[0], i[1], 'Incongruent Fixation'] \
                            if i[1] in range(9, 16, 2) else [i[0], i[1], 'Congruent Fixation'], \
                              connec_cort_fix_sep_blocks))


df_connec_cort_sep_blocks = pd.DataFrame(list(itertools.chain(connec_cort_inc_sep_blocks, \
                    connec_cort_con_sep_blocks)), columns=['con', 'block', 'task'])

df_cort_task = df_connec_cort_sep_blocks[(df_connec_cort_sep_blocks['task'] == 'Incongruent') | \
                          (df_connec_cort_sep_blocks['task'] == 'Congruent')]
df_cort_fix = pd.DataFrame(connec_cort_fix_off_task_sep, columns=['con', 'block', 'task'])


## CB
connec_cb_inc_avg = np.mean(connec_cb_allsub_inc, axis=0)
connec_cb_inc_sep_blocks = np.array([np.array(list(map(lambda x: [x, i[0]], i[1])))  \
                    for i in np.array(list(enumerate(connec_cb_inc_avg, 1)) , dtype=object)])
connec_cb_inc_sep_blocks = list(map(lambda xxx:[xxx[0], int(xxx[1]), 'Incongruent'], \
                    np.array([np.array(list(map(lambda x:[x[0], x[1]+j], i))) \
                    for i,j in zip(connec_cb_inc_sep_blocks, range(4))]).reshape((120, 2)) ))

connec_cb_con_avg = np.mean(connec_cb_allsub_con, axis=0)
connec_cb_con_sep_blocks = list(map(lambda xxx:[xxx[0], int(xxx[1]), 'Congruent'], \
                        np.array([np.array(list(map(lambda x: [x, i[0]], i[1])))  \
                            for i in np.array(list(enumerate(connec_cb_con_avg, 1))  , \
                                dtype=object )]).reshape((120, 2))))
connec_cb_con_sep_blocks = list(map(lambda x: [x[0], x[1]+x[1], x[2]], connec_cb_con_sep_blocks))

df_connec_cb_sep_blocks = pd.DataFrame(list(itertools.chain(connec_cb_inc_sep_blocks, \
                    connec_cb_con_sep_blocks)), columns=['connec', 'block', 'task'])
df_connec_cb_sep_blocks.to_csv('df_connec_cb_sep_blocks_stroop.csv', index=False)

## BG
connec_bg_inc_avg = np.mean(connec_bg_allsub_inc, axis=0)
connec_bg_inc_sep_blocks = np.array([np.array(list(map(lambda x: [x, i[0]], i[1])))  \
                    for i in np.array(list(enumerate(connec_bg_inc_avg, 1)) , dtype=object)])
connec_bg_inc_sep_blocks = list(map(lambda xxx:[xxx[0], int(xxx[1]), 'Incongruent'], \
                    np.array([np.array(list(map(lambda x:[x[0], x[1]+j], i))) \
                    for i,j in zip(connec_bg_inc_sep_blocks, range(4))]).reshape((120, 2)) ))

connec_bg_con_avg = np.mean(connec_bg_allsub_con, axis=0)
connec_bg_con_sep_blocks = list(map(lambda xxx:[xxx[0], int(xxx[1]), 'Congruent'], \
                        np.array([np.array(list(map(lambda x: [x, i[0]], i[1])))  \
                            for i in np.array(list(enumerate(connec_bg_con_avg, 1))  , \
                                dtype=object )]).reshape((120, 2))))
connec_bg_con_sep_blocks = list(map(lambda x: [x[0], x[1]+x[1], x[2]], connec_bg_con_sep_blocks))

df_connec_bg_sep_blocks = pd.DataFrame(list(itertools.chain(connec_bg_inc_sep_blocks, \
                    connec_bg_con_sep_blocks)), columns=['connec', 'block', 'task'])
df_connec_bg_sep_blocks.to_csv('df_connec_bg_sep_blocks_stroop.csv', index=False)


###### POINTPLOT OF CI CORT - two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
sns.pointplot(ax=ax1, x='block', y='con', hue='task', data=df_cort_task, \
                            join=False, palette=p_dict)
sns.pointplot(ax=ax2, x='block', y='con', hue='task', data=df_cort_fix[ \
                    df_cort_fix['block'] != 15], join=False, palette=p_dict)
plt.xticks(np.arange(0,6), labels=[f'F{i}' for i in range(1, 7)])
ax1.set_xlabel('On Task Block', size=12, fontname='serif')
ax2.set_xlabel('Off Task Block', size=12, fontname='serif')
ax1.set_ylabel('Connectivity (z-score)', size=12, fontname='serif')
ax2.set_ylabel('')
ax1.legend((inc_circ, con_circ), ('Incongruent', 'Congruent'), numpoints=1, loc=1)
ax2.legend('', frameon=False)
fig.legend('', frameon=False)
plt.tight_layout()
plt.savefig('subjs_all_net_cort/connec/allsub_cort_connec_pointplot_ci_subplots_cc.png', dpi=300)
plt.show()

##########


## WITHIN SUBJECT TEST - MLR
# fit multiple linear regression model to connectivity for all subs
mlr_coeffs_lst = []
for i in connec_cort_allsub_smooth_z:
    # lin reg model
    mlr = LinearRegression()
    mlr.fit(df_reg[['inc_reg', 'con_reg']], i)

    # slope
    mlr_slope = mlr.intercept_
    # coeffs, best fit line
    mlr_coeffs = mlr.coef_
    mlr_coeffs_lst.append(mlr_coeffs)
# print(mlr_coeffs_lst)

np.save('mlr_coeffs_subjs_all_net_cort_connec.npy', mlr_coeffs_lst)
df_coeffs = pd.DataFrame(mlr_coeffs_lst, columns=['inc_coeff', 'con_coeff'])

## inc and con beta coeff - single
inc_reg = np.mean(df_coeffs['inc_coeff'].values)
con_reg = np.mean(df_coeffs['con_coeff'].values)
inc_interval = st.norm.interval(alpha=0.95, loc=inc_reg, scale=st.sem(df_coeffs['inc_coeff'].values))
con_interval = st.norm.interval(alpha=0.95, loc=con_reg, scale=st.sem(df_coeffs['con_coeff'].values))

print(f'\ninc_coeff  {inc_reg}    {inc_interval}')
print(f'con_coeff  {con_reg}    {con_interval}\n')

# save model output to csv
m_output = open(f'subjs_all_net_cort/connec/connec_mlr_output_cort.csv', 'w')
m_output = open(f'subjs_all_net_cort/connec/connec_mlr_output_cort.csv', 'a')
writer = csv.writer(m_output)

writer.writerow(['inc_coeff ', inc_reg, inc_interval])
writer.writerow(['con_coeff ', con_reg, con_interval])


##### ON TASK
# prepare melt df with subj_id, task, block and mean connec by block
connec_allsub_con_tmean = np.array(list(map(lambda sub: np.mean(sub, axis=1), connec_cort_allsub_con)))
df_mlm_con = pd.DataFrame(connec_allsub_con_tmean, columns=range(1,5))
df_mlm_con.insert(0, 'subj_ID', [int(i) for i in subj_lst])
df_mlm_con['task'] = 'Congruent'#1
df_mlm_con_melt = pd.melt(df_mlm_con, id_vars=['subj_ID', 'task'], var_name='block', \
                          value_name='mu_connec')

connec_allsub_inc_tmean = np.array(list(map(lambda sub: np.mean(sub, axis=1), connec_cort_allsub_inc)))
df_mlm_inc = pd.DataFrame(connec_allsub_inc_tmean, columns=range(1,5))
df_mlm_inc.insert(0, 'subj_ID', [int(i) for i in subj_lst])
df_mlm_inc['task'] = 'Incongruent'#0
df_mlm_inc_melt = pd.melt(df_mlm_inc, id_vars=['subj_ID', 'task'], var_name='block', \
                          value_name='mu_connec')

# concat con and inc melt df
df_mlm = pd.concat([df_mlm_con_melt, df_mlm_inc_melt], ignore_index=True)
# print(df_mlm)
df_mlm.to_csv('on_task_block_mu_connec.csv', index=False)


##### OFF TASK
connec_allsub_fix_tmean = np.array(list(map(lambda sub: np.mean(sub, axis=1), connec_cort_allsub_fix)))

connec_allsub_fix_inc_tmean = list(map(lambda sub: sub[list(range(0,6,2))], connec_allsub_fix_tmean))
df_mlm_inc_off_task = pd.DataFrame(connec_allsub_fix_inc_tmean, columns=range(1,4))
df_mlm_inc_off_task.insert(0, 'subj_ID', [int(i) for i in subj_lst])
df_mlm_inc_off_task['task'] = 'Incongruent'
df_mlm_inc_melt_off = pd.melt(df_mlm_inc_off_task, id_vars=['subj_ID', 'task'], var_name='block', \
                          value_name='mu_connec')

connec_allsub_fix_con_tmean = list(map(lambda sub: sub[list(range(1,7,2))], connec_allsub_fix_tmean))
df_mlm_con_off_task = pd.DataFrame(connec_allsub_fix_con_tmean, columns=range(1,4))
df_mlm_con_off_task.insert(0, 'subj_ID', [int(i) for i in subj_lst])
df_mlm_con_off_task['task'] = 'Congruent'
df_mlm_con_melt_off = pd.melt(df_mlm_con_off_task, id_vars=['subj_ID', 'task'], var_name='block', \
                          value_name='mu_connec')

# concat con and inc melt df
df_mlm_off_task = pd.concat([df_mlm_con_melt_off, df_mlm_inc_melt_off], ignore_index=True)
# print(df_mlm_off_task)
df_mlm_off_task.to_csv('off_task_block_mu_connec.csv', index=False)



# FULL TS, BG-CTX
df_mlm_ts = pd.DataFrame(connec_bg_allsub_smooth_z)
df_mlm_ts.columns +=1 # start column count from 1
df_mlm_ts.insert(0, 'subj_ID', [int(i) for i in subj_lst])
df_mlm_ts_melt = pd.melt(df_mlm_ts, id_vars=['subj_ID'], \
                                 var_name='frame', value_name='connec_bg')
df_mlm_ts_melt.to_csv('bg_connec_ts.csv', index=False)

# FULL TS, CB-CTX
df_mlm_ts = pd.DataFrame(connec_cb_allsub_smooth_z)
df_mlm_ts.columns +=1 # start column count from 1
df_mlm_ts.insert(0, 'subj_ID', [int(i) for i in subj_lst])
df_mlm_ts_melt = pd.melt(df_mlm_ts, id_vars=['subj_ID'], \
                                 var_name='frame', value_name='connec_cb')
df_mlm_ts_melt.to_csv('cb_connec_ts.csv', index=False)

# FULL TS,THAL-CTX
df_mlm_ts = pd.DataFrame(connec_thal_allsub_smooth_z)
df_mlm_ts.columns +=1 # start column count from 1
df_mlm_ts.insert(0, 'subj_ID', [int(i) for i in subj_lst])
df_mlm_ts_melt = pd.melt(df_mlm_ts, id_vars=['subj_ID'], \
                                 var_name='frame', value_name='connec_thal')
df_mlm_ts_melt.to_csv('thal_connec_ts.csv', index=False)

