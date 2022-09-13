"""

Participation coefficient (signed): https://tinyurl.com/2p89y43a
"""

from os.path import join as opj
import sys
import itertools
import csv

import bct

import jr_funcs as jr
from bg_cb_funcs import *
from task_blocks import *

import numpy as np
import pandas as pd
import math
from scipy import stats
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression
from scipy import stats
import scipy.stats as st

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.lines import Line2D

# np.set_printoptions(threshold=sys.maxsize)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_columns', None)

rt = 2
frs = 280


# get subject list
subj_lst = np.loadtxt(opj(main_dir, dep_path, 'subjects_intersect_motion_035.txt'))


# shen parcels region assignment - brain areas from Andrew Gerlach
df_net_assign = pd.read_csv(opj(main_dir, dep_path, 'shen_268_parcellation_networklabels_mod.csv'))
ci = df_net_assign.iloc[:,1].values

# get individual node info only
c_idx = node_info('cort')[0]
ci_cort = node_info('cort')[1]

cb_idx = node_info('cb')[0]
ci_cb = node_info('cb')[1]

bg_idx = node_info('bg')[0]
ci_bg = node_info('bg')[1]

thal_idx = node_info('thal')[0]
ci_thal = node_info('thal')[1]


"""
# get efc matrices - bg, cb, cort - ALL SUBJS
task = 'stroop'
out_name = 'subjs_all_net_'

p_coeff_cort_subj_lst = []
p_coeff_cb_subj_lst = []
p_coeff_bg_subj_lst = []
p_coeff_thal_subj_lst = []
for subj in subj_lst:
    # efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)
    efc_mat = np.absolute( jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj) )

    efc_mat_cort = np.array(list(map(lambda frame: np.array(list(map(lambda col: \
                                        col[c_idx], frame[c_idx]))), efc_mat)))
    efc_mat_cb = np.array(list(map(lambda frame: np.array(list(map(lambda col: \
                                        col[cb_idx], frame[cb_idx]))), efc_mat)))
    efc_mat_bg = np.array(list(map(lambda frame: np.array(list(map(lambda col: \
                                        col[bg_idx], frame[bg_idx]))), efc_mat)))
    efc_mat_thal = np.array(list(map(lambda frame: np.array(list(map(lambda col: \
                                        col[thal_idx], frame[thal_idx]))), efc_mat)))


    # p_coef_cort = list(map(lambda x: np.average(bct.participation_coef_sign(x, ci_cort)), efc_mat_cort))
    # p_coef_cb = list(map(lambda x: np.average(bct.participation_coef_sign(x, ci_cb)), efc_mat_cb))
    # p_coef_bg = list(map(lambda x: np.average(bct.participation_coef_sign(x, ci_bg)), efc_mat_bg))
    # p_coef_thal = list(map(lambda x: np.average(bct.participation_coef_sign(x, ci_thal)), efc_mat_thal))

    p_coef_cort = list(map(lambda x: np.average(bct.participation_coef(x, ci_cort)), efc_mat_cort))
    p_coef_cb = list(map(lambda x: np.average(bct.participation_coef(x, ci_cb)), efc_mat_cb))
    p_coef_bg = list(map(lambda x: np.average(bct.participation_coef(x, ci_bg)), efc_mat_bg))
    p_coef_thal = list(map(lambda x: np.average(bct.participation_coef(x, ci_thal)), efc_mat_thal))

    p_coeff_cort_subj_lst.append(p_coef_cort)
    p_coeff_cb_subj_lst.append(p_coef_cb)
    p_coeff_bg_subj_lst.append(p_coef_bg)
    p_coeff_thal_subj_lst.append(p_coef_thal)

# np.save(f'{out_name}cort_p_coeff_{task}.npy', p_coeff_cort_subj_lst)
# np.save(f'{out_name}cb_p_coeff_{task}.npy', p_coeff_cb_subj_lst)
# np.save(f'{out_name}bg_p_coeff_{task}.npy', p_coeff_bg_subj_lst)
# np.save(f'{out_name}thal_p_coeff_{task}.npy', p_coeff_thal_subj_lst)

np.save(f'{out_name}cort_p_coeff_{task}_abs.npy', p_coeff_cort_subj_lst)
np.save(f'{out_name}cb_p_coeff_{task}_abs.npy', p_coeff_cb_subj_lst)
np.save(f'{out_name}bg_p_coeff_{task}_abs.npy', p_coeff_bg_subj_lst)
np.save(f'{out_name}thal_p_coeff_{task}_abs.npy', p_coeff_thal_subj_lst)
"""

"""
# get efc matrices, for SINGLE SUBJ in subj_lst
task = 'stroop'
efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj_lst[0])

efc_mat_cort = np.array(list(map(lambda frame: np.array(list(map(lambda col: \
                    col[c_idx], frame[c_idx]))), efc_mat)))
efc_mat_cb = np.array(list(map(lambda frame: np.array(list(map(lambda col: \
                    col[cb_idx], frame[cb_idx]))), efc_mat)))
efc_mat_bg = np.array(list(map(lambda frame: np.array(list(map(lambda col: \
                    col[bg_idx], frame[bg_idx]))), efc_mat)))

# calculate p_coeff all separately
p_coef_cort = list(map( lambda x: np.average(bct.participation_coef_sign(x, ci_cort)), efc_mat_cort))
p_coef_cb = list(map( lambda x: np.average(bct.participation_coef_sign(x, ci_cb)), efc_mat_cb))
p_coef_bg = list(map( lambda x: np.average(bct.participation_coef_sign(x, ci_bg)), efc_mat_bg)) # flat, 0 because all in network 4
"""



# get p_coeff for all subjs
p_coef_cort_allsub = np.load(f'subjs_all_net_cort_p_coeff_stroop.npy')
p_coef_cb_allsub = np.load(f'subjs_all_net_cb_p_coeff_stroop.npy')
p_coef_bg_allsub = np.load(f'subjs_all_net_bg_p_coeff_stroop.npy')
p_coef_thal_allsub = np.load(f'subjs_all_net_thal_p_coeff_stroop.npy')

# p_coef_cort_allsub = np.load(f'subjs_all_net_cort_p_coeff_stroop_abs.npy')
# p_coef_cb_allsub = np.load(f'subjs_all_net_cb_p_coeff_stroop_abs.npy')
# p_coef_bg_allsub = np.load(f'subjs_all_net_bg_p_coeff_stroop_abs.npy')
# p_coef_thal_allsub = np.load(f'subjs_all_net_thal_p_coeff_stroop_abs.npy')

p_coef_cort_allsub_z = np.array(list(map(lambda i: stats.zscore(i), p_coef_cort_allsub)))
p_coef_cb_allsub_z = np.array(list(map(lambda i: stats.zscore(i), p_coef_cb_allsub)))
p_coef_bg_allsub_z = np.array(list(map(lambda i: stats.zscore(i), p_coef_bg_allsub)))
p_coef_thal_allsub_z = np.array(list(map(lambda i: stats.zscore(i), p_coef_thal_allsub)))

p_coef_cort_allsub_smooth = np.array(list(map(lambda i: gaussian_filter(i, sigma=1), p_coef_cort_allsub)))
p_coef_cb_allsub_smooth = np.array(list(map(lambda i: gaussian_filter(i, sigma=1), p_coef_cb_allsub)))
p_coef_bg_allsub_smooth = np.array(list(map(lambda i: gaussian_filter(i, sigma=1), p_coef_bg_allsub)))
p_coef_thal_allsub_smooth = np.array(list(map(lambda i: gaussian_filter(i, sigma=1), p_coef_thal_allsub)))

p_coef_cort_allsub_smooth_z = np.array(list(map(lambda i: stats.zscore(i), p_coef_cort_allsub_smooth)))
p_coef_cb_allsub_smooth_z = np.array(list(map(lambda i: stats.zscore(i), p_coef_cb_allsub_smooth)))
p_coef_bg_allsub_smooth_z = np.array(list(map(lambda i: stats.zscore(i), p_coef_bg_allsub_smooth)))
p_coef_thal_allsub_smooth_z = np.array(list(map(lambda i: stats.zscore(i), p_coef_thal_allsub_smooth)))


# avg p_coeff across subjs, smooth and z score
p_coeff_cort_avg = np.average(p_coef_cort_allsub, axis=0)
p_coeff_cort_avg_smooth = gaussian_filter(p_coeff_cort_avg, sigma=1)
p_coeff_cort_avg_smooth_z = stats.zscore(p_coeff_cort_avg_smooth)

p_coeff_cb_avg = np.average(p_coef_cb_allsub, axis=0)
p_coeff_cb_avg_smooth = gaussian_filter(p_coeff_cb_avg, sigma=1)
p_coeff_cb_avg_smooth_z = stats.zscore(p_coeff_cb_avg_smooth)

p_coeff_bg_avg = np.average(p_coef_bg_allsub, axis=0)
p_coeff_bg_avg_smooth = gaussian_filter(p_coeff_bg_avg, sigma=1)
p_coeff_bg_avg_smooth_z = stats.zscore(p_coeff_bg_avg_smooth) # nan due to 0 vals

p_coeff_thal_avg = np.average(p_coef_thal_allsub, axis=0)
p_coeff_thal_avg_smooth = gaussian_filter(p_coeff_thal_avg, sigma=1)
p_coeff_thal_avg_smooth_z = stats.zscore(p_coeff_thal_avg_smooth) # nan due to 0 vals


#### P_COEFF LINE GRAPH - ALL TOGETHER
# plot part coeff, all timepoint
# plt.plot(np.arange(0, frs*rt, 2), p_coef_all, linewidth=1)
plt.plot(np.arange(0, frs*rt, 2), p_coeff_cort_avg_smooth_z, lw=1, \
                            label='p_coef_cort', c=p_dict['cort_line'])
plt.plot(np.arange(0, frs*rt, 2), p_coeff_cb_avg_smooth_z, lw=1, label='p_coef_cb', \
                            c=p_dict['cb_line'], ls='--')
plt.plot(np.arange(0, frs*rt, 2), p_coeff_bg_avg_smooth, lw=1, label='p_coef_bg', \
                            c=p_dict['bg_line'] , ls='--') # flat

plt.xticks(np.arange(0, frs*rt, 60))
plt.xlabel('Time (s)', size=11, fontname='serif')
plt.ylabel('Participation coefficient (z-score)', size=11, fontname='serif') # integration
# colour plot background with breakpoints
for i in range(len(inc_block)):
    # incongruent
    plt.axvspan(inc_block[i][0], inc_block[i][1], facecolor=p_dict['Incongruent'], alpha=0.35) # tab:orange, 0.22
    # congruent
    plt.axvspan(con_block[i][0], con_block[i][1], facecolor=p_dict['Congruent'], alpha=0.35) # tab:blue, 0.2, #91C1E2
plt.legend(handles=[inc_patch, con_patch, cort_line, bg_line, cb_line], prop={'size':7.5}, loc=1)
plt.tight_layout()
plt.savefig('subjs_all_net_cort/p_coef/allsub_cort_cb_bg_p_coeff_smooth_z_cc.png', dpi=300)
plt.show()


## SINGLE CORTEX LINE GRAPH
plt.plot(np.arange(0, frs*rt, 2), p_coeff_cort_avg_smooth_z, lw=1, c=p_dict['cort_line'])
plt.xticks(np.arange(0, frs*rt, 60))
plt.xlabel('Time (s)', size=15, fontname='serif')
plt.ylabel('Participation coefficient (z-score)', size=15, fontname='serif')

# colour plot background with breakpoints
for i in range(len(inc_block)):
    plt.axvspan(inc_block[i][0], inc_block[i][1], facecolor=p_dict['Incongruent'], alpha=0.35)
    plt.axvspan(con_block[i][0], con_block[i][1], facecolor=p_dict['Congruent'], alpha=0.35)
plt.legend(handles=[inc_patch, con_patch], loc=4, prop={'size':9.75})
plt.tight_layout()
# plt.savefig('subjs_all_net_cort/p_coef/allsub_cortnet_p_coeff_smooth_sig1_blocks_cc.png', dpi=300)
plt.show()



# line graph of p_coeff - SUBPLOTS WITH CORT, BG, CB, THAL
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12,7), sharex=True)

ax1.plot(np.arange(0, frs*rt, 2), p_coeff_cort_avg_smooth_z, lw=1, c=p_dict['cort_line'])
ax1.plot(np.arange(0, frs*rt, 2), p_coeff_bg_avg_smooth, lw=1.5, c=p_dict['bg_line'], ls=':', alpha=0.7)
ax1.get_xaxis().set_visible(False)

ax2.plot(np.arange(0, frs*rt, 2), p_coeff_cort_avg_smooth_z, lw=1, c=p_dict['cort_line'])
ax2.plot(np.arange(0, frs*rt, 2), p_coeff_cb_avg_smooth_z, lw=1.5, c=p_dict['cb_line'], ls=':', alpha=0.7)
ax2.get_xaxis().set_visible(False)

ax3.plot(np.arange(0, frs*rt, 2), p_coeff_cort_avg_smooth_z, lw=1, c=p_dict['cort_line'])
ax3.plot(np.arange(0, frs*rt, 2), p_coeff_thal_avg_smooth, lw=1.5, c=p_dict['thal_line'], ls=':', alpha=0.7)
ax3.get_xaxis().set_visible(False)

ax4.plot(np.arange(0, frs*rt, 2), p_coeff_bg_avg_smooth, lw=1, c=p_dict['bg_line'], ls='-', alpha=0.7)
ax4.plot(np.arange(0, frs*rt, 2), p_coeff_cb_avg_smooth_z, lw=1, c=p_dict['cb_line'], ls='-', alpha=1)
ax4.plot(np.arange(0, frs*rt, 2), p_coeff_thal_avg_smooth, lw=1, c=p_dict['thal_line'], ls='--', alpha=0.45)

plt.xticks(np.arange(0, frs*rt, 60))
plt.xlabel('Time (s)', size=13, fontname='serif')
fig.text(0.003, 0.5, 'Participation coefficient (z-score)', size=11, fontname='serif', va='center', rotation='vertical')


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

ax1.legend(handles=[inc_patch, con_patch, cort_line, bg_line, cb_line, thal_line], loc=1, prop={'size':6.75})
plt.tight_layout()
plt.savefig('subjs_all_net_cort/p_coef/allsub_cort_cb_bg_thal_p_coeff_smooth_sig1_blocks_cc_subplots.png', dpi=300)
plt.show()
##############


##### SEPARATE INC AND CON BLOCKS
# for all subjects get only inc and con mod
p_coef_cort_allsub_inc = np.array(list(map(lambda subj: np.array(list(map(lambda block: \
                                subj[block], inc_frames_idx))), p_coef_cort_allsub_z)))
p_coef_cort_allsub_con = np.array(list(map(lambda subj: np.array(list(map(lambda block: \
                                subj[block], con_frames_idx))), p_coef_cort_allsub_z)))
p_coef_cort_allsub_fix = np.array(list(map(lambda subj: np.array(list(map(lambda block: \
                                    subj[block] , fix_frames_idx))) ,p_coef_cort_allsub_z)))

p_coef_cb_allsub_inc = np.array(list(map(lambda subj: np.array(list(map(lambda block: \
                                subj[block], inc_frames_idx))), p_coef_cb_allsub_z)))
p_coef_cb_allsub_con = np.array(list(map(lambda subj: np.array(list(map(lambda block: \
                                subj[block], con_frames_idx))), p_coef_cb_allsub_z)))


# make dataframe for plotting
## CORTEX
p_coef_cort_inc_avg = np.mean(p_coef_cort_allsub_inc, axis=0)
p_coef_cort_inc_sep_blocks = np.array([np.array(list(map(lambda x: [x, i[0]], i[1])))  \
                    for i in np.array(list(enumerate(p_coef_cort_inc_avg, 1)) , dtype=object)])
p_coef_cort_inc_sep_blocks = list(map(lambda xxx:[xxx[0], int(xxx[1]), 'Incongruent'], \
                    np.array([np.array(list(map(lambda x:[x[0], x[1]+j], i))) \
                    for i,j in zip(p_coef_cort_inc_sep_blocks, range(4))]).reshape((120, 2)) ))

p_coef_cort_con_avg = np.mean(p_coef_cort_allsub_con, axis=0)
p_coef_cort_con_sep_blocks = list(map(lambda xxx:[xxx[0], int(xxx[1]), 'Congruent'], \
                        np.array([np.array(list(map(lambda x: [x, i[0]], i[1])))  \
                            for i in np.array(list(enumerate(p_coef_cort_con_avg, 1))  , \
                                dtype=object )]).reshape((120, 2))))
p_coef_cort_con_sep_blocks = list(map(lambda x: [x[0], x[1]+x[1], x[2]], p_coef_cort_con_sep_blocks))


p_coef_cort_fix_avg = np.mean(p_coef_cort_allsub_fix, axis=0)
p_coef_cort_fix_sep_blocks = list(map(lambda xxx:[xxx[0], int(xxx[1]), 'Fixation'], \
                        np.array([np.array(list(map(lambda x: [x, i[0]], i[1])))  \
                            for i in np.array(list(enumerate(p_coef_cort_fix_avg, 9))  , \
                                dtype=object )]).reshape((35, 2))))
p_coef_cort_fix_off_task_sep = list(map(lambda i: [i[0], i[1], 'Incongruent Fixation'] \
                            if i[1] in range(9, 16, 2) else [i[0], i[1], 'Congruent Fixation'], \
                              p_coef_cort_fix_sep_blocks))

df_p_coef_cort_sep_blocks = pd.DataFrame(list(itertools.chain(p_coef_cort_inc_sep_blocks, \
                    p_coef_cort_con_sep_blocks)), columns=['p_coeff', 'block', 'task'])

df_cort_task = df_p_coef_cort_sep_blocks[(df_p_coef_cort_sep_blocks['task'] == 'Incongruent') | \
                          (df_p_coef_cort_sep_blocks['task'] == 'Congruent')]
df_cort_fix = pd.DataFrame(p_coef_cort_fix_off_task_sep, columns=['p_coeff', 'block', 'task'])


## CB
p_coef_cb_inc_avg = np.mean(p_coef_cb_allsub_inc, axis=0)
p_coef_cb_inc_sep_blocks = np.array([np.array(list(map(lambda x: [x, i[0]], i[1])))  \
                    for i in np.array(list(enumerate(p_coef_cb_inc_avg, 1)) , dtype=object)])
p_coef_cb_inc_sep_blocks = list(map(lambda xxx:[xxx[0], int(xxx[1]), 'Incongruent'], \
                    np.array([np.array(list(map(lambda x:[x[0], x[1]+j], i))) \
                    for i,j in zip(p_coef_cb_inc_sep_blocks, range(4))]).reshape((120, 2)) ))

p_coef_cb_con_avg = np.mean(p_coef_cb_allsub_con, axis=0)
p_coef_cb_con_sep_blocks = list(map(lambda xxx:[xxx[0], int(xxx[1]), 'Congruent'], \
                        np.array([np.array(list(map(lambda x: [x, i[0]], i[1])))  \
                            for i in np.array(list(enumerate(p_coef_cb_con_avg, 1))  , \
                                dtype=object )]).reshape((120, 2))))
p_coef_cb_con_sep_blocks = list(map(lambda x: [x[0], x[1]+x[1], x[2]], p_coef_cb_con_sep_blocks))

df_p_coef_cb_sep_blocks = pd.DataFrame(list(itertools.chain(p_coef_cb_inc_sep_blocks, \
                    p_coef_cb_con_sep_blocks)), columns=['p_coeff', 'block', 'task'])


#### POINTPLOTS - CORT and CB SUBPLOTS
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
sns.pointplot(ax=ax1, x='block', y='p_coeff', hue='task', data=df_p_coef_cort_sep_blocks, \
                            join=False, palette=p_dict)
sns.pointplot(ax=ax2, x='block', y='p_coeff', hue='task', data=df_p_coef_cb_sep_blocks, \
                            join=False, palette=p_dict)
# plt.xticks(np.arange(0,7), labels=list(sum(zip(range(1,9), ['']*7), ()))+[8])
ax1.set_xlabel('Cortex', size=12, fontname='serif')
ax2.set_xlabel('Cerebellum', size=12, fontname='serif')
ax1.set_ylabel('Participation coefficient (z-score)', size=12, fontname="serif")
ax2.set_ylabel('')
# plt.legend((inc_circ, con_circ, fix_circ), ('Incongruent', 'Congruent', 'Fixation'), numpoints=1)
plt.tight_layout()
# plt.savefig('subjs_all_net_cort/p_coef/allsub_cort_cb_p_coeff_pointplot_ci_cc.png', dpi=300)
plt.show()


###### POINTPLOT OF CI CORT - two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
sns.pointplot(ax=ax1, x='block', y='p_coeff', hue='task', data=df_cort_task, \
                            join=False, palette=p_dict)
sns.pointplot(ax=ax2, x='block', y='p_coeff', hue='task', data=df_cort_fix[ \
                    df_cort_fix['block'] != 15], join=False, palette=p_dict)
plt.xticks(np.arange(0,6), labels=[f'F{i}' for i in range(1, 7)])
ax1.set_xlabel('On Task Block', size=12, fontname='serif')
ax2.set_xlabel('Off Task Block', size=12, fontname='serif')
ax1.set_ylabel('Participation coefficient', size=12, fontname='serif')
ax2.set_ylabel('')
ax1.legend((inc_circ, con_circ), ('Incongruent', 'Congruent'), numpoints=1, loc=4)
ax2.legend('', frameon=False)
fig.legend('', frameon=False)
plt.tight_layout()
plt.savefig('subjs_all_net_cort/p_coef/allsub_cort_p_coeff_pointplot_ci_subplots_cc.png', dpi=300)
plt.show()

#################


sys.exit()


## WITHIN SUBJECT TEST - MLR
# fit multiple linear regression model to p_coeff for all subs
mlr_coeffs_lst = []
for i in p_coef_cort_allsub_smooth_z:
    # lin reg model
    mlr = LinearRegression()
    mlr.fit(df_reg[['inc_reg', 'con_reg']], i)

    # slope
    mlr_slope = mlr.intercept_
    # coeffs, best fit line
    mlr_coeffs = mlr.coef_
    mlr_coeffs_lst.append(mlr_coeffs)
# print(mlr_coeffs_lst)

np.save('mlr_coeffs_subjs_all_net_cort_p_coeff.npy', mlr_coeffs_lst)
df_coeffs = pd.DataFrame(mlr_coeffs_lst, columns=['inc_coeff', 'con_coeff'])

## inc and con beta coeff - single
inc_reg = np.mean(df_coeffs['inc_coeff'].values)
con_reg = np.mean(df_coeffs['con_coeff'].values)
inc_interval = st.norm.interval(alpha=0.95, loc=inc_reg, scale=st.sem(df_coeffs['inc_coeff'].values))
con_interval = st.norm.interval(alpha=0.95, loc=con_reg, scale=st.sem(df_coeffs['con_coeff'].values))

print(f'\ninc_coeff  {inc_reg}    {inc_interval}')
print(f'con_coeff  {con_reg}    {con_interval}\n')

# save model output to csv
m_output = open(f'subjs_all_net_cort/p_coef/p_coeff_mlr_output_cort.csv', 'w')
m_output = open(f'subjs_all_net_cort/p_coef/p_coeff_mlr_output_cort.csv', 'a')
writer = csv.writer(m_output)

writer.writerow(['inc_coeff ', inc_reg, inc_interval])
writer.writerow(['con_coeff ', con_reg, con_interval])


##### ON TASK
# prepare melt df with subj_id, task, block and mean p_coeff by block
p_coef_allsub_con_tmean = np.array(list(map(lambda sub: np.mean(sub, axis=1), p_coef_cort_allsub_con)))
df_mlm_con = pd.DataFrame(p_coef_allsub_con_tmean, columns=range(1,5))
df_mlm_con.insert(0, 'subj_ID', [int(i) for i in subj_lst])
df_mlm_con['task'] = 'Congruent'#1
df_mlm_con_melt = pd.melt(df_mlm_con, id_vars=['subj_ID', 'task'], var_name='block', \
                          value_name='p_coef')

p_coef_allsub_inc_tmean = np.array(list(map(lambda sub: np.mean(sub, axis=1), p_coef_cort_allsub_inc)))
df_mlm_inc = pd.DataFrame(p_coef_allsub_inc_tmean, columns=range(1,5))
df_mlm_inc.insert(0, 'subj_ID', [int(i) for i in subj_lst])
df_mlm_inc['task'] = 'Incongruent'#0
df_mlm_inc_melt = pd.melt(df_mlm_inc, id_vars=['subj_ID', 'task'], var_name='block', \
                          value_name='p_coef')

# concat con and inc melt df
df_mlm = pd.concat([df_mlm_con_melt, df_mlm_inc_melt], ignore_index=True)
# print(df_mlm)
df_mlm.to_csv('on_task_block_mu_p_coef.csv', index=False)


##### OFF TASK
p_coef_allsub_fix_tmean = np.array(list(map(lambda sub: np.mean(sub, axis=1), p_coef_cort_allsub_fix)))

p_coef_allsub_fix_inc_tmean = list(map(lambda sub: sub[list(range(0,6,2))], p_coef_allsub_fix_tmean))
df_mlm_inc_off_task = pd.DataFrame(p_coef_allsub_fix_inc_tmean, columns=range(1,4))
df_mlm_inc_off_task.insert(0, 'subj_ID', [int(i) for i in subj_lst])
df_mlm_inc_off_task['task'] = 'Incongruent'
df_mlm_inc_melt_off = pd.melt(df_mlm_inc_off_task, id_vars=['subj_ID', 'task'], var_name='block', \
                          value_name='p_coef')

p_coef_allsub_fix_con_tmean = list(map(lambda sub: sub[list(range(1,7,2))], p_coef_allsub_fix_tmean))
df_mlm_con_off_task = pd.DataFrame(p_coef_allsub_fix_con_tmean, columns=range(1,4))
df_mlm_con_off_task.insert(0, 'subj_ID', [int(i) for i in subj_lst])
df_mlm_con_off_task['task'] = 'Congruent'
df_mlm_con_melt_off = pd.melt(df_mlm_con_off_task, id_vars=['subj_ID', 'task'], var_name='block', \
                          value_name='p_coef')

# concat con and inc melt df
df_mlm_off_task = pd.concat([df_mlm_con_melt_off, df_mlm_inc_melt_off], ignore_index=True)
# print(df_mlm_off_task)
df_mlm_off_task.to_csv('off_task_block_mu_p_coef.csv', index=False)
