"""


"""


import sys
import os
from pathlib import Path

# set shared scripts dir for both msit and stroop up one level
curr_dir = os.getcwd()
pars = Path(curr_dir).parents
par_dir = pars[0]
sys.path.insert(0, str(par_dir))

import jr_funcs as jr
from bg_cb_funcs import *
from task_blocks import *
from config import *

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import scipy.stats as st
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

from matplotlib import gridspec
import matplotlib.colors as colors


task = 'msit'


q_avg_smooth_z_cort = np.load(f'{main_dir}IntermediateData/{task}/q_avg_smooth_z_allpts_cort_{task}.npy')
q_avg_smooth_z_mask_cort = np.ma.array(np.insert(np.load(\
                            f'{main_dir}IntermediateData/{task}/q_avg_smooth_z_cort_{task}.npy'), 0, \
                                np.ones(5)), mask=np.pad(np.ones(5), (0,frs-5)))

eigen_cen_cb_avg_smooth_z = np.load(f'{main_dir}IntermediateData/{task}/eigen_cen_cb_avg_smooth_z_{task}.npy')
eigen_cen_bg_avg_smooth_z = np.load(f'{main_dir}IntermediateData/{task}/eigen_cen_bg_avg_smooth_z_{task}.npy')

print(f'\ncorr_coef bg x cb eigen: {np.corrcoef(eigen_cen_cb_avg_smooth_z, eigen_cen_bg_avg_smooth_z)[0][1]}')
print(f'\ncorr_coef cort mod x cb eigen: {np.corrcoef(q_avg_smooth_z_cort, eigen_cen_cb_avg_smooth_z)[0][1]}')
print(f'corr_coef cort mod x bg eigen: {np.corrcoef(q_avg_smooth_z_cort, eigen_cen_bg_avg_smooth_z)[0][1]}\n')


#### AVERAGE BLOCKS - MOD, EIGEN CB AND BG
q_allsub = np.load(f'{main_dir}IntermediateData/{task}/subjs_all_net_cort_q_{task}.npy')
q_allsub_pad = np.array(list(map(lambda sub:np.pad(sub, (0,10), mode='constant', \
                                                constant_values=np.nan), q_allsub)))
q_allsub_avg_blocks = np.array(list(map(lambda sub:np.nanmean(np.array(list(map(lambda i : \
                            sub[int(i[0]):int(i[1])+1], group_blocks))), axis=0), q_allsub_pad)))
q_allsub_avg_blocks_smooth = np.array(list(map(lambda i: gaussian_filter(i, sigma=1), q_allsub_avg_blocks)))
q_allsub_avg_blocks_smooth_z = np.array(list(map(lambda i: stats.zscore(i), q_allsub_avg_blocks_smooth)))
q_sterr_allpts = np.array(list(map(lambda i: st.sem(i), q_allsub_avg_blocks_smooth_z.T)))
q_ci_allpts = q_sterr_allpts*ci

eigen_cb_allsub = np.load(f'{main_dir}IntermediateData/{task}/eigen_cen_cb_allsub_{task}.npy')
eigen_cb_allsub_pad = np.array(list(map(lambda sub:np.pad(sub, (0,10), mode='constant', \
                                                constant_values=np.nan), eigen_cb_allsub)))
eigen_cb_allsub_avg_blocks = np.array(list(map(lambda sub:np.nanmean(np.array(list(map(lambda i : \
                            sub[int(i[0]):int(i[1])+1], group_blocks))), axis=0), eigen_cb_allsub_pad)))
eigen_cb_allsub_avg_blocks_smooth = np.array(list(map(lambda i: gaussian_filter(i, sigma=1), eigen_cb_allsub_avg_blocks)))
eigen_cb_allsub_avg_blocks_smooth_z = np.array(list(map(lambda i: stats.zscore(i), eigen_cb_allsub_avg_blocks_smooth)))
eigen_cb_sterr_allpts = np.array(list(map(lambda i: st.sem(i), eigen_cb_allsub_avg_blocks_smooth_z.T)))
eigen_cb_ci_allpts = eigen_cb_sterr_allpts*ci
eigen_cb_std_allpts = list(map(lambda i: np.std(i), eigen_cb_allsub_avg_blocks_smooth_z.T))

eigen_bg_allsub = np.load(f'{main_dir}IntermediateData/{task}/eigen_cen_bg_allsub_{task}.npy')
eigen_bg_allsub_pad = np.array(list(map(lambda sub:np.pad(sub, (0,10), mode='constant', \
                                                constant_values=np.nan), eigen_bg_allsub)))
eigen_bg_allsub_avg_blocks = np.array(list(map(lambda sub:np.nanmean(np.array(list(map(lambda i : \
                            sub[int(i[0]):int(i[1])+1], group_blocks))), axis=0), eigen_bg_allsub_pad)))
eigen_bg_allsub_avg_blocks_smooth = np.array(list(map(lambda i: gaussian_filter(i, sigma=1), eigen_bg_allsub_avg_blocks)))
eigen_bg_allsub_avg_blocks_smooth_z = np.array(list(map(lambda i: stats.zscore(i), eigen_bg_allsub_avg_blocks_smooth)))
eigen_bg_sterr_allpts = np.array(list(map(lambda i: st.sem(i), eigen_bg_allsub_avg_blocks_smooth_z.T)))
eigen_bg_ci_allpts = eigen_bg_sterr_allpts*ci
eigen_bg_std_allpts = list(map(lambda i: np.std(i), eigen_bg_allsub_avg_blocks_smooth_z.T))


## ALL IN ONE PLOT
## plot eigenvec cen from cb, bg -- with modularity of cort
plt.rcParams['figure.figsize'] = (8,4)

plt.plot(np.arange(0, frs*rt, 2), q_avg_smooth_z_cort, lw=1.5, c=p_dict['cort_line'], ls=':')
plt.plot(np.arange(0, frs*rt, 2), eigen_cen_bg_avg_smooth_z, lw=1, c=p_dict['bg_line_eig'])
plt.plot(np.arange(0, frs*rt, 2), eigen_cen_cb_avg_smooth_z, lw=1, c=p_dict['cb_line_eig'])

plt.xticks(np.arange(0, frs*rt, 60))
plt.ylim(-3.75, 5.75)
plt.xlabel('Time (s)', size=12, fontname=font_lst[0])
plt.ylabel('Z-scored values', size=12, fontname=font_lst[0])

# colour plot background with breakpoints
for i in range(len(inc_block)):
    plt.axvspan(inc_block[i][0], inc_block[i][1], facecolor=p_dict['Incongruent'], alpha=0.35)
    plt.axvspan(con_block[i][0], con_block[i][1], facecolor=p_dict['Congruent'], alpha=0.35)
plt.legend(handles=[inc_patch, con_patch, cort_line_mod_eig, bg_line_eig_lbl, cb_line_eig_lbl], prop={'size':7.5}, loc=1)
plt.tight_layout()
# plt.savefig(f'eigenvec_cen_bg_cb_mod_cort_smooth_sig1_blocks_cc_z_{task}.png', dpi=300)
# plt.show()
plt.close()


# AVERAGE TASK BLOCKS line graph - eigen cb, bg; mod cort - average subjects
q_avg_smooth_z_cort_pad = np.pad(q_avg_smooth_z_cort, (0,10), mode='constant', constant_values=np.nan)
eigen_cen_bg_avg_smooth_z_pad = np.pad(eigen_cen_bg_avg_smooth_z, (0,10), mode='constant', constant_values=np.nan)
eigen_cen_cb_avg_smooth_z_pad = np.pad(eigen_cen_cb_avg_smooth_z, (0,10), mode='constant', constant_values=np.nan)

mod_cort_avg_blocks = np.nanmean(np.array(list(map(lambda i : \
                        q_avg_smooth_z_cort_pad[int(i[0]):int(i[1])+1], group_blocks))), axis=0)
eigen_bg_avg_blocks = np.nanmean(np.array(list(map(lambda i : \
                        eigen_cen_bg_avg_smooth_z_pad[int(i[0]):int(i[1])+1], group_blocks))), axis=0)
eigen_cb_avg_blocks = np.nanmean(np.array(list(map(lambda i : \
                        eigen_cen_cb_avg_smooth_z_pad[int(i[0]):int(i[1])+1], group_blocks))), axis=0)


#### AVERAGE BLOCKS - with error - eigen cb bg, mod cort
fig, ax = plt.subplots()
plt.axhline(y=0, c='k', lw=1.2, alpha=0.28, ls='--', dashes=(5, 7))

ax.plot(np.arange(group_blocks[0][0]*rt, (group_blocks[0][1]+1)*rt, 2), \
        gaussian_filter(mod_cort_avg_blocks, sigma=1), lw=1, c=p_dict['cort_line_cb'], ls='--')
ax.fill_between(np.arange(group_blocks[0][0]*rt, (group_blocks[0][1]+1)*rt, 2), \
            gaussian_filter(mod_cort_avg_blocks, sigma=1)-q_ci_allpts, \
                 gaussian_filter(mod_cort_avg_blocks, sigma=1)+q_ci_allpts, \
                        lw=0, color=p_dict['cort_line_cb'], alpha=0.4)
ax.set_xlabel('Time (s)', size=15, fontname=font_lst[0])
ax.set_ylim(-1.5, 2.25)
ax.set_yticks(np.arange(-1.5, 2.25, 1))

# make eigen y axis label diff colours
ybox1 = TextArea('(z-scored)', textprops=dict(c='k', size=15, fontname=font_lst[0], rotation=90, ha='left', va='bottom'))
ybox2 = TextArea('Centrality', textprops=dict(c=p_dict['bg_line_eig'], size=15, fontname=font_lst[0], rotation=90, ha='left', va='bottom'))
ybox3 = TextArea('Eigenvector', textprops=dict(c=p_dict['cb_line_eig'], size=15, fontname=font_lst[0], rotation=90, ha='left', va='bottom'))

ybox = VPacker(children=[ybox1, ybox2, ybox3], align='left', pad=0, sep=6)
anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.1, -0.045), bbox_transform=ax.transAxes, borderpad=0)
ax.add_artist(anchored_ybox)

# make separate y axis for modularity
ax2 = ax.twinx()
ax2.plot(np.arange(group_blocks[0][0]*rt, (group_blocks[0][1]+1)*rt, 2), \
         gaussian_filter(eigen_bg_avg_blocks, sigma=0.8), lw=1, c=p_dict['bg_line_eig']) # 0.5
ax2.fill_between(np.arange(group_blocks[0][0]*rt, (group_blocks[0][1]+1)*rt, 2), \
            gaussian_filter(eigen_bg_avg_blocks, sigma=1)-eigen_bg_ci_allpts, \
                 gaussian_filter(eigen_bg_avg_blocks, sigma=1)+eigen_bg_ci_allpts, \
                        lw=0, color=p_dict['bg_line_eig'], alpha=0.4)

ax2.plot(np.arange(group_blocks[0][0]*rt, (group_blocks[0][1]+1)*rt, 2), \
         gaussian_filter(eigen_cb_avg_blocks, sigma=0.8), lw=1, c=p_dict['cb_line_eig']) # 0.5
ax2.fill_between(np.arange(group_blocks[0][0]*rt, (group_blocks[0][1]+1)*rt, 2), \
            gaussian_filter(eigen_cb_avg_blocks, sigma=1)-eigen_cb_ci_allpts, \
                 gaussian_filter(eigen_cb_avg_blocks, sigma=1)+eigen_cb_ci_allpts, \
                        lw=0, color=p_dict['cb_line_eig'], alpha=0.4)

ax2.set_ylabel('Modularity (z-scored)', size=15, fontname=font_lst[0])
ax2.set_ylim(-1.5, 2.25)
ax2.set_yticks(np.arange(-1.5, 2.25, 1))

# colour plot background with breakpoints
plt.axvspan(inc_block_frames[0][0]*rt, inc_block_frames[0][1]*rt, facecolor=p_dict['Incongruent_cb'], alpha=0.15)
plt.axvspan(con_block_frames[0][0]*rt, con_block_frames[0][1]*rt, facecolor=p_dict['Congruent_cb'], alpha=0.16)
# plt.text(18, 2.1, 'Incongruent', size=20, fontname=font_lst[0], c=p_dict['Incongruent_cb'])
# plt.text(91, 2.1, 'Congruent', size=20, fontname=font_lst[0], c=p_dict['Congruent_cb'])

plt.legend(handles=[inc_patch_cb, con_patch_cb, cort_line_mod_eig_less, bg_line_eig, cb_line_eig], prop={'size':7.5}, loc=1)
plt.title(task.upper(), size=28, fontname=font_lst[0], fontweight='bold')
plt.tight_layout()
plt.savefig(f'{pars[1]}/output/{task}/eigenvec_cen_bg_cb_mod_cort_smooth_sig2_avg_blocks_cb_z_yerr_ci_{task}.png', dpi=2000)
plt.show()


## CROSS CORRELATION - ALL SUBJS - HOMEMADE
lags = np.array(cross_corr_vec(eigen_cb_avg_blocks, eigen_bg_avg_blocks, lags=15)[1])
# cross_corr_vec(eigen_cb_avg_blocks, eigen_bg_avg_blocks, lags=15, pval=True)

# get cross correlation for vectors
cross_corr_modxcb = np.array(list(map(lambda sub: cross_corr_vec(sub[0], sub[1], lags=15)[0], \
                        zip(q_allsub_avg_blocks_smooth_z, eigen_cb_allsub_avg_blocks_smooth_z))))
cross_corr_modxcb_std = np.array(list(map(lambda i: np.std(i), cross_corr_modxcb.T)))
cross_corr_modxcb_sem = np.array(list(map(lambda i: st.sem(i), cross_corr_modxcb.T)))
cross_corr_modxcb_ci = cross_corr_modxcb_sem*ci
cross_corr_modxcb_avg = np.mean(cross_corr_modxcb, axis=0)

cross_corr_modxbg = np.array(list(map(lambda sub: cross_corr_vec(sub[0], sub[1], lags=15)[0], \
                        zip(q_allsub_avg_blocks_smooth_z, eigen_bg_allsub_avg_blocks_smooth_z))))
cross_corr_modxbg_std = np.array(list(map(lambda i: np.std(i), cross_corr_modxbg.T)))
cross_corr_modxbg_sem = np.array(list(map(lambda i: st.sem(i), cross_corr_modxbg.T)))
cross_corr_modxbg_ci = cross_corr_modxbg_sem*ci
cross_corr_modxbg_avg = np.mean(cross_corr_modxbg, axis=0)

cross_corr_cbxbg = np.array(list(map(lambda sub: cross_corr_vec(sub[0], sub[1], lags=15)[0], \
                        zip(eigen_cb_allsub_avg_blocks_smooth_z, eigen_bg_allsub_avg_blocks_smooth_z))))
cross_corr_cbxbg_std = np.array(list(map(lambda i: np.std(i), cross_corr_cbxbg.T)))
cross_corr_cbxbg_sem = np.array(list(map(lambda i: st.sem(i), cross_corr_cbxbg.T)))
cross_corr_cbxbg_ci = cross_corr_cbxbg_sem*ci
cross_corr_cbxbg_avg = np.mean(cross_corr_cbxbg, axis=0)

"""
# get pvals for each vector
cross_corr_modxcb = np.array(list(map(lambda sub: cross_corr_vec(sub[0], sub[1], lags=15, pval=True), \
                        zip(q_allsub_avg_blocks_smooth_z, eigen_cb_allsub_avg_blocks_smooth_z))))
cross_corr_modxcb_std = np.array(list(map(lambda i: np.std(i), cross_corr_modxcb.T)))
cross_corr_modxcb_sem = np.array(list(map(lambda i: st.sem(i), cross_corr_modxcb.T)))
cross_corr_modxcb_ci = cross_corr_modxcb_sem*ci
cross_corr_modxcb_avg = np.mean(cross_corr_modxcb, axis=0)

cross_corr_modxbg = np.array(list(map(lambda sub: cross_corr_vec(sub[0], sub[1], lags=15, pval=True), \
                        zip(q_allsub_avg_blocks_smooth_z, eigen_bg_allsub_avg_blocks_smooth_z))))
cross_corr_modxbg_std = np.array(list(map(lambda i: np.std(i), cross_corr_modxbg.T)))
cross_corr_modxbg_sem = np.array(list(map(lambda i: st.sem(i), cross_corr_modxbg.T)))
cross_corr_modxbg_ci = cross_corr_modxbg_sem*ci
cross_corr_modxbg_avg = np.mean(cross_corr_modxbg, axis=0)

cross_corr_cbxbg = np.array(list(map(lambda sub: cross_corr_vec(sub[0], sub[1], lags=15, pval=True), \
                        zip(eigen_cb_allsub_avg_blocks_smooth_z, eigen_bg_allsub_avg_blocks_smooth_z))))
cross_corr_cbxbg_std = np.array(list(map(lambda i: np.std(i), cross_corr_cbxbg.T)))
cross_corr_cbxbg_sem = np.array(list(map(lambda i: st.sem(i), cross_corr_cbxbg.T)))
cross_corr_cbxbg_ci = cross_corr_cbxbg_sem*ci
cross_corr_cbxbg_avg = np.mean(cross_corr_cbxbg, axis=0)
"""

#### swap bg and cb

## plot correlations in subplots
fig, ax = plt.subplots(3,1, figsize=(7,7), sharex=True)
fig.suptitle(task.upper(), size=28, fontname=font_lst[0], fontweight='bold', x=0.575)

# mod x bg
ax[0].axhline(y=0, c='k', lw=1.2, alpha=0.28, ls='--', dashes=(11, 21))
ax[0].plot(lags*rt, cross_corr_modxbg_avg, lw=1.7, c=p_dict['var_corr_cort_bg'])
ax[0].fill_between(lags*rt, cross_corr_modxbg_avg-cross_corr_modxbg_ci, \
    cross_corr_modxbg_avg+cross_corr_modxbg_ci, lw=0, color=p_dict['var_corr_cort_bg'], alpha=0.3)
ax[0].set_title('Bg \u2192 Cortex', size=16, fontname=font_lst[0], weight='bold')
ax[0].set_ylim(-0.1, 0.1)
ax[0].set_yticks(np.arange(-0.1, 0.15, 0.1))

# mod x cb
ax[1].axhline(y=0, c='k', lw=1.2, alpha=0.28, ls='--', dashes=(11, 21))
ax[1].plot(lags*rt, cross_corr_modxcb_avg, lw=1.7, c=p_dict['var_corr_cort_cb'])
ax[1].fill_between(lags*rt, cross_corr_modxcb_avg-cross_corr_modxcb_ci, \
    cross_corr_modxcb_avg+cross_corr_modxcb_ci, lw=0, color=p_dict['var_corr_cort_cb'], alpha=0.3)
ax[1].set_title('Cb \u2192 Cortex', size=16, fontname=font_lst[0], weight='bold')
ax[1].set_ylim(-0.1, 0.25)
ax[1].set_yticks(np.arange(-0.1, 0.2, 0.1))

# cb x bg
ax[2].axhline(y=0, c='k', lw=1.2, alpha=0.28, ls='--', dashes=(11, 21))
ax[2].plot(lags*rt, cross_corr_cbxbg_avg, lw=1.7, c=p_dict['var_corr_cb_bg'])
ax[2].fill_between(lags*rt, cross_corr_cbxbg_avg-cross_corr_cbxbg_ci, \
    cross_corr_cbxbg_avg+cross_corr_cbxbg_ci, lw=0, color=p_dict['var_corr_cb_bg'], alpha=0.3)
ax[2].set_title('Bg \u2192 Cb', size=16, fontname=font_lst[0], weight='bold')
ax[2].set_ylim(-0.15, 0.1)
ax[2].set_yticks(np.arange(-0.15, 0.1, 0.1))

ax[0].text(-40.69, 0.13, 'A', size=30, fontname=font_lst[0])
ax[1].text(-40.69, 0.25, 'B', size=30, fontname=font_lst[0])
ax[2].text(-40.69, 0.1, 'C', size=30, fontname=font_lst[0])

plt.xlabel('Lags', size=16, fontname=font_lst[0])
fig.supylabel('Correlation', size=16, fontname=font_lst[0])
# ax[0].legend(handles=[var_corr_cort_cb, var_corr_cort_bg, var_corr_cb_bg], prop={'size':9.2}, bbox_to_anchor=(1.009, 1.03), loc=1) #
plt.tight_layout()
plt.savefig(f'{pars[1]}/output/{task}/cross_corr_eigenvec_cen_bg_cb_mod_cort_subplots_{task}.png', dpi=2000)
plt.show()


#### Reliability test prep
# cort, bg
cort_bg_diff_lag4 = np.array(list(map(lambda tup: tup[1][4:] - tup[0][:-4], \
                        zip(q_allsub_avg_blocks_smooth_z, eigen_bg_allsub_avg_blocks_smooth_z))))
np.save(f'{main_dir}IntermediateData/{task}/cross_corr_cort_bg_peak_diff_{task}.npy', cort_bg_diff_lag4)


cort_cb_diff_lag0 = np.array(list(map(lambda tup: tup[1] - tup[0], \
                        zip(q_allsub_avg_blocks_smooth_z, eigen_cb_allsub_avg_blocks_smooth_z))))
np.save(f'{main_dir}IntermediateData/{task}/cross_corr_cort_cb_peak_diff_{task}.npy', cort_cb_diff_lag0)


cb_bg_diff_lag0 = np.array(list(map(lambda tup: tup[1] - tup[0], \
                        zip(eigen_cb_allsub_avg_blocks_smooth_z, eigen_bg_allsub_avg_blocks_smooth_z))))
np.save(f'{main_dir}IntermediateData/{task}/cross_corr_cb_bg_peak_diff_{task}.npy', cb_bg_diff_lag0)


sys.exit()

#### GRANGER CAUSALITY - transfer entropy
# bottom up, subcortical to cortical -- difference correction - for stationarity
# gc_cort_cb = grangercausalitytests(list(zip(np.diff(mod_cort_avg_blocks)[1:], np.diff(eigen_cb_avg_blocks)[1:])), 8)
# gc_cort_bg = grangercausalitytests(list(zip(np.diff(mod_cort_avg_blocks)[1:], np.diff(eigen_bg_avg_blocks)[1:])), 8)
# gc_cb_bg = grangercausalitytests(list(zip(np.diff(eigen_cb_avg_blocks)[1:], np.diff(eigen_bg_avg_blocks)[1:])), 8)
gc_cort_cb = grangercausalitytests(list(zip(mod_cort_avg_blocks, eigen_cb_avg_blocks)), 8)
gc_cort_bg = grangercausalitytests(list(zip(mod_cort_avg_blocks, eigen_bg_avg_blocks)), 8)
gc_cb_bg = grangercausalitytests(list(zip(eigen_cb_avg_blocks, eigen_bg_avg_blocks)), 8)

# get statistic
gc_fstat_cort_cb = [i[0] for i in get_gc_stat(gc_cort_cb, 'ssr_ftest')]
gc_fstat_cort_bg = [i[0] for i in get_gc_stat(gc_cort_bg, 'ssr_ftest')]
gc_fstat_cb_bg = [i[0] for i in get_gc_stat(gc_cb_bg, 'ssr_ftest')]

# reverse - top down, cortical to subcortical
# gc_cort_cb_rev = grangercausalitytests(list(zip(np.diff(eigen_cb_avg_blocks)[1:], np.diff(mod_cort_avg_blocks)[1:])), 8)
# gc_cort_bg_rev = grangercausalitytests(list(zip(np.diff(eigen_bg_avg_blocks)[1:], np.diff(mod_cort_avg_blocks)[1:])), 8)
# gc_cb_bg_rev = grangercausalitytests(list(zip(np.diff(eigen_bg_avg_blocks)[1:], np.diff(eigen_cb_avg_blocks)[1:])), 8)
gc_cort_cb_rev = grangercausalitytests(list(zip(eigen_cb_avg_blocks, mod_cort_avg_blocks)), 8)
gc_cort_bg_rev = grangercausalitytests(list(zip(eigen_bg_avg_blocks, mod_cort_avg_blocks)), 8)
gc_cb_bg_rev = grangercausalitytests(list(zip(eigen_bg_avg_blocks, eigen_cb_avg_blocks)), 8)

# get statistic
gc_fstat_cort_cb_rev = [i[0] for i in get_gc_stat(gc_cort_cb_rev, 'ssr_ftest')]
gc_fstat_cort_bg_rev = [i[0] for i in get_gc_stat(gc_cort_bg_rev, 'ssr_ftest')]
gc_fstat_cb_bg_rev = [i[0] for i in get_gc_stat(gc_cb_bg_rev, 'ssr_ftest')]

## PLOT
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
fig.suptitle(task.upper(), size=20, fontname=font_lst[0], fontweight='bold', x=0.53) # , x=0.575

# plot bottom up
ax1.plot(np.arange(1,len(gc_fstat_cort_cb)+1)*rt, gc_fstat_cort_cb, label='Cerebellum \u2192 Cortex', c=p_dict['var_corr_cort_cb'])
ax1.plot(np.arange(1,len(gc_fstat_cort_cb)+1)*rt, gc_fstat_cort_bg, label='Basal ganglia \u2192 Cortex', c=p_dict['var_corr_cort_bg'])
ax1.plot(np.arange(1,len(gc_fstat_cort_cb)+1)*rt, gc_fstat_cb_bg, label='Basal ganglia \u2192 Cerebellum', c=p_dict['var_corr_cb_bg'])
ax1.set_ylabel('F-statistic', size=11, fontname=font_lst[0])
ax1.set_xlabel('Lags', size=11, fontname=font_lst[0])
ax1.legend(prop={'size': 7.5}, loc=1)

# plot reverse - top down
ax2.plot(np.arange(1,len(gc_fstat_cort_cb)+1)*rt, gc_fstat_cort_cb_rev, label='Cerebellum \u2190 Cortex', c=p_dict['var_corr_cort_cb'])
ax2.plot(np.arange(1,len(gc_fstat_cort_cb)+1)*rt, gc_fstat_cort_bg_rev, label='Basal ganglia \u2190 Cortex', c=p_dict['var_corr_cort_bg'])
ax2.plot(np.arange(1,len(gc_fstat_cort_cb)+1)*rt, gc_fstat_cb_bg_rev, label='Basal ganglia \u2190 Cerebellum', c=p_dict['var_corr_cb_bg'])
ax2.set_xlabel('Lags', size=11, fontname=font_lst[0])
ax2.legend(prop={'size': 7.5}, loc=1)

# fig.supxlabel('Lags', size=11, fontname=font_lst[0])
plt.tight_layout()
# plt.savefig(f'{pars[1]}/output/{task}/gc_stat_eigenvec_cen_bg_cb_mod_cort_subplots_{task}.png', dpi=2000)
# plt.show()
plt.close()

"""
from pyinform import transfer_entropy

te_cort_cb = transfer_entropy(abs(mod_cort_avg_blocks), abs(eigen_cb_avg_blocks), k=1)
te_cort_bg = transfer_entropy(abs(mod_cort_avg_blocks), abs(eigen_bg_avg_blocks), k=1)
te_cb_bg = transfer_entropy(abs(eigen_cb_avg_blocks), abs(eigen_bg_avg_blocks), k=1)

print(te_cort_cb, te_cort_bg, te_cb_bg)

# gc_cort_cb_rev = grangercausalitytests(list(zip(eigen_cb_avg_blocks, mod_cort_avg_blocks)), 8)
# gc_cort_bg_rev = grangercausalitytests(list(zip(eigen_bg_avg_blocks, mod_cort_avg_blocks)), 8)
# gc_cb_bg_rev = grangercausalitytests(list(zip(eigen_bg_avg_blocks, eigen_cb_avg_blocks)), 8)
"""


#### VECTOR AUTOREGRESSION - mod cortex, eigen bg, eigen cb
# get var lags for each subject, do impyutation for sample statistics, save
"""
coeff_lags_lst = []
corr_lags_lst = []
std_lags_lst = []
stderr_lags_lst = []
for i in range(1,36):
    print(i)
    var_allsub_lag = var_allsub(q_allsub_avg_blocks_smooth_z, eigen_cb_allsub_avg_blocks_smooth_z, \
                             eigen_bg_allsub_avg_blocks_smooth_z, lag=i)
    lag_corr_std = np.std(var_allsub_lag[-1], axis=0)
    lag_corr_sem = st.sem(var_allsub_lag[-1], axis=0)

    impy_vars = impy_params(var_allsub_lag)

    coeff_lags_lst.append(impy_vars[0])
    corr_lags_lst.append(impy_vars[1])
    std_lags_lst.append(lag_corr_std)
    stderr_lags_lst.append(lag_corr_sem)
np.save(f'{main_dir}IntermediateData/{task}/var_coeff_lags35_{task}.npy', coeff_lags_lst)
np.save(f'{main_dir}IntermediateData/{task}/var_corr_lags35_{task}.npy', corr_lags_lst)
np.save(f'{main_dir}IntermediateData/{task}/var_corr_std_lags35_{task}.npy', std_lags_lst)
np.save(f'{main_dir}IntermediateData/{task}/var_corr_sem_lags35_{task}.npy', stderr_lags_lst)
"""

var_coeff_lags35 = np.load(f'{main_dir}IntermediateData/{task}/var_coeff_lags35_{task}.npy')
var_corr_lags35 = np.load(f'{main_dir}IntermediateData/{task}/var_corr_lags35_{task}.npy')
var_corr_std_lags35 = np.load(f'{main_dir}IntermediateData/{task}/var_corr_std_lags35_{task}.npy')
var_corr_sem_lags35 = np.load(f'{main_dir}IntermediateData/{task}/var_corr_sem_lags35_{task}.npy')


# PLOT lag35 VAR coefficients and correlation - subplots - inset, error bars
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

# plot VAR beta coefficients - eigen cb and eigen bg
ax1.plot(list(range(1, len(var_corr_lags35.T[1])+1)), var_coeff_lags35.T[1], c=p_dict['cb_line_eig_var_b'], lw=1)
ax1.plot(list(range(1, len(var_corr_lags35.T[2])+1)), var_coeff_lags35.T[2], c=p_dict['bg_line_eig_var_b'], lw=1)
ax1.get_xaxis().set_visible(True)
ax1.set_title(task.upper(), size=20, fontname=font_lst[0], fontweight='bold')

# make eigen y axis label diff colours
ybox1 = TextArea('(\u03B2)', textprops=dict(c='k', size=11, fontname=font_lst[0], \
                                            rotation=90, ha='left', va='top'))
ybox2 = TextArea('Centrality', textprops=dict(c=p_dict['bg_line_eig_var_b'], \
                            size=11, fontname=font_lst[0], rotation=90, va='center'))
ybox3 = TextArea('Eigenvector', textprops=dict(c=p_dict['cb_line_eig_var_b'], \
                    size=11, fontname=font_lst[0], rotation=90, ha='left', va='bottom'))

ybox = VPacker(children=[ybox2, ybox3], align='left', pad=0, sep=35)
anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.139, 0.1),
                                  bbox_transform=ax1.transAxes, borderpad=0)
ax1.add_artist(anchored_ybox)

ybox = VPacker(children=[ybox1], align='left', pad=0, sep=25)
anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.139, 0.86),
                                  bbox_transform=ax1.transAxes, borderpad=0)
ax1.add_artist(anchored_ybox)

# make separate y axis for modularity
ax1_mod = ax1.twinx()
ax1_mod.set_ylabel('Modularity (β)', size=11, fontname=font_lst[0])
ax1_mod.set_ylim(-0.65, 0.85)
ax1.plot(list(range(1, len(var_corr_lags35.T[0])+1)), var_coeff_lags35.T[0], c=p_dict['cort_line_cb'], lw=1, zorder=0)

ax1.legend(handles=[cort_line_mod_less, cb_line_eig_var_b, bg_line_eig_var_b], prop={'size': 8}, loc=4)


# plot correlation
# mod x cb
ax2.plot(list(range(1, len(var_corr_lags35.T[0])+1)), var_corr_lags35.T[0], label='mod x cb', c=p_dict['var_corr_cort_cb'], lw=1.5)  # sky blue = #17a8fa
ax2.fill_between(list(range(1, len(var_corr_lags35.T[0])+1)), var_corr_lags35.T[0]-var_corr_sem_lags35.T[0], \
    var_corr_lags35.T[0]+var_corr_sem_lags35.T[0], lw=0, color=p_dict['var_corr_cort_cb'], alpha=0.3)

# mod x bg
ax2.plot(list(range(1, len(var_corr_lags35.T[1])+1)), var_corr_lags35.T[1], label='mod x bg', c=p_dict['var_corr_cort_bg'], lw=1.5)  # berry pink = #ea00a5 - salmon bright pink = #ff007f
ax2.fill_between(list(range(1, len(var_corr_lags35.T[1])+1)), var_corr_lags35.T[1]-var_corr_sem_lags35.T[1], \
    var_corr_lags35.T[1]+var_corr_sem_lags35.T[1], lw=0, color=p_dict['var_corr_cort_bg'], alpha=0.3)

# cb x bg
ax2.plot(list(range(1, len(var_corr_lags35.T[2])+1)), var_corr_lags35.T[2], label='cb x bg', c=p_dict['var_corr_cb_bg'], lw=1.5)  # lime green less bright = #03c90a
ax2.fill_between(list(range(1, len(var_corr_lags35.T[2])+1)), var_corr_lags35.T[2]-var_corr_sem_lags35.T[2], \
    var_corr_lags35.T[2]+var_corr_sem_lags35.T[2], lw=0, color=p_dict['var_corr_cb_bg'], alpha=0.3)

ax2.set_ylabel('Correlation', size=11, fontname=font_lst[0])
ax2.set_xlabel('Lags', size=12, fontname=font_lst[0])
ax2.legend(handles=[var_corr_cort_cb, var_corr_cort_bg, var_corr_cb_bg], prop={'size': 7.5}, \
           loc=1,bbox_to_anchor=(1, 0.7))

ax1.text(-6.8, 0.8, 'A', size=30, fontname=font_lst[0])
ax2.text(-6.8, 0.125, 'B', size=30, fontname=font_lst[0])

# plot inset
left, bottom, width, height = [0.688, 0.737, 0.2, 0.2] # x, y
ax1 = fig.add_axes([left, bottom, width, height])
ax1.plot(list(range(1, len(var_corr_lags35.T[1])+1)), var_coeff_lags35.T[1], c=p_dict['cb_line_eig_var_b'], lw=1.25)
ax1.plot(list(range(1, len(var_corr_lags35.T[2])+1)), var_coeff_lags35.T[2], c=p_dict['bg_line_eig_var_b'], lw=1.25)
ax1.plot(list(range(1, len(var_corr_lags35.T[0])+1)), var_coeff_lags35.T[0], c=p_dict['cort_line_cb'], lw=1.25, zorder=0)
ax1.set_xlim(15, 20)
ax1.set_ylim(-0.15, 0.09)
ax1.set_xticks(np.arange(16, 20, 2))
ax1.set_yticks(np.arange(-0.15, 0.1, 0.1))
ax1.xaxis.set_tick_params(labelsize=8)
ax1.yaxis.set_tick_params(labelsize=8)

plt.tight_layout()
# plt.savefig(f'{pars[1]}/output/{task}/var_coeffs_corr_eigenvec_cen_bg_cb_mod_cort_subplots_yerr_inset_{task}.png', dpi=2000)
# plt.show()
plt.close()


#### ALL SUBJS RSS HEATMAP
## RSS - make all subjects rss lst for tasks, save to npy
# rss_lst = jr.load_rss_task(opj(main_dir, d_path), task, subj_lst)
# np.save(f'{main_dir}IntermediateData/{task}/subjs_rss_{task}.npy', rss_lst)

# load saved npy with task rss
rss_lst = np.load(f'{main_dir}IntermediateData/{task}/subjs_rss_{task}.npy')


fig  = plt.figure(figsize=(12, 7))
gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 5])

ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])


## plot task hrf
ax0.plot(con_regressor, label='Congruent', c=p_dict['Congruent_cb'], lw=3)
ax0.plot(inc_regressor, label='Incongruent', c=p_dict['Incongruent_cb'], lw=3)
ax0.set_yticklabels([])
ax0.set_xticklabels([])
ax0.tick_params(size=0)
ax0.set_xlim([0,280])
ax0.legend(loc=4, prop={'size': 11})

im3 = ax1.pcolormesh(np.array(rss_lst), cmap='RdBu_r', \
            norm=colors.TwoSlopeNorm(vmin=50, vcenter=200, vmax=350)) # cmap= 'RdBu_r', plt.cm.RdYlBu_r

cbar = fig.colorbar(im3, ax=ax1, orientation='vertical', fraction=0.02, pad=0.0001)
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_ylabel('RSS', size=15, fontname=font_lst[0], rotation=90)

ax1.set_xticks(np.arange(30, 280, 30))
ax1.set_xticklabels((np.arange(30, 280, 30)*2.0).astype(int), size=13)
ax1.set_yticklabels([])
ax1.set_xlabel('Time (s)', size=22, fontname=font_lst[0])
ax1.set_ylabel('Subjects', size=22, fontname=font_lst[0])
ax1.set_xlim([0,280])
ax1.tick_params(labelsize=18)
plt.suptitle(task.upper(), size=33, fontname=font_lst[0], fontweight='bold')

plt.tight_layout()
plt.savefig(f'{pars[1]}/output/{task}/allsubs_hrf_rss_ax2_cbar_{task}.png', dpi=2000)
plt.show()
# plt.close()
