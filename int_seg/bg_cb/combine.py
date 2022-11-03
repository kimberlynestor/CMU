"""


"""

import sys

import jr_funcs as jr
from bg_cb_funcs import *
from task_blocks import *
from config import *

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import matplotlib.colors as colors

from nilearn.glm.first_level import compute_regressor

q_avg_smooth_z_cort = np.load('q_avg_smooth_z_allpts_cort_stroop.npy')
q_avg_smooth_z_mask_cort = np.ma.array(np.insert(np.load('q_avg_smooth_z_cort_stroop.npy'), 0, \
                                np.ones(5)), mask=np.pad(np.ones(5), (0,frs-5)))

con_cort_smooth_z = np.load(f'con_cort_smooth_z_stroop.npy')
con_cb_smooth_z = np.load('con_cb_smooth_z_cort_stroop.npy')
con_bg_smooth_z = np.load('con_bg_smooth_z_cort_stroop.npy')
connec_cb_bg_avg_smooth_z = np.average((con_cb_smooth_z, con_bg_smooth_z), axis=0)

eigen_cen_cb_avg_smooth_z = np.load('eigen_cen_cb_avg_smooth_z_stroop.npy')
eigen_cen_bg_avg_smooth_z = np.load('eigen_cen_bg_avg_smooth_z_stroop.npy')


# ALL IN ONE PLOT
## plot connectivity from cb, bg -- with modularity of cort

plt.plot(np.arange(0, frs*rt, 2), q_avg_smooth_z_mask_cort, lw=1, c=p_dict['cort_line'], label='mod_cort_line')
plt.plot(np.arange(0, frs*rt, 2), con_cb_smooth_z, lw=1, c=p_dict['cb_line'], label='connec_cb_line', ls='--')
plt.plot(np.arange(0, frs*rt, 2), con_bg_smooth_z, lw=1, c=p_dict['bg_line'], label='connec_bg_line', ls='--')

plt.xticks(np.arange(0, frs*rt, 60))
plt.xlabel('Time (s)', size=15, fontname='serif')
plt.ylabel('Z-scored values', size=15, fontname='serif')

# colour plot background with breakpoints
for i in range(len(inc_block)):
    plt.axvspan(inc_block[i][0], inc_block[i][1], facecolor=p_dict['Incongruent'], alpha=0.35)
    plt.axvspan(con_block[i][0], con_block[i][1], facecolor=p_dict['Congruent'], alpha=0.35)
plt.legend(handles=[inc_patch, con_patch, cort_line_mod, bg_line_connec, cb_line_connec], prop={'size':7.5}, loc=1)
plt.tight_layout()
plt.savefig('subjs_all_net_cort/allsub_cort_mod_cb_bg_con_smooth_sig1_blocks_cc_abs.png', dpi=300)
plt.show()


# ALL IN ONE PLOT
## plot eigenvec cen from cb, bg -- with modularity of cort
plt.rcParams['figure.figsize'] = (8,4)

plt.plot(np.arange(0, frs*rt, 2), con_cort_smooth_z, lw=1.5, c=p_dict['cort_line'], ls=':')
plt.plot(np.arange(0, frs*rt, 2), eigen_cen_bg_avg_smooth_z, lw=0.8, c=p_dict['bg_line_eig'])
plt.plot(np.arange(0, frs*rt, 2), eigen_cen_cb_avg_smooth_z, lw=0.8, c=p_dict['cb_line_eig'], alpha=0.8)

plt.xticks(np.arange(0, frs*rt, 60))
# plt.ylim(-3, 4.5)
plt.xlabel('Time (s)', size=15, fontname='serif')
plt.ylabel('Z-scored values', size=15, fontname='serif')

# colour plot background with breakpoints
for i in range(len(inc_block)):
    plt.axvspan(inc_block[i][0], inc_block[i][1], facecolor=p_dict['Incongruent'], alpha=0.35)
    plt.axvspan(con_block[i][0], con_block[i][1], facecolor=p_dict['Congruent'], alpha=0.35)
plt.legend(handles=[inc_patch, con_patch, cort_line_connec, bg_line_eig_lbl, cb_line_eig_lbl], prop={'size':7.5}, loc=1)
plt.tight_layout()
plt.savefig('eigenvec_cen_bg_cb_connec_cort_smooth_sig1_blocks_cc_z.png', dpi=300)
plt.show()

sys.exit()

# ALL IN ONE PLOT
## plot avg bg, cb connec -- with mod of cort
plt.plot(np.arange(0, frs*rt, 2), q_avg_smooth_z_mask_cort, lw=1, c=p_dict['cort_line'], label='mod_cort_line')
plt.plot(np.arange(0, frs*rt, 2), con_cb_smooth_z, lw=1, c='tab:brown', label='connec_cb_bg_line', ls='--')

plt.xticks(np.arange(0, frs*rt, 60))
plt.xlabel('Time (s)', size=15, fontname='serif')
plt.ylabel('Z-scored values', size=15, fontname='serif')

# colour plot background with breakpoints
for i in range(len(inc_block)):
    plt.axvspan(inc_block[i][0], inc_block[i][1], facecolor=p_dict['Incongruent'], alpha=0.35)
    plt.axvspan(con_block[i][0], con_block[i][1], facecolor=p_dict['Congruent'], alpha=0.35)
plt.legend(handles=[inc_patch, con_patch, cort_line_mod, cb_bg_line_connec], prop={'size':7.5}, loc=1)
plt.tight_layout()
plt.savefig('subjs_all_net_cort/allsub_cort_mod_avg_cb_bg_con_smooth_sig1_blocks_cc_abs.png', dpi=300)
plt.show()





## POINTPLOT - mod cort, connec cb and bg
df_q_cort_sep_blocks = pd.read_csv('df_q_cort_sep_blocks_stroop.csv')
df_q_cort_sep_blocks = df_q_cort_sep_blocks[(df_q_cort_sep_blocks['task']=='Incongruent') | \
                                (df_q_cort_sep_blocks['task']=='Congruent')]
df_q_cort_sep_blocks['region'] = 'Cort'

df_connec_bg_sep_blocks = pd.read_csv('df_connec_bg_sep_blocks_stroop.csv')
df_connec_bg_sep_blocks['block'] = df_connec_bg_sep_blocks['block'].add(8)
df_connec_bg_sep_blocks['region'] = 'Bg'

df_connec_cb_sep_blocks = pd.read_csv('df_connec_cb_sep_blocks_stroop.csv')
df_connec_cb_sep_blocks['block'] = df_connec_cb_sep_blocks['block'].add(8*2)
df_connec_cb_sep_blocks['region'] = 'Cb'

df_mod_connec_pp = pd.DataFrame(itertools.chain(df_q_cort_sep_blocks.values, \
                    df_connec_bg_sep_blocks.values, df_connec_cb_sep_blocks.values), \
                        columns=['vals', 'block', 'task', 'region'])
# df_mod_connec_pp['task_region'] = df_mod_connec_pp[['task', 'region']].agg(' '.join, axis=1)
# df_mod_connec_pp = df_mod_connec_pp.drop(['task', 'region'], axis=1)


# from collections import Counter
# print( *Counter(df_mod_connec_pp['task_region'].values) )
# print( list(Counter(df_mod_connec_pp['task_region'].values).keys()) )


sns.pointplot(x='block', y='vals', hue='task', data=df_mod_connec_pp, \
              join=False, palette=p_dict, order=list(sum(zip(range(1,9), \
                                        range(9,17), range(17,25)), ())) )
# plt.xticks(np.arange(0,24), labels=list(sum(zip(range(1,9), ['']*8, ['']*8), ())))
plt.xticks(np.arange(0,24), labels=list(sum(zip(range(1,9), ['bg']*8, ['cb']*8), ())) )
plt.xlabel('Task block', size=14, fontname='serif')
plt.ylabel('Z-scored values', size=14, fontname='serif')
plt.legend((inc_circ, con_circ), ('Incongruent', 'Congruent', 'Fixation'), numpoints=1)
plt.tight_layout()
plt.savefig('subjs_all_net_cort/allsub_cortnet_cort_mod_avg_cb_bg_smooth_sig1_pointplot_ci_cc.png', dpi=300)
plt.show()

# maybe: coolwarm, twilight, twilight_shifted, BuPu, PuBu, Purples, YlGnBu, binary, cool, gist_gray, gist_yarg, pink, BuPu_r, PuOr_r, Purples_r, gist_gray_r, hot_r, pink_r, vlag_r
# for i in list(plt.cm.cmap_d.keys()):

#### ALL SUBJS RSS HEATMAP
# load saved npy with task rss
rss_stroop_lst = np.load('subjs_rss_stroop.npy')
# rss_msit_lst = np.load('subjs_rss_msit.npy')
# rss_rest_lst = np.load('subjs_rss_rest.npy')

fig  = plt.figure(figsize=(12, 7))
gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 5])

ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])


## plot task hrf
ax0.plot(con_regressor, label='Congruent', c=p_dict['Congruent'], lw=3)
ax0.plot(inc_regressor, label='Incongruent', c=p_dict['Incongruent'], lw=3)
ax0.set_yticklabels([])
ax0.set_xticklabels([])
ax0.tick_params(size=0)
ax0.set_xlim([0,280])
ax0.legend(loc=4)

im3 = ax1.pcolormesh(np.array(rss_stroop_lst), cmap='RdBu_r', \
                     norm=colors.TwoSlopeNorm(vmin=50, vcenter=200, vmax=350)) # cmap= 'vlag', 'RdBu_r', 'coolwarm', plt.cm.RdYlBu_r

cbar = fig.colorbar(im3, ax=ax1, orientation='horizontal', fraction=0.08, pad=0.22)
cbar.ax.tick_params(labelsize=11)
cbar.ax.set_ylabel('RSS', size=13, fontname='serif', rotation=90)

ax1.set_xticks(np.arange(30, 280, 30))
ax1.set_xticklabels((np.arange(30, 280, 30)*2.0).astype(int), size=13)
ax1.set_yticklabels([])
ax1.set_xlabel('Time (s)', size=20, fontname='serif')
ax1.set_ylabel('Subjects', size=20, fontname='serif')
ax1.set_xlim([0,280])
ax1.tick_params(labelsize=18)
plt.suptitle('Stroop', size=30, fontname='serif')

plt.tight_layout()
plt.savefig('allsubs_hrf_rss_ax2_cbar_stroop.png', dpi=300)
plt.show()



## combine all timescale dataaframes
df_mod_cort_ts = pd.read_csv('cort_mod_idx_reg_ts.csv')
df_connec_cb_ts = pd.read_csv('cb_connec_ts.csv')
df_connec_bg_ts = pd.read_csv('bg_connec_ts.csv')
df_connec_thal_ts = pd.read_csv('thal_connec_ts.csv')

df_mlm_ts = pd.concat([df_mod_cort_ts, df_connec_cb_ts['connec_cb'], \
                df_connec_bg_ts['connec_bg'], df_connec_thal_ts['connec_thal']], axis=1)
df_mlm_ts.to_csv('cort_mod_bgcb_connec_ts.csv', index=False)
# print(df_mlm_ts)


# examine lag shift for connectivity and modularity
from statsmodels.tsa.ar_model import AutoReg
ar_model = AutoReg(df_mlm_ts['mod_idx'], exog=df_mlm_ts[['connec_thal', 'frame', 'inc_reg', 'con_reg']], lags=1).fit()
print(ar_model.summary())

ar_model = AutoReg(df_mlm_ts['mod_idx'], exog=df_mlm_ts[['connec_thal']], lags=1).fit()
print(ar_model.summary())