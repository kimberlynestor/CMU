"""


"""

import sys

import jr_funcs as jr
from bg_cb_funcs import *
from task_blocks import *
from config import *

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import scipy.stats as st
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

import seaborn as sns
from matplotlib import gridspec
import matplotlib.colors as colors

import itertools

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


print(f'\ncorr_coef bg x cb eigen: {np.corrcoef(eigen_cen_cb_avg_smooth_z, eigen_cen_bg_avg_smooth_z)[0][1]}')
print(f'corr_coef cort connec x cb eigen: {np.corrcoef(con_cort_smooth_z, eigen_cen_cb_avg_smooth_z)[0][1]}')
print(f'corr_coef cort connec x bg eigen: {np.corrcoef(con_cort_smooth_z, eigen_cen_bg_avg_smooth_z)[0][1]}\n')

print(f'\ncorr_coef cort mod x cb eigen: {np.corrcoef(q_avg_smooth_z_cort, eigen_cen_cb_avg_smooth_z)[0][1]}')
print(f'corr_coef cort mod x bg eigen: {np.corrcoef(q_avg_smooth_z_cort, eigen_cen_bg_avg_smooth_z)[0][1]}\n')


#### AVERAGE BLOCKS
q_allsub = np.load(f'subjs_all_net_cort_q_stroop.npy')
q_allsub_pad = np.array(list(map(lambda sub:np.pad(sub, (0,10), mode='constant', \
                                                constant_values=np.nan), q_allsub)))
q_allsub_avg_blocks = np.array(list(map(lambda sub:np.nanmean(np.array(list(map(lambda i : \
                            sub[int(i[0]):int(i[1])+1], group_blocks))), axis=0), q_allsub_pad)))
q_sterr_allpts = list(map(lambda i: st.sem(i), q_allsub_avg_blocks.T))

eigen_cb_allsub = np.load(f'eigen_cen_cb_allsub_stroop.npy')
eigen_cb_allsub_pad = np.array(list(map(lambda sub:np.pad(sub, (0,10), mode='constant', \
                                                constant_values=np.nan), eigen_cb_allsub)))
eigen_cb_allsub_avg_blocks = np.array(list(map(lambda sub:np.nanmean(np.array(list(map(lambda i : \
                            sub[int(i[0]):int(i[1])+1], group_blocks))), axis=0), eigen_cb_allsub_pad)))
eigen_cb_allsub_avg_blocks_smooth = np.array(list(map(lambda i: gaussian_filter(i, sigma=1), eigen_cb_allsub_avg_blocks)))
eigen_cb_allsub_avg_blocks_smooth_z = np.array(list(map(lambda i: stats.zscore(i), eigen_cb_allsub_avg_blocks_smooth)))
eigen_cb_sterr_allpts = list(map(lambda i: st.sem(i), eigen_cb_allsub_avg_blocks_smooth_z.T))
eigen_cb_std_allpts = list(map(lambda i: np.std(i), eigen_cb_allsub_avg_blocks_smooth_z.T))

eigen_bg_allsub = np.load(f'eigen_cen_bg_allsub_stroop.npy')
eigen_bg_allsub_pad = np.array(list(map(lambda sub:np.pad(sub, (0,10), mode='constant', \
                                                constant_values=np.nan), eigen_bg_allsub)))
eigen_bg_allsub_avg_blocks = np.array(list(map(lambda sub:np.nanmean(np.array(list(map(lambda i : \
                            sub[int(i[0]):int(i[1])+1], group_blocks))), axis=0), eigen_bg_allsub_pad)))
eigen_bg_allsub_avg_blocks_smooth = np.array(list(map(lambda i: gaussian_filter(i, sigma=1), eigen_bg_allsub_avg_blocks)))
eigen_bg_allsub_avg_blocks_smooth_z = np.array(list(map(lambda i: stats.zscore(i), eigen_bg_allsub_avg_blocks_smooth)))
eigen_bg_sterr_allpts = list(map(lambda i: st.sem(i), eigen_bg_allsub_avg_blocks_smooth_z.T))
eigen_bg_std_allpts = list(map(lambda i: np.std(i), eigen_bg_allsub_avg_blocks_smooth_z.T))



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
## plot eigenvec cen from cb, bg -- with connectivity of cort
plt.rcParams['figure.figsize'] = (8,4)

plt.plot(np.arange(0, frs*rt, 2), con_cort_smooth_z, lw=1.5, c=p_dict['cort_line'], ls=':')
plt.plot(np.arange(0, frs*rt, 2), eigen_cen_bg_avg_smooth_z, lw=0.8, c=p_dict['bg_line_eig'])
plt.plot(np.arange(0, frs*rt, 2), eigen_cen_cb_avg_smooth_z, lw=0.8, c=p_dict['cb_line_eig'], alpha=0.8)

plt.xticks(np.arange(0, frs*rt, 60))
# plt.ylim(-3, 4.5)
plt.xlabel('Time (s)', size=12, fontname='serif')
plt.ylabel('Z-scored values', size=12, fontname='serif')

# colour plot background with breakpoints
for i in range(len(inc_block)):
    plt.axvspan(inc_block[i][0], inc_block[i][1], facecolor=p_dict['Incongruent'], alpha=0.35)
    plt.axvspan(con_block[i][0], con_block[i][1], facecolor=p_dict['Congruent'], alpha=0.35)
plt.legend(handles=[inc_patch, con_patch, cort_line_connec, bg_line_eig_lbl, cb_line_eig_lbl], prop={'size':7.5}, loc=1)
plt.tight_layout()
plt.savefig('eigenvec_cen_bg_cb_connec_cort_smooth_sig1_blocks_cc_z.png', dpi=300)
plt.show()

## plot eigenvec cen from cb, bg -- with modularity of cort
plt.rcParams['figure.figsize'] = (8,4)

plt.plot(np.arange(0, frs*rt, 2), q_avg_smooth_z_cort, lw=1.5, c=p_dict['cort_line'], ls=':')
plt.plot(np.arange(0, frs*rt, 2), eigen_cen_bg_avg_smooth_z, lw=1, c=p_dict['bg_line_eig'])
plt.plot(np.arange(0, frs*rt, 2), eigen_cen_cb_avg_smooth_z, lw=1, c=p_dict['cb_line_eig'])

plt.xticks(np.arange(0, frs*rt, 60))
plt.ylim(-3.75, 5.75)
plt.xlabel('Time (s)', size=12, fontname='serif')
plt.ylabel('Z-scored values', size=12, fontname='serif')

# colour plot background with breakpoints
for i in range(len(inc_block)):
    plt.axvspan(inc_block[i][0], inc_block[i][1], facecolor=p_dict['Incongruent'], alpha=0.35)
    plt.axvspan(con_block[i][0], con_block[i][1], facecolor=p_dict['Congruent'], alpha=0.35)
plt.legend(handles=[inc_patch, con_patch, cort_line_mod_eig, bg_line_eig_lbl, cb_line_eig_lbl], prop={'size':7.5}, loc=1)
plt.tight_layout()
plt.savefig('eigenvec_cen_bg_cb_mod_cort_smooth_sig1_blocks_cc_z.png', dpi=300)
plt.show()


# AVERAGE TASK BLOCKS line graph - eigen cb, bg; mod cort
q_avg_smooth_z_cort_pad = np.pad(q_avg_smooth_z_cort, (0,10), mode='constant', constant_values=np.nan)
eigen_cen_bg_avg_smooth_z_pad = np.pad(eigen_cen_bg_avg_smooth_z, (0,10), mode='constant', constant_values=np.nan)
eigen_cen_cb_avg_smooth_z_pad = np.pad(eigen_cen_cb_avg_smooth_z, (0,10), mode='constant', constant_values=np.nan)

mod_cort_avg_blocks = np.nanmean(np.array(list(map(lambda i : q_avg_smooth_z_cort_pad[int(i[0]):int(i[1])+1], group_blocks))), axis=0)
eigen_bg_avg_blocks = np.nanmean(np.array(list(map(lambda i : eigen_cen_bg_avg_smooth_z_pad[int(i[0]):int(i[1])+1], group_blocks))), axis=0)
eigen_cb_avg_blocks = np.nanmean(np.array(list(map(lambda i : eigen_cen_cb_avg_smooth_z_pad[int(i[0]):int(i[1])+1], group_blocks))), axis=0)

## plotting
fig, ax = plt.subplots()
plt.axhline(y=0, c='k', lw=1.2, alpha=0.28, ls='--', dashes=(5, 7))

ax.plot(np.arange(group_blocks[0][0], group_blocks[0][1]+1), gaussian_filter(mod_cort_avg_blocks, sigma=1), lw=1.5, c=p_dict['cort_line_cb'], ls=':')
ax.set_xlabel('Frames (TR)', size=12, fontname='serif')
ax.set_ylim(-2, 2.75)
ax.set_yticks(np.arange(-1.5, 2, 1))

# make eigen y axis label diff colours
ybox1 = TextArea('(z-scored)', textprops=dict(c='k', size=11, fontname='serif', rotation=90, ha='left', va='top'))
ybox2 = TextArea('Centrality', textprops=dict(c=p_dict['bg_line_eig'], size=11, fontname='serif', rotation=90, va='center'))
ybox3 = TextArea('Eigenvector', textprops=dict(c=p_dict['cb_line_eig'], size=11, fontname='serif', rotation=90, ha='left', va='bottom'))

ybox = VPacker(children=[ybox1, ybox2, ybox3], align='left', pad=0, sep=32)
anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.1, 0.04), bbox_transform=ax.transAxes, borderpad=0)
ax.add_artist(anchored_ybox)

# make separate y axis for modularity
ax2 = ax.twinx()
ax2.plot(np.arange(group_blocks[0][0], group_blocks[0][1]+1), gaussian_filter(eigen_bg_avg_blocks, sigma=1), lw=1.25, c=p_dict['bg_line_eig'])
ax2.plot(np.arange(group_blocks[0][0], group_blocks[0][1]+1), gaussian_filter(eigen_cb_avg_blocks, sigma=1), lw=1.25, c=p_dict['cb_line_eig'])
ax2.set_ylabel('Modularity (Q/z)', size=12, fontname='serif')
ax2.set_ylim(-2, 2.75)
ax2.set_yticks(np.arange(-1.5, 2, 1))

# colour plot background with breakpoints
plt.axvspan(inc_block_frames[0][0], inc_block_frames[0][1], facecolor=p_dict['Incongruent_cb'], alpha=0.15)
plt.axvspan(con_block_frames[0][0], con_block_frames[0][1], facecolor=p_dict['Congruent_cb'], alpha=0.16)

plt.legend(handles=[inc_patch_cb, con_patch_cb, cort_line_mod_eig_less, bg_line_eig, cb_line_eig], prop={'size':6.5}, loc=1)
plt.tight_layout()
plt.savefig('eigenvec_cen_bg_cb_mod_cort_smooth_sig2_avg_blocks_cb_z.png', dpi=300)
plt.show()


#### AVERAGE BLOCKS - with error - eigen cb bg, mod cort
fig, ax = plt.subplots()
plt.axhline(y=0, c='k', lw=1.2, alpha=0.28, ls='--', dashes=(5, 7))

ax.plot(np.arange(group_blocks[0][0], group_blocks[0][1]+1), gaussian_filter(mod_cort_avg_blocks, sigma=1), lw=1.5, c=p_dict['cort_line_cb'], ls=':')
ax.set_xlabel('Frames (TR)', size=12, fontname='serif')
# ax.set_ylim(-2, 2.75) #sem
ax.set_ylim(-3, 4.5)
# ax.set_yticks(np.arange(-1.5, 2, 1)) #sem
ax.set_yticks(np.arange(-2.5, 4, 1))


# make eigen y axis label diff colours
ybox1 = TextArea('(z-scored)', textprops=dict(c='k', size=11, fontname='serif', rotation=90, ha='left', va='top'))
ybox2 = TextArea('Centrality', textprops=dict(c=p_dict['bg_line_eig'], size=11, fontname='serif', rotation=90, va='center'))
ybox3 = TextArea('Eigenvector', textprops=dict(c=p_dict['cb_line_eig'], size=11, fontname='serif', rotation=90, ha='left', va='bottom'))

ybox = VPacker(children=[ybox1, ybox2, ybox3], align='left', pad=0, sep=32)
anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.1, 0.04), bbox_transform=ax.transAxes, borderpad=0)
ax.add_artist(anchored_ybox)

# make separate y axis for modularity
ax2 = ax.twinx()
ax2.plot(np.arange(group_blocks[0][0], group_blocks[0][1]+1), gaussian_filter(eigen_bg_avg_blocks, sigma=0.8), lw=1, c=p_dict['bg_line_eig']) # 0.5
ax2.fill_between(np.arange(group_blocks[0][0], group_blocks[0][1]+1), \
            gaussian_filter(eigen_bg_avg_blocks, sigma=1)-eigen_bg_std_allpts, \
                 gaussian_filter(eigen_bg_avg_blocks, sigma=1)+eigen_bg_std_allpts, \
                        lw=0, color=p_dict['bg_line_eig'], alpha=0.4)

ax2.plot(np.arange(group_blocks[0][0], group_blocks[0][1]+1), gaussian_filter(eigen_cb_avg_blocks, sigma=0.8), lw=1, c=p_dict['cb_line_eig']) # 0.5
ax2.fill_between(np.arange(group_blocks[0][0], group_blocks[0][1]+1), \
            gaussian_filter(eigen_cb_avg_blocks, sigma=1)-eigen_cb_std_allpts, \
                 gaussian_filter(eigen_cb_avg_blocks, sigma=1)+eigen_cb_std_allpts, \
                        lw=0, color=p_dict['cb_line_eig'], alpha=0.4)

ax2.set_ylabel('Modularity (Q/z)', size=12, fontname='serif')
# ax2.set_ylim(-2, 2.75) #sem
ax2.set_ylim(-3, 4.5)
# ax2.set_yticks(np.arange(-1.5, 2, 1)) #sem
ax2.set_yticks(np.arange(-2.5, 4, 1))

# colour plot background with breakpoints
plt.axvspan(inc_block_frames[0][0], inc_block_frames[0][1], facecolor=p_dict['Incongruent_cb'], alpha=0.15)
plt.axvspan(con_block_frames[0][0], con_block_frames[0][1], facecolor=p_dict['Congruent_cb'], alpha=0.16)

plt.legend(handles=[inc_patch_cb, con_patch_cb, cort_line_mod_eig_less, bg_line_eig, cb_line_eig], prop={'size':6.5}, loc=1)
plt.tight_layout()
plt.savefig('eigenvec_cen_bg_cb_mod_cort_smooth_sig2_avg_blocks_cb_z_yerr_std.png', dpi=2000)
plt.show()

sys.exit()


## plot eigenvec cen from cb, bg -- with modularity of cort -- MORE SMOOTH
plt.rcParams['figure.figsize'] = (8,4)
plt.axhline(y=0, c='k', lw=1.2, alpha=0.28, ls='--', dashes=(5, 7))
plt.plot(np.arange(0, frs*rt, 2), gaussian_filter(q_avg_smooth_z_cort, sigma=1), lw=1.5, c=p_dict['cort_line'], ls=':') # tot smooth sigma=2
plt.plot(np.arange(0, frs*rt, 2), gaussian_filter(eigen_cen_bg_avg_smooth_z, sigma=1), lw=1, c=p_dict['bg_line_eig']) #
plt.plot(np.arange(0, frs*rt, 2), gaussian_filter(eigen_cen_cb_avg_smooth_z, sigma=1), lw=1, c=p_dict['cb_line_eig']) #

plt.xticks(np.arange(0, frs*rt, 60))
plt.ylim(-3.75, 5)
plt.yticks(np.arange(-3, 5, 2))
plt.xlabel('Time (s)', size=12, fontname='serif')
plt.ylabel('Z-scored values', size=12, fontname='serif')

# colour plot background with breakpoints
for i in range(len(inc_block)):
    plt.axvspan(inc_block[i][0], inc_block[i][1], facecolor=p_dict['Incongruent'], alpha=0.35)
    plt.axvspan(con_block[i][0], con_block[i][1], facecolor=p_dict['Congruent'], alpha=0.35)
plt.legend(handles=[inc_patch, con_patch, cort_line_mod_eig, bg_line_eig_lbl, cb_line_eig_lbl], prop={'size':7.5}, loc=1)
plt.tight_layout()
plt.savefig('eigenvec_cen_bg_cb_mod_cort_smooth_sig2_blocks_cc_z.png', dpi=300)
plt.show()

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
df_connec_cort_ts = pd.read_csv('cort_connec_ts.csv')

df_eigen_cb_ts = pd.read_csv('cb_eigenvec_cen_ts.csv')
df_eigen_bg_ts = pd.read_csv('bg_eigenvec_cen_ts.csv')

df_mlm_ts = pd.concat([df_mod_cort_ts, df_connec_cort_ts['connec_cort'], \
                df_connec_cb_ts['connec_cb'], df_connec_bg_ts['connec_bg'], \
                    df_connec_thal_ts['connec_thal'], df_eigen_cb_ts['eigen_cb'], \
                       df_eigen_bg_ts['eigen_bg']], axis=1)



conds_task = [(df_mlm_ts['frame'].isin(list(itertools.chain.from_iterable(inc_frames)))), \
            (df_mlm_ts['frame'].isin(list(itertools.chain.from_iterable(con_frames)))), \
            (df_mlm_ts['frame'].isin(list(itertools.chain.from_iterable(fix_frames))))]
vals_task = ['Incongruent', 'Congruent', 'Fixation']
df_mlm_ts['task'] = np.select(conds_task, vals_task)
df_mlm_ts = df_mlm_ts[df_mlm_ts['task'] != 'Fixation']

conds_block = [(df_mlm_ts['frame'].isin(inc_frames[0]) | df_mlm_ts['frame'].isin(con_frames[0])), \
                (df_mlm_ts['frame'].isin(inc_frames[1]) | df_mlm_ts['frame'].isin(con_frames[1])), \
               (df_mlm_ts['frame'].isin(inc_frames[2]) | df_mlm_ts['frame'].isin(con_frames[2])), \
               (df_mlm_ts['frame'].isin(inc_frames[3]) | df_mlm_ts['frame'].isin(con_frames[3]))]
vals_block = list(range(1,5))
df_mlm_ts['block'] = np.select(conds_block, vals_block)

df_mlm_ts.to_csv('cort_mod_cortbgcb_connec_eigen_ts.csv', index=False)
# print(df_mlm_ts)

# create lag and lead df
df_mlm_ts_shift = df_mlm_ts
for i in list(range(-6,6)):
    if i > 0:
        df_mlm_ts_shift[f'mod_idx_lag{i}'] = df_mlm_ts_shift['mod_idx'].shift(i)
    elif i < 0:
        df_mlm_ts_shift[f'mod_idx_lead{np.abs(i)}'] = df_mlm_ts_shift['mod_idx'].shift(i)
df_mlm_ts_shift.to_csv('cort_mod_cortbgcb_connec_eigen_ts_shift.csv', index=False)

# print(df_mlm_ts_shift)
# sys.exit()

# examine lag shift for connectivity and modularity
from statsmodels.tsa.ar_model import AutoReg
ar_model = AutoReg(df_mlm_ts['mod_idx'], exog=df_mlm_ts[['connec_thal', 'frame', 'inc_reg', 'con_reg']], lags=5).fit()
print(ar_model.summary())

ar_model = AutoReg(df_mlm_ts['mod_idx'], exog=df_mlm_ts[['connec_thal']], lags=1).fit()
print(ar_model.summary())

ar_model = AutoReg(df_mlm_ts['mod_idx'], exog=df_mlm_ts[['eigen_cb', 'eigen_bg', 'frame', 'inc_reg', 'con_reg', 'block']], lags=5).fit()
print(ar_model.summary())


# lag info from r
lag5_cb_bg = [[-0.006779570, -0.005703655], [0.00702539, -0.01966812], \
                [0.003240875, -0.004539574], [0.004489676, -0.007512988], \
                [-0.003102133, -0.009673152]]
lag5_cb_bg = np.array(lag5_cb_bg).T

lead5_cb_bg = [[-7.366847e-05, -4.943186e-03], [0.004655710, 0.004730023], \
                [0.003622741, -0.004213582], [-0.0099380125, 0.0007713246], \
                [-0.0009879371, -0.0018849386]]
lead5_cb_bg = np.array(lead5_cb_bg).T

fig, ax = plt.subplots(2, 1, figsize=(12,5))
ax[0].plot(lag5_cb_bg[0], label='cb_lag')
ax[0].plot(lag5_cb_bg[1], label='bg_lag')
ax[0].set_xlim(0,5)
ax[1].plot(lead5_cb_bg[0], label='cb_lag')
ax[1].plot(lead5_cb_bg[1], label='bg_lag')
ax[1].set_xlim(0,5)
ax[1].set_ylabel('lead coefficients')
ax[0].set_ylabel('lag coefficients')
ax[0].legend()
ax[1].legend()
plt.show()

# save matrix at each tr for one subject
efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), 'stroop', subj_lst[0])
for i in range(len(efc_mat)+1):
    # plot inc corr matrix
    fig, ax = plt.subplots(1, 1)
    coll = ax.imshow(efc_mat[i], vmin=-1, vmax=1, cmap='viridis')  # cmap=plt.cm.RdYlBu_r
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    try:
        os.mkdir('tr_mats')
    except:
        pass
    # plt.savefig(f'tr_mats/tr_mat_{str(i).zfill(3)}.png', dpi=300)
    plt.close()