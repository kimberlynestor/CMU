"""

"""

from config import *
import jr_funcs as jr
from bg_cb_funcs import *
from config import *
from task_blocks import *

import os
import sys
from operator import itemgetter

import numpy as np
import pandas as pd
from scipy import stats

import networkx as nx
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter


# tr = 200
task = 'stroop'
out_name = 'eigen_cen_'


"""
eigen_cb_lst = []
eigen_bg_lst = []
eigen_thal_lst = []
for subj in subj_lst:
    efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)

    evec_cen = list(map(lambda mat: nx.eigenvector_centrality(\
                nx.from_numpy_matrix(mat), weight='weight', max_iter=1000), efc_mat))

    targets_cb = target_nodes(efc_mat, node_info('cb')[0])
    targets_bg = target_nodes(efc_mat, node_info('bg')[0])
    targets_thal = target_nodes(efc_mat, node_info('thal')[0])

    # eigen_cb = list(map(lambda tr: list(map(lambda i : tr[i], targets_cb)), evec_cen))
    eigen_cb = list(map(lambda i: list(map(lambda ii : i[1][ii], i[0])) , zip(targets_cb, evec_cen)))
    eigen_bg = list(map(lambda i: list(map(lambda ii : i[1][ii], i[0])) , zip(targets_bg, evec_cen)))
    eigen_thal = list(map(lambda i: list(map(lambda ii : i[1][ii], i[0])) , zip(targets_thal, evec_cen)))

    avg_tr_cb = np.average(eigen_cb, axis=1)
    avg_tr_bg = np.average(eigen_bg, axis=1)
    avg_tr_thal = np.average(eigen_thal, axis=1)

    eigen_cb_lst.append(avg_tr_cb)
    eigen_bg_lst.append(avg_tr_bg)
    eigen_thal_lst.append(avg_tr_thal)

    # bg = gaussian_filter(avg_tr_bg, sigma=1)
    # cb = gaussian_filter(avg_tr_cb, sigma=1)

    # save each subject output to dir
    try:
        # os.mkdir(f'{out_name}cb')
        # os.mkdir(f'{out_name}bg')
        os.mkdir(f'{out_name}thal')
    except:
        pass
    # np.save(f'{out_name}cb/{out_name}cb_{int(subj)}.npy', avg_tr_cb)
    # np.save(f'{out_name}bg/{out_name}bg_{int(subj)}.npy', avg_tr_bg)
    np.save(f'{out_name}thal/{out_name}thal_{int(subj)}.npy', avg_tr_thal)

    # plt.plot(avg_tr_bg, label='eigen_bg', c='tab:blue')
    # plt.plot(avg_tr_cb, label='eigen_cb', c='tab:red')
    # plt.plot(bg, label='eigen_bg', c='tab:blue')
    # plt.plot(cb, label='eigen_cb', c='tab:red')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

# np.save(f'eigen_cen_cb_allsub_{task}.npy', eigen_cb_lst)
# np.save(f'eigen_cen_bg_allsub_{task}.npy', eigen_bg_lst)
np.save(f'eigen_cen_thal_allsub_{task}.npy', eigen_thal_lst)
"""


eigen_cen_cb_allsub = np.load(f'eigen_cen_cb_allsub_stroop.npy')
eigen_cen_bg_allsub = np.load(f'eigen_cen_bg_allsub_stroop.npy')
# eigen_cen_thal_allsub = np.load(f'eigen_cen_thal_allsub_stroop.npy')

eigen_cen_cb_allsub_smooth = np.array(list(map(lambda i: gaussian_filter(i, sigma=1), eigen_cen_cb_allsub)))
eigen_cen_cb_allsub_smooth_z = np.array(list(map(lambda i: stats.zscore(i), eigen_cen_cb_allsub_smooth)))
eigen_cen_bg_allsub_smooth = np.array(list(map(lambda i: gaussian_filter(i, sigma=1), eigen_cen_bg_allsub)))
eigen_cen_bg_allsub_smooth_z = np.array(list(map(lambda i: stats.zscore(i), eigen_cen_bg_allsub_smooth)))

eigen_cen_cb_avg = np.average(eigen_cen_cb_allsub, axis=0)
eigen_cen_bg_avg = np.average(eigen_cen_bg_allsub, axis=0)

eigen_cen_cb_avg_smooth = gaussian_filter(eigen_cen_cb_avg, sigma=1)
eigen_cen_bg_avg_smooth = gaussian_filter(eigen_cen_bg_avg, sigma=1)

eigen_cen_cb_avg_smooth_z = stats.zscore(eigen_cen_cb_avg_smooth)
eigen_cen_bg_avg_smooth_z = stats.zscore(eigen_cen_bg_avg_smooth)
np.save(f'eigen_cen_cb_avg_smooth_z_stroop.npy', eigen_cen_cb_avg_smooth_z)
np.save(f'eigen_cen_bg_avg_smooth_z_stroop.npy', eigen_cen_bg_avg_smooth_z)

print(f'\nEigenvector centrality corr_coef, BG and CB: ', \
            np.corrcoef(eigen_cen_cb_avg_smooth_z, eigen_cen_bg_avg_smooth_z)[0][1], '\n') # [1][0]


# FULL TS, CB-CTX
df_mlm_ts = pd.DataFrame(eigen_cen_cb_allsub_smooth_z)
df_mlm_ts.columns +=1 # start column count from 1
df_mlm_ts.insert(0, 'subj_ID', [int(i) for i in subj_lst])
df_mlm_ts_melt = pd.melt(df_mlm_ts, id_vars=['subj_ID'], \
                                 var_name='frame', value_name='eigen_cb')
df_mlm_ts_melt.to_csv('cb_eigenvec_cen_ts.csv', index=False)

## average task blocks
eigen_cen_cb_allsub_smooth_z_pad = np.array(list(map(lambda sub:np.pad(sub, (0,10), \
                                mode='constant', constant_values=np.nan), eigen_cen_cb_allsub_smooth_z)))
eigen_cb_avg_blocks_allsub = np.array(list(map(lambda sub: np.nanmean(np.array(list(\
                                map(lambda i : sub[int(i[0]):int(i[1])+1], group_blocks))), \
                                    axis=0), eigen_cen_cb_allsub_smooth_z_pad)))

df_mlm_ts = pd.DataFrame(eigen_cb_avg_blocks_allsub)
df_mlm_ts.columns +=5 # start column count from 1
df_mlm_ts.insert(0, 'subj_ID', [int(i) for i in subj_lst])
df_mlm_ts_melt = pd.melt(df_mlm_ts, id_vars=['subj_ID'], \
                                 var_name='frame', value_name='eigen_cb')
df_mlm_ts_melt.to_csv('cb_eigenvec_cen_ts_avg_blks.csv', index=False)

# FULL TS, BG-CTX
df_mlm_ts = pd.DataFrame(eigen_cen_bg_allsub_smooth_z)
df_mlm_ts.columns +=1 # start column count from 1
df_mlm_ts.insert(0, 'subj_ID', [int(i) for i in subj_lst])
df_mlm_ts_melt = pd.melt(df_mlm_ts, id_vars=['subj_ID'], \
                                 var_name='frame', value_name='eigen_bg')
df_mlm_ts_melt.to_csv('bg_eigenvec_cen_ts.csv', index=False)

## average task blocks
eigen_cen_bg_allsub_smooth_z_pad = np.array(list(map(lambda sub:np.pad(sub, (0,10), \
                                mode='constant', constant_values=np.nan), eigen_cen_bg_allsub_smooth_z)))
eigen_bg_avg_blocks_allsub = np.array(list(map(lambda sub: np.nanmean(np.array(list(\
                                map(lambda i : sub[int(i[0]):int(i[1])+1], group_blocks))), \
                                    axis=0), eigen_cen_bg_allsub_smooth_z_pad)))
"""
for bg, cb in zip(eigen_bg_avg_blocks_allsub, eigen_cb_avg_blocks_allsub):
    plt.plot(bg, label='eigen_bg', c='tab:blue')
    plt.plot(cb, label='eigen_cb', c='tab:orange')
    plt.axvline(x=35, c='k', alpha=0.5)
    plt.axvline(x=40, c='k', alpha=0.5)
    plt.legend()
    plt.show()
"""

df_mlm_ts = pd.DataFrame(eigen_bg_avg_blocks_allsub)
df_mlm_ts.columns +=5 # start column count from 1
df_mlm_ts.insert(0, 'subj_ID', [int(i) for i in subj_lst])
df_mlm_ts_melt = pd.melt(df_mlm_ts, id_vars=['subj_ID'], \
                                 var_name='frame', value_name='eigen_bg')
df_mlm_ts_melt.to_csv('bg_eigenvec_cen_ts_avg_blks.csv', index=False)



## plot avg subj bg and cb eigenvector connectivity
plt.rcParams['figure.figsize'] = (8,4)

plt.plot(np.arange(0, frs*rt, 2), gaussian_filter(eigen_cen_bg_avg_smooth_z, sigma=1), lw=0.8, c=p_dict['bg_line_eig']) #8A9A5B #3c0008
plt.plot(np.arange(0, frs*rt, 2), gaussian_filter(eigen_cen_cb_avg_smooth_z, sigma=1), lw=0.8, c=p_dict['cb_line_eig']) # #BC544B, #A91B0D
plt.xticks(np.arange(0, frs*rt, 60))
plt.xlabel('Time (s)', size=12, fontname='serif')
plt.ylabel('Eigenvector Centrality (z-score)', size=12, fontname='serif')

# colour plot background with breakpoints
for i in range(len(inc_block)):
    # incongruent
    plt.axvspan(inc_block[i][0], inc_block[i][1], facecolor=p_dict['Incongruent'], alpha=0.35)
    # congruent
    plt.axvspan(con_block[i][0], con_block[i][1], facecolor=p_dict['Congruent'], alpha=0.35)
# plt.ylim(-3.5, 2.25)
plt.legend(handles=[inc_patch, con_patch, bg_line_eig, cb_line_eig], loc=1, prop={'size':7.5})
plt.tight_layout()
# plt.savefig('eigenvec_cen_bg_cb_smooth_sig2_blocks_cc_z.png', dpi=300)
# plt.show()
plt.close()

