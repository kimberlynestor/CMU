"""


"""

from os.path import join as opj
from csv import writer

import numpy as np
import pandas as pd

import jr_funcs as jr
import bct

# data path and subjects
main_dir = '/home/kimberlynestor/gitrepo/int_seg/data/'
d_path = 'pip_edge_ts/shen/'
dep_path = 'depend/'


def save_mod_idx(subj_lst, task):
    """This function takes as input the subject list and the task and saves ouput
    files with modularity at all timepoints for all subjects. cortical networks only."""

    out_name = 'subjs_all_net_cort_q_'

    # shen parcels region assignment
    net_assign = pd.read_csv(opj(main_dir, dep_path, \
                    'shen_268_parcellation_networklabels.csv')).values
    cort_assign = np.array(list(filter(lambda i: i[1] != 4, net_assign)))
    c_idx = [i[0] for i in cort_assign]

    if task == 'all':
        # get efc mat for all subjects, cortical networks only
        for task in ['stroop', 'msit', 'rest']:
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
            # save to npy
            np.save(f'{out_name}{task}.npy', q_subj_lst)
        # loop_end = process_time()
        # loop_time = loop_end - loop_start
        # print(f'loop time for execution: {loop_time}')

    else:
        q_file = open(f'{out_name}{task}.csv', 'w')
        q_subj_lst = []
        for subj in subj_lst:
            efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)  # all networks
            efc_mat_cort = np.array(list(map(lambda frame: \
                                np.array(list(map(lambda col: col[c_idx], \
                                    frame[c_idx]))), efc_mat)))  # cortex only

            mod_all_frs = np.array(list(map(lambda mat: bct.community_louvain(mat, \
                                                                              gamma=1.5, B='negative_sym'),
                                            efc_mat_cort)), dtype=object)

            ci_all_frs = np.array(list(map(lambda x: x[0], mod_all_frs)))
            q_all_frs = np.array(list(map(lambda x: x[1], mod_all_frs)))
            q_subj_lst.append(q_all_frs)

            # save to csv
            with open(f'{out_name}{task}.csv', 'a') as f_obj:
                w_obj = writer(f_obj)
                w_obj.writerow(q_all_frs)
                f_obj.close()
        np.save(f'{out_name}{task}.npy', q_subj_lst)
    return


# subj_lst = np.loadtxt(opj(main_dir, dep_path, 'subjects_intersect_motion_035.txt'))
# save_mod_idx(subj_lst, 'all')



