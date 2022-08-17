"""


"""

from os.path import join as opj
from csv import writer

import numpy as np
import pandas as pd

import jr_funcs as jr
import bct

# np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)


# data path and subjects
main_dir = '/home/kimberlynestor/gitrepo/int_seg/data/'
d_path = 'pip_edge_ts/shen/'
dep_path = 'depend/'

p_dict = {'Incongruent':'#c6b5ff', 'Congruent':'#a5cae7', 'Fixation':'#ffbcd9', \
          'Difference':'#8A2D1C', 'Incongruent Fixation':'#c6b5ff', \
          'Congruent Fixation':'#a5cae7', 'cort_line': '#717577', \
          'cb_line': 'tab:orange', 'bg_line': 'tab:green', 'thal_line': 'tab:red'}
# 'cb_line': '#6e7f80'
# 'Incongruent':'tab:orange', 'Congruent':'tab:blue', # 'Fixation':'#e1c1de'
# cc_yellow: #f9ffb8, cc_green: #b8ffbe, cc_salmon: #ffcbcb, burnt_orange: #cc5500, #964000

def node_info(region):
    """This function takes as input the desired region ('cort', 'cb', 'bg', 'thal'),
    returns the shen indices and the network each node is assigned to."""
    # shen parcels region assignment - brain areas from Andrew Gerlach
    df_net_assign = pd.read_csv(opj(main_dir, dep_path, 'shen_268_parcellation_networklabels_mod.csv'))
    ci = df_net_assign.iloc[:,1].values

    if region == 'cort':
        # cortical nodes only
        assign = np.array(list(filter(lambda i: i[1] != 4, df_net_assign.values)))
        reg_idx = [i[0] for i in assign]
        ci_reg = ci[reg_idx]
        return(list(reg_idx), list(ci_reg))
    elif region == 'cb':
        # cb nodes only
        df_net_assign_reg = df_net_assign[(df_net_assign['anat_reg'].str.contains('Cerebellum')) \
                                         | (df_net_assign['anat_others'].str.contains('Cerebellum')) \
                                         | (df_net_assign['anat_covered'].str.contains('Cerebellum'))]
        reg_idx = df_net_assign_reg.iloc[:, 0].values
        ci_reg = ci[reg_idx]
        return(list(reg_idx), list(ci_reg))
    elif region =='bg':
        # bg nodes only
        # df_net_assign_exc = df_net_assign.drop(df_net_assign.\
        #                         iloc[list(itertools.chain(c_idx, cb_idx))].index)
        df_net_assign_reg = df_net_assign[(df_net_assign['anat_reg'].str.contains('Caudate')) \
                                         | (df_net_assign['anat_reg'].str.contains('Putamen')) \
                                         | (df_net_assign['anat_reg'].str.contains('SN')) \
                                         | (df_net_assign['anat_reg'].str.contains('N_Acc'))]  # no pallidum ?
        reg_idx = df_net_assign_reg.iloc[:, 0].values
        ci_reg = ci[reg_idx]
        return(list(reg_idx), list(ci_reg))
    elif region == 'thal':
        # thal nodes only
        df_net_assign_reg = df_net_assign[(df_net_assign['anat_reg'].str.contains('Thal'))]
        reg_idx = df_net_assign_reg.iloc[:, 0].values
        ci_reg = ci[reg_idx]
        return(list(reg_idx), list(ci_reg))

# print(node_info('thal'))
# node_info('bg')


def save_mod_idx(subj_lst, task, region):
    """This function takes as input the subject list and the task
    ('all', 'stroop', 'msit', 'rest') and saves ouput files with modularity at
    all timepoints for all subjects. Can also specify region ()."""
    if region == 'cort':
        out_name = 'subjs_all_net_cort_q_'
        c_idx = node_info('cort')[0]
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
    elif region == 'cb':
        out_name = 'subjs_all_net_cb_q_'
        c_idx = node_info('cb')[0]
        if task == 'all':
            # get efc mat for all subjects, cortical networks only
            for task in ['stroop', 'msit', 'rest']:
                # mod_file = open(f'subjs_mod_mat_{task}.csv', 'w')
                q_file = open(f'{out_name}{task}.csv', 'w')
                # loop_start = process_time()
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
    elif region == 'bg':
        out_name = 'subjs_all_net_bg_q_'
        c_idx = node_info('bg')[0]
        if task == 'all':
            # get efc mat for all subjects, cortical networks only
            for task in ['stroop', 'msit', 'rest']:
                # mod_file = open(f'subjs_mod_mat_{task}.csv', 'w')
                q_file = open(f'{out_name}{task}.csv', 'w')
                # loop_start = process_time()
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



