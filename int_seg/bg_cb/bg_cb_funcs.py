"""


"""

from os.path import join as opj
import os

from csv import writer
import itertools

import numpy as np
import pandas as pd

import jr_funcs as jr
import bct

from config import *


# np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)


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



def save_mod_idx(subj_lst, task, region):
    """This function takes as input the subject list and the task
    ('all', 'stroop', 'msit', 'rest') and saves ouput files with modularity at
    all timepoints for all subjects. Can also specify region ('cort', 'cb', 'bg')."""
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
    elif region == 'thal':
        out_name = 'subjs_all_net_thal_q_'
        c_idx = node_info('thal')[0]
        if task == 'all':
            # get efc mat for all subjects, cortical networks only
            for task in ['stroop', 'msit', 'rest']:
                # mod_file = open(f'subjs_mod_mat_{task}.csv', 'w')
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
                # save to npy
                np.save(f'{out_name}{task}.npy', q_subj_lst)
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



subj_lst = np.loadtxt(opj(main_dir, dep_path, 'subjects_intersect_motion_035.txt'))
# save_mod_idx(subj_lst, 'all')


def save_connec(subj_lst, task, region):
    """This function takes as input the subject list and the task
    ('all', 'stroop', 'msit', 'rest') and saves ouput files with matrices of
    only the nodes from the specified region all timepoints for all subjects.
    Region options: ('cort', 'cb', 'bg', 'thal') """
    if region == 'cort':
        out_name = 'subjs_all_net_cort_'
        c_idx = node_info('cort')[0]
        if task == 'all':
            # get efc mat for all subjects
            for task in ['stroop', 'msit', 'rest']:
                con_subj_lst = []
                for subj in subj_lst:
                    efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)
                    efc_mat_lim = np.array(list(map(lambda frame: \
                                    np.array(list(map(lambda col: col[c_idx], \
                                        frame[c_idx]))), efc_mat)))
                    con_subj_lst.append(efc_mat_lim)
                # save to npy
                np.save(f'{out_name}{task}.npy', con_subj_lst)
        else:
            con_subj_lst = []
            for subj in subj_lst:
                efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)
                efc_mat_lim = np.array(list(map(lambda frame: \
                                    np.array(list(map(lambda col: col[c_idx], \
                                        frame[c_idx]))), efc_mat)))
                con_subj_lst.append(efc_mat_lim)
            # save to npy
            np.save(f'{out_name}{task}.npy', con_subj_lst)
        return
    elif region == 'cb':
        out_name = 'subjs_all_net_cb_'
        c_idx = node_info('cb')[0]
        if task == 'all':
            # get efc mat for all subjects
            for task in ['stroop', 'msit', 'rest']:
                con_subj_lst = []
                for subj in subj_lst:
                    efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)
                    efc_mat_lim = np.array(list(map(lambda frame: \
                                    np.array(list(map(lambda col: col[c_idx], \
                                        frame[c_idx]))), efc_mat)))
                    con_subj_lst.append(efc_mat_lim)
                # save to npy
                np.save(f'{out_name}{task}.npy', con_subj_lst)
        else:
            con_subj_lst = []
            for subj in subj_lst:
                efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)
                efc_mat_lim = np.array(list(map(lambda frame: \
                                    np.array(list(map(lambda col: col[c_idx], \
                                        frame[c_idx]))), efc_mat)))
                con_subj_lst.append(efc_mat_lim)
            # save to npy
            np.save(f'{out_name}{task}.npy', con_subj_lst)
        return
    elif region == 'bg':
        out_name = 'subjs_all_net_bg_'
        c_idx = node_info('bg')[0]
        if task == 'all':
            # get efc mat for all subjects
            for task in ['stroop', 'msit', 'rest']:
                con_subj_lst = []
                for subj in subj_lst:
                    efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)
                    efc_mat_lim = np.array(list(map(lambda frame: \
                                    np.array(list(map(lambda col: col[c_idx], \
                                        frame[c_idx]))), efc_mat)))
                    con_subj_lst.append(efc_mat_lim)
                # save to npy
                np.save(f'{out_name}{task}.npy', con_subj_lst)
        else:
            con_subj_lst = []
            for subj in subj_lst:
                efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)
                efc_mat_lim = np.array(list(map(lambda frame: \
                                    np.array(list(map(lambda col: col[c_idx], \
                                        frame[c_idx]))), efc_mat)))
                con_subj_lst.append(efc_mat_lim)
            # save to npy
            np.save(f'{out_name}{task}.npy', con_subj_lst)
        return
    elif region == 'thal':
        out_name = 'subjs_all_net_thal_'
        c_idx = node_info('thal')[0]
        if task == 'all':
            # get efc mat for all subjects
            for task in ['stroop', 'msit', 'rest']:
                con_subj_lst = []
                for subj in subj_lst:
                    efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)
                    efc_mat_lim = np.array(list(map(lambda frame: \
                                    np.array(list(map(lambda col: col[c_idx], \
                                        frame[c_idx]))), efc_mat)))
                    con_subj_lst.append(efc_mat_lim)
                # save to npy
                np.save(f'{out_name}{task}.npy', con_subj_lst)
        else:
            con_subj_lst = []
            for subj in subj_lst:
                efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)
                efc_mat_lim = np.array(list(map(lambda frame: \
                                    np.array(list(map(lambda col: col[c_idx], \
                                        frame[c_idx]))), efc_mat)))
                con_subj_lst.append(efc_mat_lim)
            # save to npy
            np.save(f'{out_name}{task}.npy', con_subj_lst)
        return


def save_connec_adj_cort(subj_lst, task, region):
    """This function takes as input the subject list and the task
    ('all', 'stroop', 'msit', 'rest') and saves ouput files with matrices of
    only the nodes from the specified region all timepoints for all subjects.
    Only regions with adjacent cortical connections are saved.
    Region options: ('cb', 'bg', 'thal') """
    if region == 'cb':
        out_name = 'subjs_all_net_cb_adj_cort_'
        cort_nodes = node_info('cort')[0]
        nodes = node_info('cb')[0]
        if task == 'all':
            # get efc mat for all subjects
            for task in ['stroop', 'msit', 'rest']:
                con_subj_lst = []
                for subj in subj_lst:
                    efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)
                    adj_mat = list(map(lambda i: np.where(np.abs(i) > 0, 1, 0), efc_mat))
                    adj_mat_reg = list(map(lambda i: i[nodes], adj_mat))

                    cort_bool = list(map(lambda ii: list(map(lambda i: \
                                    any(i[cort_nodes] == 1), ii)), adj_mat_reg))
                    nodes_cort_connec = list(map(lambda i: \
                                list(itertools.compress(nodes, i)), cort_bool))

                    efc_mat_lim = [list(map(lambda col: col[c_idx], \
                                    frame[c_idx])) for frame, c_idx in \
                                        zip(efc_mat, nodes_cort_connec)]
                    efc_mat_lim = np.array(efc_mat_lim, dtype=object)
                    con_subj_lst.append(efc_mat_lim)
                    # save each subj efc_mat_lim to dir
                    try:
                        os.mkdir(out_name[14:-1])
                    except:
                        pass
                    np.save(f'{out_name[14:-1]}/{out_name[14:]}{int(subj)}.npy', efc_mat_lim)
                # save to npy
                # np.save(f'{out_name}{task}.npy', con_subj_lst)
        else:
            con_subj_lst = []
            for subj in subj_lst:
                efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)
                adj_mat = list(map(lambda i: np.where(np.abs(i) > 0, 1, 0), efc_mat))
                adj_mat_reg = list(map(lambda i: i[nodes], adj_mat))

                cort_bool = list(map(lambda ii: list(map(lambda i: \
                                any(i[cort_nodes] == 1), ii)), adj_mat_reg))
                nodes_cort_connec = list(map(lambda i: \
                                list(itertools.compress(nodes, i)), cort_bool))

                efc_mat_lim = [list(map(lambda col: col[c_idx], frame[c_idx])) \
                               for frame, c_idx in zip(efc_mat, nodes_cort_connec)]
                efc_mat_lim = np.array(efc_mat_lim, dtype=object)
                con_subj_lst.append(efc_mat_lim)
                # save each subj efc_mat_lim to dir
                try:
                    os.mkdir(out_name[14:-1])
                except:
                    pass
                np.save(f'{out_name[14:-1]}/{out_name[14:]}{int(subj)}.npy', efc_mat_lim)
            # save to npy
            # np.save(f'{out_name}{task}.npy', con_subj_lst) # tofile() # fromfile()

        return
    elif region == 'bg':
        out_name = 'subjs_all_net_bg_adj_cort_'
        cort_nodes = node_info('cort')[0]
        nodes = node_info('bg')[0]
        if task == 'all':
            # get efc mat for all subjects
            for task in ['stroop', 'msit', 'rest']:
                con_subj_lst = []
                for subj in subj_lst:
                    efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)
                    adj_mat = list(map(lambda i: np.where(np.abs(i) > 0, 1, 0), efc_mat))
                    adj_mat_reg = list(map(lambda i: i[nodes], adj_mat))

                    cort_bool = list(map(lambda ii: list(map(lambda i: \
                                    any(i[cort_nodes] == 1), ii)), adj_mat_reg))
                    nodes_cort_connec = list(map(lambda i: \
                                list(itertools.compress(nodes, i)), cort_bool))

                    efc_mat_lim = [list(map(lambda col: col[c_idx], \
                                    frame[c_idx])) for frame, c_idx in \
                                        zip(efc_mat, nodes_cort_connec)]
                    efc_mat_lim = np.array(efc_mat_lim, dtype=object)
                    con_subj_lst.append(efc_mat_lim)
                    # save each subj efc_mat_lim to dir
                    try:
                        os.mkdir(out_name[14:-1])
                    except:
                        pass
                    np.save(f'{out_name[14:-1]}/{out_name[14:]}{int(subj)}.npy', efc_mat_lim)
                # save to npy
                # np.save(f'{out_name}{task}.npy', con_subj_lst)
        else:
            con_subj_lst = []
            for subj in subj_lst:
                efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)
                adj_mat = list(map(lambda i: np.where(np.abs(i) > 0, 1, 0), efc_mat))
                adj_mat_reg = list(map(lambda i: i[nodes], adj_mat))

                cort_bool = list(map(lambda ii: list(map(lambda i: \
                                any(i[cort_nodes] == 1), ii)), adj_mat_reg))
                nodes_cort_connec = list(map(lambda i: \
                                list(itertools.compress(nodes, i)), cort_bool))

                efc_mat_lim = [list(map(lambda col: col[c_idx], frame[c_idx])) \
                               for frame, c_idx in zip(efc_mat, nodes_cort_connec)]
                efc_mat_lim = np.array(efc_mat_lim, dtype=object)
                con_subj_lst.append(efc_mat_lim)
                # save each subj efc_mat_lim to dir
                try:
                    os.mkdir(out_name[14:-1])
                except:
                    pass
                np.save(f'{out_name[14:-1]}/{out_name[14:]}{int(subj)}.npy', efc_mat_lim)
            # save to npy
            # np.save(f'{out_name}{task}.npy', con_subj_lst)
        return
    elif region == 'thal':
        out_name = 'subjs_all_net_thal_adj_cort_'
        cort_nodes = node_info('cort')[0]
        nodes = node_info('thal')[0]
        if task == 'all':
            # get efc mat for all subjects
            for task in ['stroop', 'msit', 'rest']:
                con_subj_lst = []
                for subj in subj_lst:
                    efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)
                    adj_mat = list(map(lambda i: np.where(np.abs(i) > 0, 1, 0), efc_mat))
                    adj_mat_reg = list(map(lambda i: i[nodes], adj_mat))

                    cort_bool = list(map(lambda ii: list(map(lambda i: \
                                    any(i[cort_nodes] == 1), ii)), adj_mat_reg))
                    nodes_cort_connec = list(map(lambda i: \
                                list(itertools.compress(nodes, i)), cort_bool))

                    efc_mat_lim = [np.array(list(map(lambda col: col[c_idx], \
                                    frame[c_idx]))) for frame, c_idx in \
                                        zip(efc_mat, nodes_cort_connec)]
                    efc_mat_lim = np.array(efc_mat_lim, dtype=object)
                    con_subj_lst.append(efc_mat_lim)
                    # save each subj efc_mat_lim to dir
                    try:
                        os.mkdir(out_name[14:-1])
                    except:
                        pass
                    np.save(f'{out_name[14:-1]}/{out_name[14:]}{int(subj)}.npy', efc_mat_lim)
                # save to npy
                # np.save(f'{out_name}{task}.npy', con_subj_lst)
        else:
            con_subj_lst = []
            for subj in subj_lst:
                efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)
                adj_mat = list(map(lambda i: np.where(np.abs(i) > 0, 1, 0), efc_mat))
                adj_mat_reg = list(map(lambda i: i[nodes], adj_mat))

                cort_bool = list(map(lambda ii: list(map(lambda i: \
                                any(i[cort_nodes] == 1), ii)), adj_mat_reg))
                nodes_cort_connec = list(map(lambda i: \
                                list(itertools.compress(nodes, i)), cort_bool))

                efc_mat_lim = [list(map(lambda col: col[c_idx], frame[c_idx])) \
                               for frame, c_idx in zip(efc_mat, nodes_cort_connec)]
                efc_mat_lim = np.array(efc_mat_lim, dtype=object)
                con_subj_lst.append(efc_mat_lim)
                # save each subj efc_mat_lim to dir
                try:
                    os.mkdir(out_name[14:-1])
                except:
                    pass
                np.save(f'{out_name[14:-1]}/{out_name[14:]}{int(subj)}.npy', efc_mat_lim)
            # save to npy
            # np.save(f'{out_name}{task}.npy', con_subj_lst)
        return


def save_mod_idx_adj_cort(subj_lst, task, region):
    """This function takes as input the subject list and the task
    ('all', 'stroop', 'msit', 'rest') and saves ouput files with modularity of
    only the nodes from the specified region all timepoints for all subjects.
    Only regions with adjacent cortical connections are saved.
    Region options: ('cb', 'bg', 'thal') """
    if region == 'cb':
        out_name = 'subjs_all_net_cb_adj_cort_q_'
        # cort_nodes = node_info('cort')[0]
        nodes = node_info('cb')[0]
        if task == 'all':
            # get efc mat for all subjects
            for task in ['stroop', 'msit', 'rest']:
                con_subj_lst = []
                for subj in subj_lst:
                    efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)
                    efc_mat_lim = mat_adj_cort(efc_mat, nodes, thres=0)
                    q_all_frs = [bct.community_louvain(mat, gamma=1.5, B='negative_sym')[1] for mat in efc_mat_lim]
                    con_subj_lst.append(q_all_frs)
                # save to npy
                np.save(f'{out_name}{task}.npy', con_subj_lst)
        else:
            con_subj_lst = []
            for subj in subj_lst[5:]:
                efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)
                efc_mat_lim = mat_adj_cort(efc_mat, nodes, thres=0)
                q_all_frs = [bct.community_louvain(mat, gamma=1.5, B='negative_sym')[1] for mat in efc_mat_lim]
                con_subj_lst.append(q_all_frs)
            # save to npy
            np.save(f'{out_name}{task}.npy', con_subj_lst)
        return
    elif region == 'bg':
        out_name = 'subjs_all_net_bg_adj_cort_q_'
        cort_nodes = node_info('cort')[0]
        nodes = node_info('bg')[0]
        if task == 'all':
            # get efc mat for all subjects
            for task in ['stroop', 'msit', 'rest']:
                con_subj_lst = []
                for subj in subj_lst:
                    efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)
                    efc_mat_lim = mat_adj_cort(efc_mat, nodes, thres=0)
                    q_all_frs = [bct.community_louvain(mat, gamma=1.5, B='negative_sym')[1] for mat in efc_mat_lim]
                    con_subj_lst.append(q_all_frs)
                # save to npy
                np.save(f'{out_name}{task}.npy', con_subj_lst)
        else:
            con_subj_lst = []
            for subj in subj_lst:
                efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)
                efc_mat_lim = mat_adj_cort(efc_mat, nodes, thres=0)
                q_all_frs = [bct.community_louvain(mat, gamma=1.5, B='negative_sym')[1] for mat in efc_mat_lim]
                con_subj_lst.append(q_all_frs)
                # save to npy
            np.save(f'{out_name}{task}.npy', con_subj_lst)
        return
    elif region == 'thal':
        out_name = 'subjs_all_net_thal_adj_cort_q_'
        cort_nodes = node_info('cort')[0]
        nodes = node_info('thal')[0]
        if task == 'all':
            # get efc mat for all subjects
            for task in ['stroop', 'msit', 'rest']:
                con_subj_lst = []
                for subj in subj_lst:
                    efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)
                    efc_mat_lim = mat_adj_cort(efc_mat, nodes, thres=0)
                    q_all_frs = [bct.community_louvain(mat, gamma=1.5, B='negative_sym')[1] for mat in efc_mat_lim]
                    con_subj_lst.append(q_all_frs)
                    # save to npy
                np.save(f'{out_name}{task}.npy', con_subj_lst)
        else:
            con_subj_lst = []
            for subj in subj_lst:
                efc_mat = jr.get_efc_trans_sing(opj(main_dir, d_path), task, subj)
                efc_mat_lim = mat_adj_cort(efc_mat, nodes, thres=0)
                q_all_frs = [bct.community_louvain(mat, gamma=1.5, B='negative_sym')[1] for mat in efc_mat_lim]
                con_subj_lst.append(q_all_frs)
                # save to npy
            np.save(f'{out_name}{task}.npy', con_subj_lst)
        return

def mat_adj_cort(efc_mat, reg_nodes, thres, cort_nodes=node_info('cort')[0]):
    """Takes a matrix as input and a set of region nodes and returns a
    matrix of that region nodes only with connections to cortex. thres=float between 0,1"""
    adj_mat = list(map(lambda i: np.where(np.abs(i) > thres, 1, 0), efc_mat)) #0.2, 0.6
    adj_mat_reg = list(map(lambda i: i[reg_nodes], adj_mat))

    cort_bool = list(map(lambda ii: list(map(lambda i: any(i[cort_nodes] == 1), ii)), adj_mat_reg))
    nodes_cort_connec = list(map(lambda i: list(itertools.compress(reg_nodes, i)), cort_bool))

    efc_mat_lim = [list(map(lambda col: col[c_idx], frame[c_idx])) \
                   for frame, c_idx in zip(efc_mat, nodes_cort_connec)]
    efc_mat_lim = np.array(list(map(lambda x: np.array(x), efc_mat_lim)))
    return(efc_mat_lim)