"""



"""
import numpy as np
import pandas as pd

# data path and subjects
main_dir = '/home/kimberlynestor/gitrepo/int_seg/data/'
d_path = 'pip_edge_ts/shen/'
dep_path = 'depend/'


## TASK BLOCK ONSET
df_task_events = pd.read_csv(opj(main_dir, dep_path, 'task-stroop_events.tsv'), sep="\t")

# timing for inter task rest
rest = df_task_events.iloc[0,0]

# make list of onset times for incongruent and congruent tasks
inc_cond = df_task_events[df_task_events.loc[:,"trial_type"]=="Incongruent"].to_numpy().T
inc_cond[2,:] = np.ones(len(inc_cond[0]))

con_cond = df_task_events[df_task_events.loc[:,"trial_type"]=="Congruent"].to_numpy().T
con_cond[2,:] = np.ones(len(con_cond[0]))

# onset times for inter trial 10sec rest
fix_cond = np.array( [df_task_events['onset'].to_numpy() - rest] + \
           [np.full(len(df_task_events['onset']), inc_cond[0][0])] + \
           [np.ones(len(df_task_events['onset']))] )

# task timing blocks - modularity
inc_block = list(zip(inc_cond[0], inc_cond[0]+inc_cond[1][0]))
con_block = list(zip(con_cond[0], con_cond[0]+con_cond[1][0]))
fix_block = list(zip(fix_cond[0], fix_cond[0]+fix_cond[1][0]))

# indices for inc and con, fix
inc_block_frames = np.array(inc_block)/2
inc_frames_idx = np.array(list(map(lambda i: list(range(int(i[0]), int(i[1]))), inc_block_frames))) -5

con_block_frames = np.array(con_block)/2
con_frames_idx = np.array(list(map(lambda i: list(range(int(i[0]), int(i[1]))), con_block_frames))) -5

fix_block_frames = np.array(fix_block)/2
fix_frames_idx = np.array(list(map(lambda i: list(range(int(i[0]), int(i[1]))), fix_block_frames)))[1:] -5

