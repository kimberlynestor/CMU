"""


"""


import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import numpy as np
from os.path import join as opj

# data path and subjects
main_dir = '/home/kimberlynestor/gitrepo/int_seg/data/'
d_path = 'pip_edge_ts/shen/'
dep_path = 'depend/'

# get subject list
subj_lst = np.loadtxt(opj(main_dir, dep_path, 'subjects_intersect_motion_035.txt'))


p_dict = {'Incongruent':'#c6b5ff', 'Congruent':'#a5cae7', 'Fixation':'#ffbcd9', \
          'Difference':'#8A2D1C', 'Incongruent Fixation':'#c6b5ff', \
          'Congruent Fixation':'#a5cae7', 'cort_line': '#717577', \
          'cb_line': 'tab:orange', 'bg_line': 'tab:green', 'thal_line': 'tab:red'}
# 'cb_line': '#6e7f80'
# 'Incongruent':'tab:orange', 'Congruent':'tab:blue', # 'Fixation':'#e1c1de'
# cc_yellow: #f9ffb8, cc_green: #b8ffbe, cc_salmon: #ffcbcb, burnt_orange: #cc5500, #964000


# legend patches
inc_patch = mpatches.Patch(color=p_dict['Incongruent'], label='Incongruent', alpha=0.7) # 0.35
con_patch = mpatches.Patch(color=p_dict['Congruent'], label='Congruent', alpha=0.7) # 0.35

cort_line = Line2D([0], [0], c=p_dict['cort_line'], lw=2, label='cort_line')
cb_line = Line2D([0], [0], c=p_dict['cb_line'], lw=2, label='cb_line', ls='-', alpha=0.8)
bg_line = Line2D([0], [0], c=p_dict['bg_line'], lw=2, label='bg_line', ls='-', alpha=0.8)
thal_line = Line2D([0], [0], c=p_dict['thal_line'], lw=2, label='thal_line', ls='-', alpha=0.8)

cort_line_mod = Line2D([0], [0], c=p_dict['cort_line'], lw=2, label='mod_cort_line')
cb_line_connec = Line2D([0], [0], c=p_dict['cb_line'], lw=2, label='connec_cb_line', ls='-', alpha=0.8)
bg_line_connec = Line2D([0], [0], c=p_dict['bg_line'], lw=2, label='connec_bg_line', ls='-', alpha=0.8)

cb_bg_line_connec = Line2D([0], [0], c='tab:brown', lw=2, label='connec_cb_bg_line', ls='-', alpha=0.8)

inc_circ = Line2D(range(1), range(1), color='white', marker='o', markersize=11, \
                   markerfacecolor=p_dict['Incongruent'])
con_circ = Line2D(range(1), range(1), color='white', marker='o', markersize=11, \
                   markerfacecolor=p_dict['Congruent'])
fix_circ = Line2D(range(1), range(1), color='white', marker='o', markersize=11, \
                   markerfacecolor=p_dict['Fixation'])
