"""


"""


import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd

from os.path import join as opj
import sys

# np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(linewidth=np.inf)
# np.set_printoptions(linewidth=150)

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_columns', None)

# data path and subjects
main_dir = '/home/kimberlynestor/gitrepo/int_seg/data/'
d_path = 'AnalysisData/pip_edge_ts/shen/'
dep_path = 'depend/'
at_path = 'atlas/'

# params
rt = 2
frs = 280
ci = 1.96

# get subject list
subj_lst = np.loadtxt(opj(main_dir, dep_path, 'subjects_intersect_motion_035.txt'))


p_dict = {'Incongruent_cb':'#ff0080', 'Congruent_cb':'#0080ff', 'Difference_cb':'#ffff00', \
          'Incongruent':'#c6b5ff', 'Congruent':'#a5cae7', 'Fixation':'#ffbcd9', \
          'Difference':'#8A2D1C', 'Incongruent Fixation':'#c6b5ff', \
          'Congruent Fixation':'#a5cae7','cort_line_cb': '#575859', 'cort_line': '#717577', \
          'cb_line': 'tab:orange', 'bg_line': 'tab:green', 'thal_line': 'tab:red', \
          'cb_line_eig':'#710462', 'bg_line_eig':'#fb4321', 'bg_line_eig_var_b':'#fa7500', 'cb_line_eig_var_b':'#d103ff', \
          'cb_line_eig_var_c':'#e605c7', 'var_corr_cort_cb':'#07eff2', 'var_corr_cort_bg':'#ff0099', 'var_corr_cb_bg':'#05ed0d'}

# 'cb_line': '#6e7f80', cb_line_eig':'#800000 - light and dark red
# 'cb_line_eig':'#bf4f45', 'bg_line_eig':'#7a8c45' - light red and sage green
# 'Incongruent':'tab:orange', 'Congruent':'tab:blue', # 'Fixation':'#e1c1de'
# cc_yellow: #f9ffb8, cc_green: #b8ffbe, cc_salmon: #ffcbcb, burnt_orange: #cc5500, #964000
# 'Incongruent.cb':'#bd1250', 'Congruent.cb':'#0166a0'
# 'cb_line_eig_var_b':'#a1068b'

# p_dict_cb = {'Incongruent':'#ff3098', 'Congruent':'#1b8bfa', 'inc':'#b0b0b0', 'con':'#808080'}
p_dict_cb = {'Incongruent':'#ff0080', 'Congruent':'#0080ff', 'inc':'#b0b0b0', 'con':'#808080'} # super bright

# legend patches
inc_patch = mpatches.Patch(color=p_dict['Incongruent'], label='Incongruent', alpha=0.7) # 0.35
con_patch = mpatches.Patch(color=p_dict['Congruent'], label='Congruent', alpha=0.7) # 0.35

inc_patch_cb = mpatches.Patch(color=p_dict['Incongruent_cb'], label='Incongruent', alpha=0.35)
con_patch_cb = mpatches.Patch(color=p_dict['Congruent_cb'], label='Congruent', alpha=0.35)


cort_line = Line2D([0], [0], c=p_dict['cort_line'], lw=2, label='cort_line')
cb_line = Line2D([0], [0], c=p_dict['cb_line'], lw=2, label='Cerebellum', ls='-', alpha=0.8)
bg_line = Line2D([0], [0], c=p_dict['bg_line'], lw=2, label='Basal ganglia', ls='-', alpha=0.8)
thal_line = Line2D([0], [0], c=p_dict['thal_line'], lw=2, label='thal_line', ls='-', alpha=0.8)

cort_line_mod = Line2D([0], [0], c=p_dict['cort_line'], lw=2, label='Cortical modularity')
cort_line_mod_less = Line2D([0], [0], c=p_dict['cort_line'], lw=2, label='Cortex')
cort_line_mod_eig = Line2D([0], [0], c=p_dict['cort_line'], lw=2, label='Cortical modularity', ls=':', alpha=1)
cort_line_mod_eig_less = Line2D([0], [0], c=p_dict['cort_line_cb'], lw=1.5, label='Cortex', ls='--', alpha=1)

cb_line_connec = Line2D([0], [0], c=p_dict['cb_line'], lw=2, label='Cerebellum connectivity', ls='-', alpha=0.8)
bg_line_connec = Line2D([0], [0], c=p_dict['bg_line'], lw=2, label='Basal ganglia connectivity', ls='-', alpha=0.8)
cort_line_connec = Line2D([0], [0], c=p_dict['cort_line'], lw=2, label='Cortical connectivity', ls=':', alpha=0.9)

cb_bg_line_connec = Line2D([0], [0], c='tab:brown', lw=2, label='connec_cb_bg_line', ls='-', alpha=0.8)

inc_circ = Line2D(range(1), range(1), color='white', marker='o', markersize=11, \
                   markerfacecolor=p_dict['Incongruent'])
con_circ = Line2D(range(1), range(1), color='white', marker='o', markersize=11, \
                   markerfacecolor=p_dict['Congruent'])
fix_circ = Line2D(range(1), range(1), color='white', marker='o', markersize=11, \
                   markerfacecolor=p_dict['Fixation'])

inc_circ_cb = Line2D(range(1), range(1), color='white', marker='o', markersize=11, \
                   markerfacecolor=p_dict['Incongruent_cb'])
con_circ_cb = Line2D(range(1), range(1), color='white', marker='o', markersize=11, \
                   markerfacecolor=p_dict['Congruent_cb'])


cb_line_eig = Line2D([0], [0], c=p_dict['cb_line_eig'], lw=2, label='Cerebellum', ls='-', alpha=0.8)
cb_line_eig_var_b = Line2D([0], [0], c=p_dict['cb_line_eig_var_b'], lw=2, label='Cerebellum', ls='-', alpha=0.9)

bg_line_eig = Line2D([0], [0], c=p_dict['bg_line_eig'], lw=2, label='Basal ganglia', ls='-', alpha=0.8)
bg_line_eig_var_b = Line2D([0], [0], c=p_dict['bg_line_eig_var_b'], lw=2, label='Basal ganglia', ls='-', alpha=0.8)

cb_line_eig_lbl = Line2D([0], [0], c=p_dict['cb_line_eig'], lw=2, label='Cerebellum eigenvec centrality', ls='-', alpha=0.8)
bg_line_eig_lbl = Line2D([0], [0], c=p_dict['bg_line_eig'], lw=2, label='Basal ganglia eigenvec centrality', ls='-', alpha=0.8)
#
var_corr_cort_cb = Line2D([0], [0], c=p_dict['var_corr_cort_cb'], lw=2, label='Cb \u2192 Cortex', ls='-', alpha=1)
var_corr_cort_bg = Line2D([0], [0], c=p_dict['var_corr_cort_bg'], lw=2, label='Bg \u2192 Cortex', ls='-', alpha=1)
var_corr_cb_bg = Line2D([0], [0], c=p_dict['var_corr_cb_bg'], lw=2, label='Bg \u2192 Cb', ls='-', alpha=1)


# var_corr_mod = Line2D([0], [0], c=p_dict['cort_line_cb'], lw=2, ls='-')
# var_corr_cb = Line2D([0], [0], c=p_dict['cb_line_eig_var_c'], lw=2, ls='--', dashes=(3,2))
# var_corr_cb_b = Line2D([0], [0], c=p_dict['cb_line_eig_var_b'], lw=2, ls='--', dashes=(3,2))
#
# var_corr_bg = Line2D([0], [0], c=p_dict['bg_line_eig'], lw=2, ls='--', dashes=(3,2))
# var_corr_bg_flat = Line2D([0], [0], c=p_dict['bg_line_eig'], lw=2, ls='-')
