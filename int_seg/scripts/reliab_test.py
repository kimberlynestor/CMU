"""


"""

import sys
import os
from pathlib import Path

import jr_funcs as jr
from bg_cb_funcs import *
from task_blocks import *
from config import *

import numpy as np
from scipy import stats
from pingouin import ttest
from statsmodels.stats.multitest import fdrcorrection

# set shared scripts dir up one level
curr_dir = os.getcwd()
pars = Path(curr_dir).parents
par_dir = pars[0]
sys.path.insert(0, str(par_dir))


# lagged vector comparison - differences
task = 'stroop'
corr_cort_bg_stroop = np.load(f'{main_dir}{inter_path}{task}/cross_corr_cort_bg_{task}.npy')
corr_cort_cb_stroop = np.load(f'{main_dir}{inter_path}{task}/cross_corr_cort_cb_{task}.npy')
corr_cb_bg_stroop = np.load(f'{main_dir}{inter_path}{task}/cross_corr_cb_bg_{task}.npy')

task = 'msit'
corr_cort_bg_msit = np.load(f'{main_dir}{inter_path}{task}/cross_corr_cort_bg_{task}.npy')
corr_cort_cb_msit = np.load(f'{main_dir}{inter_path}{task}/cross_corr_cort_cb_{task}.npy')
corr_cb_bg_msit = np.load(f'{main_dir}{inter_path}{task}/cross_corr_cb_bg_{task}.npy')


## significance test of each lag --- mean=0 cause correlations

cort_bg_stroop_sigs = np.array(list(map(lambda subs_corr:stats.ttest_1samp(subs_corr, popmean=0 )[1], corr_cort_bg_stroop.T)))
cort_cb_stroop_sigs = np.array(list(map(lambda subs_corr:stats.ttest_1samp(subs_corr, popmean=0 )[1], corr_cort_cb_stroop.T)))
cb_bg_stroop_sigs = np.array(list(map(lambda subs_corr:stats.ttest_1samp(subs_corr, popmean=0 )[1], corr_cb_bg_stroop.T)))

cort_bg_msit_sigs = np.array(list(map(lambda subs_corr:stats.ttest_1samp(subs_corr, popmean=0 )[1], corr_cort_bg_msit.T)))
cort_cb_msit_sigs = np.array(list(map(lambda subs_corr:stats.ttest_1samp(subs_corr, popmean=0 )[1], corr_cort_cb_msit.T)))
cb_bg_msit_sigs = np.array(list(map(lambda subs_corr:stats.ttest_1samp(subs_corr, popmean=0 )[1], corr_cb_bg_msit.T)))

# do pval correction using fdr
cort_bg_stroop_sigs_fdr = fdrcorrection(cort_bg_stroop_sigs)
cort_cb_stroop_sigs_fdr = fdrcorrection(cort_cb_stroop_sigs)
cb_bg_stroop_sigs_fdr = fdrcorrection(cb_bg_stroop_sigs)

cort_bg_msit_sigs_fdr = fdrcorrection(cort_bg_msit_sigs)
cort_cb_msit_sigs_fdr = fdrcorrection(cort_cb_msit_sigs)
cb_bg_msit_sigs_fdr = fdrcorrection(cb_bg_msit_sigs)

# get idxs of sig lags
cort_bg_stroop_sigs_idx = np.where(cort_bg_stroop_sigs_fdr[1] < 0.05)[0].tolist()
cort_cb_stroop_sigs_idx = np.where(cort_cb_stroop_sigs_fdr[1] < 0.05)[0].tolist()
cb_bg_stroop_sigs_idx = np.where(cb_bg_stroop_sigs_fdr[1] < 0.05)[0].tolist()

cort_bg_msit_sigs_idx = np.where(cort_bg_msit_sigs_fdr[1] < 0.05)[0].tolist()
cort_cb_msit_sigs_idx = np.where(cort_cb_msit_sigs_fdr[1] < 0.05)[0].tolist()
cb_bg_msit_sigs_idx = np.where(cb_bg_msit_sigs_fdr[1] < 0.05)[0].tolist()


print('\nSTROOP sig idxs')
print(cort_bg_stroop_sigs_idx)
print(cort_cb_stroop_sigs_idx)
print(cb_bg_stroop_sigs_idx, '\n')

print('MSIT sig idxs')
print(cort_bg_msit_sigs_idx)
print(cort_cb_msit_sigs_idx)
print(cb_bg_msit_sigs_idx)

# print(cort_bg_msit_sigs_fdr)


## bayes factor test for reliability


# stats.pearsonr()