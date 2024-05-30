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
from pingouin import bayesfactor_pearson
from statsmodels.stats.multitest import fdrcorrection

# set shared scripts dir up one level
curr_dir = os.getcwd()
pars = Path(curr_dir).parents
par_dir = pars[0]
sys.path.insert(0, str(par_dir))


# open file to save output
output_file = open(f'{pars[0]}/output/cross_corr_lag_sig_reliab_test.csv', 'w')
output_file = open(f'{pars[0]}/output/cross_corr_lag_sig_reliab_test.csv', 'a')


# lagged vector comparison - differences
task = 'stroop'
corr_cort_bg_stroop = np.load(f'{main_dir}{inter_path}{task}/cross_corr_cort_bg_{task}.npy')
corr_cort_cb_stroop = np.load(f'{main_dir}{inter_path}{task}/cross_corr_cort_cb_{task}.npy')
corr_cb_bg_stroop = np.load(f'{main_dir}{inter_path}{task}/cross_corr_cb_bg_{task}.npy')

task = 'msit'
corr_cort_bg_msit = np.load(f'{main_dir}{inter_path}{task}/cross_corr_cort_bg_{task}.npy')
corr_cort_cb_msit = np.load(f'{main_dir}{inter_path}{task}/cross_corr_cort_cb_{task}.npy')
corr_cb_bg_msit = np.load(f'{main_dir}{inter_path}{task}/cross_corr_cb_bg_{task}.npy')

n_sub = corr_cort_bg_stroop.shape[0]

## significance test of each lag --- mean=0 cause correlations

cort_bg_stroop_sigs = np.array(list(map(lambda subs_corr:stats.ttest_1samp(subs_corr, popmean=0 )[1], corr_cort_bg_stroop.T)))
cort_cb_stroop_sigs = np.array(list(map(lambda subs_corr:stats.ttest_1samp(subs_corr, popmean=0 )[1], corr_cort_cb_stroop.T)))
cb_bg_stroop_sigs = np.array(list(map(lambda subs_corr:stats.ttest_1samp(subs_corr, popmean=0 )[1], corr_cb_bg_stroop.T)))
cort_bg_msit_sigs = np.array(list(map(lambda subs_corr:stats.ttest_1samp(subs_corr, popmean=0 )[1], corr_cort_bg_msit.T)))
cort_cb_msit_sigs = np.array(list(map(lambda subs_corr:stats.ttest_1samp(subs_corr, popmean=0 )[1], corr_cort_cb_msit.T)))
cb_bg_msit_sigs = np.array(list(map(lambda subs_corr:stats.ttest_1samp(subs_corr, popmean=0 )[1], corr_cb_bg_msit.T)))

output_file.write(f'cort_bg_stroop_sigs = {list(cort_bg_stroop_sigs)}\n')
output_file.write(f'cort_cb_stroop_sigs = {list(cort_cb_stroop_sigs)}\n')
output_file.write(f'cb_bg_stroop_sigs = {list(cb_bg_stroop_sigs)}\n')
output_file.write(f'cort_bg_msit_sigs = {list(cort_bg_msit_sigs)}\n')
output_file.write(f'cort_cb_msit_sigs = {list(cort_cb_msit_sigs)}\n')
output_file.write(f'cb_bg_msit_sigs = {list(cb_bg_msit_sigs)}\n')

# do pval correction using fdr
cort_bg_stroop_sigs_fdr = fdrcorrection(cort_bg_stroop_sigs)
cort_cb_stroop_sigs_fdr = fdrcorrection(cort_cb_stroop_sigs)
cb_bg_stroop_sigs_fdr = fdrcorrection(cb_bg_stroop_sigs)
cort_bg_msit_sigs_fdr = fdrcorrection(cort_bg_msit_sigs)
cort_cb_msit_sigs_fdr = fdrcorrection(cort_cb_msit_sigs)
cb_bg_msit_sigs_fdr = fdrcorrection(cb_bg_msit_sigs)


output_file.write(f'cort_bg_stroop_sigs_fdr = {list(cort_bg_stroop_sigs_fdr[1])}\n')
output_file.write(f'cort_cb_stroop_sigs_fdr = {list(cort_cb_stroop_sigs_fdr[1])}\n')
output_file.write(f'cb_bg_stroop_sigs_fdr = {list(cb_bg_stroop_sigs_fdr[1])}\n')
output_file.write(f'cort_bg_msit_sigs_fdr = {list(cort_bg_msit_sigs_fdr[1])}\n')
output_file.write(f'cort_cb_msit_sigs_fdr = {list(cort_cb_msit_sigs_fdr[1])}\n')
output_file.write(f'cb_bg_msit_sigs_fdr = {list(cb_bg_msit_sigs_fdr[1])}\n')


# get idxs of sig lags
cort_bg_stroop_sigs_idx = np.where(cort_bg_stroop_sigs_fdr[1] < 0.05)[0].tolist()
cort_cb_stroop_sigs_idx = np.where(cort_cb_stroop_sigs_fdr[1] < 0.05)[0].tolist()
cb_bg_stroop_sigs_idx = np.where(cb_bg_stroop_sigs_fdr[1] < 0.05)[0].tolist()

cort_bg_msit_sigs_idx = np.where(cort_bg_msit_sigs_fdr[1] < 0.05)[0].tolist()
cort_cb_msit_sigs_idx = np.where(cort_cb_msit_sigs_fdr[1] < 0.05)[0].tolist()
cb_bg_msit_sigs_idx = np.where(cb_bg_msit_sigs_fdr[1] < 0.05)[0].tolist()


print('\nSTROOP sig idxs')
print(f'cort_bg_stroop_sigs_idx = {cort_bg_stroop_sigs_idx}')
print(f'cort_cb_stroop_sigs_idx = {cort_cb_stroop_sigs_idx}')
print(f'cb_bg_stroop_sigs_idx = {cb_bg_stroop_sigs_idx}', '\n')
output_file.write(f'cort_bg_stroop_sigs_idx = {cort_bg_stroop_sigs_idx}\n')
output_file.write(f'cort_cb_stroop_sigs_idx = {cort_cb_stroop_sigs_idx}\n')
output_file.write(f'cb_bg_stroop_sigs_idx = {cb_bg_stroop_sigs_idx}\n')

print('MSIT sig idxs')
print(f'cort_bg_msit_sigs_idx = {cort_bg_msit_sigs_idx}')
print(f'cort_cb_msit_sigs_idx = {cort_cb_msit_sigs_idx}')
print(f'cb_bg_msit_sigs_idx = {cb_bg_msit_sigs_idx}')
output_file.write(f'cort_bg_msit_sigs_idx = {cort_bg_msit_sigs_idx}\n')
output_file.write(f'cort_cb_msit_sigs_idx = {cort_cb_msit_sigs_idx}\n')
output_file.write(f'cb_bg_msit_sigs_idx = {cb_bg_msit_sigs_idx}\n')

# print(cort_bg_msit_sigs_fdr)
print('\n')


## bayes factor test for reliability
# get set intersection of significant lags
cort_bg_idxs = list(set(cort_bg_stroop_sigs_idx).intersection(set(cort_bg_msit_sigs_idx)))
cort_cb_idxs = list(set(cort_cb_stroop_sigs_idx).intersection(set(cort_cb_msit_sigs_idx)))
cb_bg_idxs = sorted(list(set(cb_bg_stroop_sigs_idx).intersection(set(cb_bg_msit_sigs_idx))))

output_file.write(f'cort_bg_sig_intersect_lags_idxs = {cort_bg_idxs}\n')
output_file.write(f'cort_cb_sig_intersect_lags_idxs = {cort_cb_idxs}\n')
output_file.write(f'cb_bg_sig_intersect_lags_idxs = {cb_bg_idxs}\n')


# limit data by overlapping sig lags
corr_cort_bg_stroop_sig_lags = corr_cort_bg_stroop.T[cort_bg_idxs, :]
corr_cort_bg_msit_sig_lags = corr_cort_bg_msit.T[cort_bg_idxs, :]

corr_cort_cb_stroop_sig_lags = corr_cort_cb_stroop.T[cort_cb_idxs, :]
corr_cort_cb_msit_sig_lags = corr_cort_cb_msit.T[cort_cb_idxs, :]

corr_cb_bg_stroop_sig_lags = corr_cb_bg_stroop.T[cb_bg_idxs, :]
corr_cb_bg_msit_sig_lags = corr_cb_bg_msit.T[cb_bg_idxs, :]


# calculate correlation
cort_bg_corr = list(map(lambda i:stats.pearsonr(i[0], i[1])[0], zip(corr_cort_bg_stroop_sig_lags, corr_cort_bg_msit_sig_lags)))
cort_cb_corr = list(map(lambda i:stats.pearsonr(i[0], i[1])[0], zip(corr_cort_cb_stroop_sig_lags, corr_cort_cb_msit_sig_lags)))
cb_bg_corr = list(map(lambda i:stats.pearsonr(i[0], i[1])[0], zip(corr_cb_bg_stroop_sig_lags, corr_cb_bg_msit_sig_lags)))

output_file.write(f'cort_bg_sig_intersect_lags_corr = {cort_bg_corr}\n')
output_file.write(f'cort_cb_sig_intersect_lags_corr = {cort_cb_corr}\n')
output_file.write(f'cb_bg_sig_intersect_lags_corr = {cb_bg_corr}\n')


# calculate bayes factor value
cort_bg_bf = list(map(lambda i:bayesfactor_pearson(stats.pearsonr(i[0], i[1])[0], n_sub, alternative='greater'), \
                      zip(corr_cort_bg_stroop_sig_lags, corr_cort_bg_msit_sig_lags)))
cort_cb_bf = list(map(lambda i:bayesfactor_pearson(stats.pearsonr(i[0], i[1])[0], n_sub, alternative='greater'), \
                      zip(corr_cort_cb_stroop_sig_lags, corr_cort_cb_msit_sig_lags)))
cb_bg_bf = list(map(lambda i:bayesfactor_pearson(stats.pearsonr(i[0], i[1])[0], n_sub, alternative='greater'), \
                      zip(corr_cb_bg_stroop_sig_lags, corr_cb_bg_msit_sig_lags)))

output_file.write(f'cort_bg_sig_intersect_lags_bf = {cort_bg_bf}\n')
output_file.write(f'cort_cb_sig_intersect_lags_bf = {cort_cb_bf}\n')
output_file.write(f'cb_bg_sig_intersect_lags_bf = {cb_bg_bf}\n')

print(f'cort_bg_idxs = {cort_bg_idxs}')
print(f'cort_bg_corr = {cort_bg_corr}')
print(f'cort_bg_bf = {cort_bg_bf}\n')

print(f'cort_cb_idxs = {cort_cb_idxs}')
print(f'cort_cb_corr = {cort_cb_corr}')
print(f'cort_cb_bf = {cort_cb_bf}\n')

print(f'cb_bg_idxs = {cb_bg_idxs}')
print(f'cb_bg_corr = {cb_bg_corr}')
print(f'cb_bg_bf = {cb_bg_bf}\n')

