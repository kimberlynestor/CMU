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

import matplotlib.pyplot as plt
import arviz as az
# import pymc3 as pm
# import seaborn as sns


# set shared scripts dir up one level
curr_dir = os.getcwd()
pars = Path(curr_dir).parents
par_dir = pars[0]
sys.path.insert(0, str(par_dir))


# lagged vector comparison - differences
task = 'stroop'
cort_bg_diff_stroop = np.load(f'{main_dir}{inter_path}{task}/cross_corr_cort_bg_peak_diff_{task}.npy')
cort_cb_diff_stroop = np.load(f'{main_dir}{inter_path}{task}/cross_corr_cort_cb_peak_diff_{task}.npy')
cb_bg_diff_stroop = np.load(f'{main_dir}{inter_path}{task}/cross_corr_cb_bg_peak_diff_{task}.npy')

task = 'msit'
cort_bg_diff_msit = np.load(f'{main_dir}{inter_path}{task}/cross_corr_cort_bg_peak_diff_{task}.npy')
cort_cb_diff_msit = np.load(f'{main_dir}{inter_path}{task}/cross_corr_cort_cb_peak_diff_{task}.npy')
cb_bg_diff_msit = np.load(f'{main_dir}{inter_path}{task}/cross_corr_cb_bg_peak_diff_{task}.npy')

# reliability test on diff vec
cort_bg_test = np.array(list(map(lambda tup: stats.spearmanr(tup[0], tup[1])[0], zip(cort_bg_diff_stroop, cort_bg_diff_msit))))
cort_cb_test = np.array(list(map(lambda tup: stats.spearmanr(tup[0], tup[1])[0], zip(cort_cb_diff_stroop, cort_cb_diff_msit))))
cb_bg_test = np.array(list(map(lambda tup: stats.spearmanr(tup[0], tup[1])[0], zip(cb_bg_diff_stroop, cb_bg_diff_msit))))

print(f'\nCross corr cort bg correlation > 0 = {len(list(filter(None, np.array(cort_bg_test)>0)))}')
print(f'Cross corr cort bg correlation < 0 = {len(list(filter(None, np.array(cort_bg_test)<0)))}\n')

print(f'\nCross corr cort cb correlation > 0 = {len(list(filter(None, np.array(cort_cb_test)>0)))}')
print(f'Cross corr cort cb correlation < 0 = {len(list(filter(None, np.array(cort_cb_test)<0)))}\n')

print(f'\nCross corr cb bg correlation > 0 = {len(list(filter(None, np.array(cb_bg_test)>0)))}')
print(f'Cross corr cb bg correlation < 0 = {len(list(filter(None, np.array(cb_bg_test)<0)))}\n')


#### kde plots
# cort x bg
az.plot_kde(cort_bg_test, rug=True)
plt.axvline(x=0, c='k', lw=1.2, alpha=0.28, ls='--', dashes=(7, 11))
plt.title('Bg \u2192 Cortex', size=12, fontname=font_lst[0], weight='bold')
plt.xlabel('Reliability', size=12, fontname=font_lst[0])
plt.ylabel('Kernel Density', size=12, fontname=font_lst[0])
# plt.savefig(f'{par_dir}/output/reliab_kde_cort_bg.png', dpi=1000)
# plt.show()
plt.close()

# cort x cb
az.plot_kde(cort_cb_test, rug=True)
plt.axvline(x=0, c='k', lw=1.2, alpha=0.28, ls='--', dashes=(7, 11))
plt.title('Cb \u2192 Cortex', size=12, fontname=font_lst[0], weight='bold')
plt.xlabel('Reliability', size=12, fontname=font_lst[0])
plt.ylabel('Kernel Density', size=12, fontname=font_lst[0])
# plt.savefig(f'{par_dir}/output/reliab_kde_cort_cb.png', dpi=1000)
# plt.show()
plt.close()

# cb x bg
az.plot_kde(cb_bg_test, rug=True)
plt.axvline(x=0, c='k', lw=1.2, alpha=0.28, ls='--', dashes=(7, 11))
plt.title('Bg \u2192 Cb', size=12, fontname=font_lst[0], weight='bold')
plt.xlabel('Reliability', size=12, fontname=font_lst[0])
plt.ylabel('Kernel Density', size=12, fontname=font_lst[0])
# plt.savefig(f'{par_dir}/output/reliab_kde_cb_bg.png', dpi=1000)
# plt.show()
plt.close()

task = 'stroop'
cort_bg_diff_stroop
cort_cb_diff_stroop
cb_bg_diff_stroop

task = 'msit'
cort_bg_diff_msit
cort_cb_diff_msit
cb_bg_diff_msit



# print(  bayes_fact(cort_bg_diff_stroop[0], cort_bg_diff_msit[0])  )

# get bayes factor
cort_bg_bf = np.array(list(map(lambda tup: bayes_fact(tup[0], tup[1]), zip(cort_bg_diff_stroop, cort_bg_diff_msit))))
cort_cb_bf = np.array(list(map(lambda tup: bayes_fact(tup[0], tup[1]), zip(cort_cb_diff_stroop, cort_cb_diff_msit))))
cb_bg_bf = np.array(list(map(lambda tup: bayes_fact(tup[0], tup[1]), zip(cb_bg_diff_stroop, cb_bg_diff_msit))))

print(f'cort_bg_diff BIC_stroop, BIC_msit, B_01 = {np.mean(cort_bg_bf, axis=0)}')
print(f'cort_cb_diff BIC_stroop, BIC_msit, B_01 = {np.mean(cort_cb_bf, axis=0)}')
print(f'cb_bg_diff BIC_stroop, BIC_msit, B_01 = {np.mean(cb_bg_bf, axis=0)}\n')

sys.exit()



from pingouin import ttest

# get bayes factor ttest
cort_bg_bf = np.array(list(map(lambda tup: float(ttest(tup[0], tup[1]).BF10[0]), zip(cort_bg_diff_stroop, cort_bg_diff_msit))))
cort_cb_bf = np.array(list(map(lambda tup: float(ttest(tup[0], tup[1]).BF10[0]), zip(cort_cb_diff_stroop, cort_cb_diff_msit))))
cb_bg_bf = np.array(list(map(lambda tup: float(ttest(tup[0], tup[1]).BF10[0]), zip(cb_bg_diff_stroop, cb_bg_diff_msit))))


# bf_cort_bg = ttest(cort_bg_diff_stroop[0], cort_bg_diff_msit[1]).BF10[0]
# print(bf_cort_bg.BF10[0])

print(f'Bayes factor cort bg avg = {np.mean(cort_bg_bf).round(2)}')
print(f'Bayes factor cort cb avg = {np.mean(cort_cb_bf).round(2)}')
print(f'Bayes factor cb bg avg = {np.mean(cb_bg_bf).round(2)}\n')


sys.exit()

#### bayes factor analysis
# cort x bg
with pm.Model() as model_g:
    mean = pm.Uniform('mean', lower=min(cort_bg_test), upper=max(cort_bg_test))
    std = pm.HalfNormal('std', sd=np.std(cort_bg_test))
    y = pm.Normal('y', mu=mean, sd=std, observed=cort_bg_test)
    trace_g = pm.sample(1000, tune=1000)

bayes_sum = az.summary(trace_g)
print(bayes_sum)
# print(model_g[1].marginal_likelihood)

sys.exit()

# plot simulated probs
az.plot_trace(trace_g)
plt.show()

# cort x cb
with pm.Model() as model_g:
    mean = pm.Uniform('mean', lower=min(cort_cb_test), upper=max(cort_cb_test))
    std = pm.HalfNormal('std', sd=np.std(cort_cb_test))
    y = pm.Normal('y', mu=mean, sd=std, observed=cort_cb_test)
    trace_g = pm.sample(1000, tune=1000)

bayes_sum = az.summary(trace_g)
print(bayes_sum)

# plot simulated probs
az.plot_trace(trace_g)
plt.show()

# cb x bg
with pm.Model() as model_g:
    mean = pm.Uniform('mean', lower=min(cb_bg_test), upper=max(cb_bg_test))
    std = pm.HalfNormal('std', sd=np.std(cb_bg_test))
    y = pm.Normal('y', mu=mean, sd=std, observed=cb_bg_test)
    trace_g = pm.sample(1000, tune=1000)

bayes_sum = az.summary(trace_g)
print(bayes_sum)

# plot simulated probs
az.plot_trace(trace_g)
plt.show()




sys.exit()

# scatter of reliab test
plt.scatter(np.ones(len(cort_bg_test)), cort_bg_test)
plt.scatter(np.ones(len(cort_bg_test))*2, cort_cb_test)
plt.scatter(np.ones(len(cort_bg_test))*3, cb_bg_test)
plt.axhline(0)
plt.show()


#######
# vector to vector comparison
task = 'stroop'
q_allsub_stroop = np.load(f'{main_dir}{inter_path}{task}/subjs_all_net_cort_q_{task}.npy')
eigen_cb_allsub_stroop = np.load(f'{main_dir}{inter_path}{task}/eigen_cen_cb_allsub_{task}.npy')
eigen_bg_allsub_stroop = np.load(f'{main_dir}{inter_path}{task}/eigen_cen_bg_allsub_{task}.npy')

task = 'msit'
q_allsub_msit = np.load(f'{main_dir}{inter_path}{task}/subjs_all_net_cort_q_{task}.npy')
eigen_cb_allsub_msit = np.load(f'{main_dir}{inter_path}{task}/eigen_cen_cb_allsub_{task}.npy')
eigen_bg_allsub_msit = np.load(f'{main_dir}{inter_path}{task}/eigen_cen_bg_allsub_{task}.npy')


cort_test = np.array(list(map(lambda tup: stats.spearmanr(tup[0], tup[1]).statistic, zip(q_allsub_stroop, q_allsub_msit))))
cb_test = np.array(list(map(lambda tup: stats.spearmanr(tup[0], tup[1]).statistic, zip(eigen_cb_allsub_stroop, eigen_cb_allsub_msit))))
bg_test = np.array(list(map(lambda tup: stats.spearmanr(tup[0], tup[1]).statistic, zip(eigen_bg_allsub_stroop, eigen_bg_allsub_msit))))



plt.scatter(np.ones(len(cort_test)), cort_test)
plt.scatter(np.ones(len(cb_test))*2, cb_test)
plt.scatter(np.ones(len(bg_test))*3, bg_test)
plt.axhline(0)
plt.show()



# plot ind subjs
q_allsub_avgts_stroop = np.average(q_allsub_stroop, axis=1)
q_allsub_avgts_msit = np.average(q_allsub_msit, axis=1)

x_stroop = np.ones(len(q_allsub_avgts_stroop))
x_msit = np.ones(len(q_allsub_avgts_msit))*2


# plot all subj avg ts
plt.scatter(x_stroop, q_allsub_avgts_stroop)
plt.scatter(x_msit, q_allsub_avgts_msit)

# plot vecs, subj stroop to msit
for i in range(len(x_stroop))[::6]:
    plt.plot([x_stroop[i], x_msit[i]], [q_allsub_avgts_stroop[i], q_allsub_avgts_msit[i]], c='k')

plt.show()


# mlr coeffs test
task = 'stroop'
q_mlr_allsub_stroop = np.load(f'{main_dir}{inter_path}{task}/mlr_coeffs_subjs_all_net_cort_mod_{task}.npy')

task = 'msit'
q_mlr_allsub_msit = np.load(f'{main_dir}{inter_path}{task}/mlr_coeffs_subjs_all_net_cort_mod_{task}.npy')


cort_mlr_test = list(map(lambda tup: stats.spearmanr(tup[0], tup[1]).statistic, zip(q_mlr_allsub_stroop, q_mlr_allsub_msit)))



plt.scatter(range(len(cort_mlr_test)), cort_mlr_test)
# plt.scatter(np.ones(len(cort_bg_test))*2, cort_cb_test)
# plt.scatter(np.ones(len(cort_bg_test))*3, cb_bg_test)
plt.axhline(0)
plt.show()

print(f'Cortex mlr correlation > 0 = {len(list(filter(None, np.array(cort_mlr_test)>0)))}')
print(f'Cortex mlr correlation < 0 = {len(list(filter(None, np.array(cort_mlr_test)<0)))}\n')

