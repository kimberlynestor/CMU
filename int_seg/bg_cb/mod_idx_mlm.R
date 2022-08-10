# Kimberly Nestor
# 08.2022
# This program takes as input a csv file with subject, task and mean modularity 
# index block information and inputs that data into a mixed linear model.


# install.packages(‘lmerTest’)
# install.packages('broom') # glance()
library(lmerTest)
library(lme4)
library(broom)

filepath = '/home/kimberlynestor/gitrepo/int_seg/bg_cb'

# on task load data
on.task.mod.idx = read.csv(file=file.path(filepath, 'on_task_block_mu_mod_idx.csv'), header=TRUE)
# head(on.task.mod.idx)

# run on task mixed linear model
on.task.mlm.sum = summary(lmer(mu_mod_idx ~ task + block + task:block + 
                                 (task | subj_ID), on.task.mod.idx))

write.csv(on.task.mlm.sum$coefficients, file.path(filepath, 'on_task_mod_idx_mlm_output.csv'))


# off task load data
off.task.mod.idx = read.csv(file=file.path(filepath, 'off_task_block_mu_mod_idx.csv'), header=TRUE)
# head(off.task.mod.idx)

# run off task mixed linear model
off.task.mlm.sum = summary(lmer(mu_mod_idx ~ task + block + task:block + 
                                 (task | subj_ID), off.task.mod.idx))
off.task.mlm.sum


