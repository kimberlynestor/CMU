# Kimberly Nestor
# 09.2022
# This program takes as input a csv file with subject, task and mean absolute 
# connectivity block information and inputs that data into a mixed linear model.


library(lmerTest)
library(lme4)
library(broom)

filepath = '/home/kimberlynestor/gitrepo/int_seg/bg_cb'

## ON TASK
# load data
on.task.connec = read.csv(file=file.path(filepath, 'on_task_block_mu_connec.csv'), header=TRUE)
# head(on.task.connec)

# run on task mixed linear model
on.task.mlm.sum = summary(lmer(mu_connec ~ task + block + task:block + 
                                 (task | subj_ID), on.task.connec))

write.csv(on.task.mlm.sum$coefficients, file.path(filepath, 'subjs_all_net_cort/connec/', 
                                                  'on_task_connec_mlm_output_cort.csv'))
on.task.mlm.sum


## OFF TASK - mean task
# off task load data
off.task.connec = read.csv(file=file.path(filepath, 'off_task_block_mu_connec.csv'), header=TRUE)
# head(off.task.connec)

# run off task mixed linear model
off.task.mlm.sum = summary(lmer(mu_connec ~ task + block + task:block + 
                                 (task | subj_ID), off.task.connec))

write.csv(off.task.mlm.sum$coefficients, file.path(filepath, 'subjs_all_net_cort/connec/', 
                                                   'off_task_connec_mlm_output_cort.csv'))
off.task.mlm.sum

