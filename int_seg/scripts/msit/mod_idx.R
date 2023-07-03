# Kimberly Nestor
# 03.2023
# This program takes as input a csv file with subject, task and mean modularity 
# index block information and inputs that data into a mixed linear model.


library(lmerTest)
library(lme4)
library(broom)

filepath = '/home/kimberlynestor/gitrepo/int_seg/data/IntermediateData/msit'
output = '/home/kimberlynestor/gitrepo/int_seg/output/msit'

## ON TASK
# load data
on.task.mod.idx = read.csv(file=file.path(filepath, 'on_task_block_mu_mod_idx_msit.csv'), header=TRUE)
# head(on.task.mod.idx)

# run on task mixed linear model
on.task.mlm.sum = summary(lmer(mu_mod_idx ~ task + block + task:block + 
                                 (task | subj_ID), on.task.mod.idx))

write.csv(on.task.mlm.sum$coefficients, file.path(output, 'on_task_mod_idx_mlm_output_cort_msit.csv'))
on.task.mlm.sum


# only block 1 con and inc
df.blk1 = on.task.mod.idx[on.task.mod.idx$block == 1,]

# run on task mixed linear model
sum.blk1 = summary(lmer(mu_mod_idx ~ task + (1 | subj_ID), df.blk1))
write.csv(sum.blk1$coefficients, file.path(output, 'on_task_blk1_mod_idx_mlm_output_cort_msit.csv'))
sum.blk1



# t.test(mu_mod_idx ~ task, data=df.blk1, paired=TRUE)


