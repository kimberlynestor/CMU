# Kimberly Nestor
# 10.2022
# This program


library(lme4)
library(tidyverse)
library(dplyr)

filepath = '/home/kimberlynestor/gitrepo/int_seg/bg_cb'


## load data - Cort mod, bg and cb connec full timescale
# off task load data
mlm.ts = read.csv(file=file.path(filepath, 'cort_mod_cortbgcb_connec_eigen_ts.csv'), header=TRUE)
# mlm.ts$connec_cb = abs(mlm.ts$connec_cb)
# mlm.ts$connec_bg = abs(mlm.ts$connec_bg)
head(mlm.ts)


mlm.ts.lim =  mlm.ts %>% select(1,5:7) 

# mixed linear model
ts.mlm = lmer(mod_idx ~ connec_bg + connec_cb + connec_bg:connec_cb + 
                            frame + inc_reg + con_reg + (1 | subj_ID), mlm.ts)
ts.mlm.sum = summary(ts.mlm)
ts.mlm.sum

ts.mlm.lim = lmer(mod_idx ~ connec_bg + connec_cb + connec_bg:connec_cb + (1 | subj_ID), mlm.ts.lim)
ts.mlm.lim.sum = summary(ts.mlm.lim)
ts.mlm.lim.sum

# check model AIC
ts.mlm.aic = AIC(ts.mlm)
ts.mlm.aic

ts.mlm.lim.aic = AIC(ts.mlm.lim)
ts.mlm.lim.aic


#### thalamic models
# model 2 - state model - mag of change q
mlm.ts$mod_idx_abs = abs(mlm.ts$mod_idx)

# mixed linear model
ts.mlm.thal = lmer(mod_idx_abs ~ connec_thal + frame + inc_reg + con_reg + (1 | subj_ID), mlm.ts)
ts.mlm.thal.sum = summary(ts.mlm.thal)
ts.mlm.thal.sum


ts.mlm.thal.lim = lmer(mod_idx_abs ~ connec_thal + (1 | subj_ID), mlm.ts)
ts.mlm.thal.lim.sum = summary(ts.mlm.thal.lim)
ts.mlm.thal.lim.sum


## EIGEN MODEL
ts.mlm.eigen = lmer(connec_cort ~ eigen_cb + eigen_bg + eigen_cb:eigen_bg + 
                frame + task + block + task:block + (task | subj_ID), mlm.ts)
ts.mlm.eigen.sum = summary(ts.mlm.eigen)
ts.mlm.eigen.sum

ts.mlm.eigen = lmer(mod_idx ~ eigen_cb + eigen_bg + eigen_cb:eigen_bg + 
                      frame + task + block + task:block + (task | subj_ID), mlm.ts)
ts.mlm.eigen.sum = summary(ts.mlm.eigen)
ts.mlm.eigen.sum


ts.mlm.eigen = lmer(mod_idx ~ eigen_cb + eigen_bg + eigen_cb:eigen_bg + (1 | subj_ID), mlm.ts)
ts.mlm.eigen.sum = summary(ts.mlm.eigen)
ts.mlm.eigen.sum

anova(ts.mlm.eigen)

## lag time shfit with eigen vals
mlm.ts.shift = read.csv(file=file.path(filepath, 'cort_mod_cortbgcb_connec_eigen_ts_shift.csv'), header=TRUE)
# head(mlm.ts.shift)

# lag 1
ts.mlm.eigen.lag1 = lmer(mod_idx_lag1 ~ eigen_cb + eigen_bg + eigen_cb:eigen_bg + 
                      frame + task + block + task:block + (task | subj_ID), mlm.ts.shift)
ts.mlm.eigen.sum.lag1 = summary(ts.mlm.eigen.lag1)
ts.mlm.eigen.sum.lag1


# lag 2
ts.mlm.eigen.lag2 = lmer(mod_idx_lag2 ~ eigen_cb + eigen_bg + eigen_cb:eigen_bg + 
                           frame + task + block + task:block + (task | subj_ID), mlm.ts.shift)
ts.mlm.eigen.sum.lag2 = summary(ts.mlm.eigen.lag2)
ts.mlm.eigen.sum.lag2

# lag 3
ts.mlm.eigen.lag3 = lmer(mod_idx_lag3 ~ eigen_cb + eigen_bg + eigen_cb:eigen_bg + 
                           frame + task + block + task:block + (task | subj_ID), mlm.ts.shift)
ts.mlm.eigen.sum.lag3 = summary(ts.mlm.eigen.lag3)
ts.mlm.eigen.sum.lag3

# lag 4
ts.mlm.eigen.lag4 = lmer(mod_idx_lag4 ~ eigen_cb + eigen_bg + eigen_cb:eigen_bg + 
                           frame + task + block + task:block + (task | subj_ID), mlm.ts.shift)
ts.mlm.eigen.sum.lag4 = summary(ts.mlm.eigen.lag4)
ts.mlm.eigen.sum.lag4

# lag 5
ts.mlm.eigen.lag5 = lmer(mod_idx_lag5 ~ eigen_cb + eigen_bg + eigen_cb:eigen_bg + 
                           frame + task + block + task:block + (task | subj_ID), mlm.ts.shift)
ts.mlm.eigen.sum.lag5 = summary(ts.mlm.eigen.lag5)
ts.mlm.eigen.sum.lag5


# lead 1
ts.mlm.eigen.lead1 = lmer(mod_idx_lead1 ~ eigen_cb + eigen_bg + eigen_cb:eigen_bg + 
                           frame + task + block + task:block + (task | subj_ID), mlm.ts.shift)
ts.mlm.eigen.sum.lead1 = summary(ts.mlm.eigen.lead1)
ts.mlm.eigen.sum.lead1

# lead 2
ts.mlm.eigen.lead2 = lmer(mod_idx_lead2 ~ eigen_cb + eigen_bg + eigen_cb:eigen_bg + 
                            frame + task + block + task:block + (task | subj_ID), mlm.ts.shift)
ts.mlm.eigen.sum.lead2 = summary(ts.mlm.eigen.lead2)
ts.mlm.eigen.sum.lead2

# lead 3
ts.mlm.eigen.lead3 = lmer(mod_idx_lead3 ~ eigen_cb + eigen_bg + eigen_cb:eigen_bg + 
                            frame + task + block + task:block + (task | subj_ID), mlm.ts.shift)
ts.mlm.eigen.sum.lead3 = summary(ts.mlm.eigen.lead3)
ts.mlm.eigen.sum.lead3

# lead 4
ts.mlm.eigen.lead4 = lmer(mod_idx_lead4 ~ eigen_cb + eigen_bg + eigen_cb:eigen_bg + 
                            frame + task + block + task:block + (task | subj_ID), mlm.ts.shift)
ts.mlm.eigen.sum.lead4 = summary(ts.mlm.eigen.lead4)
ts.mlm.eigen.sum.lead4

# lead 5
ts.mlm.eigen.lead5 = lmer(mod_idx_lead5 ~ eigen_cb + eigen_bg + eigen_cb:eigen_bg + 
                            frame + task + block + task:block + (task | subj_ID), mlm.ts.shift)
ts.mlm.eigen.sum.lead5 = summary(ts.mlm.eigen.lead5)
ts.mlm.eigen.sum.lead5


fixef(ts.mlm.eigen.lag1) [c('eigen_cb', 'eigen_bg')]
fixef(ts.mlm.eigen.lag2) [c('eigen_cb', 'eigen_bg')]
fixef(ts.mlm.eigen.lag3) [c('eigen_cb', 'eigen_bg')]
fixef(ts.mlm.eigen.lag4) [c('eigen_cb', 'eigen_bg')]
fixef(ts.mlm.eigen.lag5) [c('eigen_cb', 'eigen_bg')]

fixef(ts.mlm.eigen.lead1) [c('eigen_cb', 'eigen_bg')]
fixef(ts.mlm.eigen.lead2) [c('eigen_cb', 'eigen_bg')]
fixef(ts.mlm.eigen.lead3) [c('eigen_cb', 'eigen_bg')]
fixef(ts.mlm.eigen.lead4) [c('eigen_cb', 'eigen_bg')]
fixef(ts.mlm.eigen.lead5) [c('eigen_cb', 'eigen_bg')]

fixef(ts.mlm.eigen.lead4) ['eigen_bg']


##################################################################################
## average task blocks     
mlm.ts.avg.blks = read.csv(file=file.path(filepath, 'cort_mod_bgcb_eigen_ts_avg_blks.csv'), header=TRUE)
# mlm.ts.avg.blks

ts.mlm.eigen.avg.blks = lmer(mod_idx ~ eigen_cb + eigen_bg + eigen_cb:eigen_bg + (1 | subj_ID), mlm.ts.avg.blks)
ts.mlm.eigen.avg.blks.sum = summary(ts.mlm.eigen.avg.blks)
ts.mlm.eigen.avg.blks.sum

ts.mlm.eigen.avg.blks = lmer(mod_idx ~ eigen_cb + eigen_bg + (1 | subj_ID), mlm.ts.avg.blks)
ts.mlm.eigen.avg.blks.sum = summary(ts.mlm.eigen.avg.blks)
ts.mlm.eigen.avg.blks.sum

ts.mlm.eigen.avg.blks = lmer(mod_idx ~ eigen_cb + eigen_bg + eigen_cb:eigen_bg + 
                      frame + task + block + task:block + (task | subj_ID), mlm.ts.avg.blks)
ts.mlm.eigen.avg.blks.sum = summary(ts.mlm.eigen.avg.blks)
ts.mlm.eigen.avg.blks.sum


## lead and lag - average task blocks
mlm.ts.avg.blks.shift = read.csv(file=file.path(filepath, 'cort_mod_baba_eigen_ts_avg_blks_shift.csv'), header=TRUE)

shift.cols = tail(colnames(mlm.ts.avg.blks.shift), -10)
len.col = length(shift.col.lst)

cb.coeff.lst = as.vector(1:len.col)
bg.coeff.lst = as.vector(1:len.col)

# mlm.ts.avg.blks.shift[,10]
# loop to do mlm for all lags
for (i in 10:length(colnames(mlm.ts.avg.blks.shift))) {
  col.data = mlm.ts.avg.blks.shift[,i]
  
  ts.mlm.eigen.avg.blks = lmer(col.data ~ eigen_cb + eigen_bg + (1 | subj_ID), mlm.ts.avg.blks.shift)
  ts.mlm.eigen.avg.blks.sum = summary(ts.mlm.eigen.avg.blks)
  print(ts.mlm.eigen.avg.blks.sum)
  
  cb.coeff.lst[i-9] = fixef(ts.mlm.eigen.avg.blks) ['eigen_cb'] 
  bg.coeff.lst[i-9] = fixef(ts.mlm.eigen.avg.blks) ['eigen_bg']
  # confint(ts.mlm.eigen.avg.blks)
  
}

cb.coeff.lst
bg.coeff.lst

plot(seq(-35,35), cb.coeff.lst, col='red', type='o')
lines(seq(-35,35), bg.coeff.lst, col='blue')

cor(cb.coeff.lst, bg.coeff.lst)




