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


## lead and lag time shfit with eigen vals
mlm.ts.shift = read.csv(file=file.path(filepath, 'cort_mod_cortbgcb_connec_eigen_ts_shift.csv'), header=TRUE)
head(mlm.ts.shift)

# loop to do mlm for all lags
for (i in 14:length(colnames(mlm.ts.shift))) {
  col.data = mlm.ts.shift[,i]
  
  ts.mlm.eigen.lag = lmer(col.data ~ eigen_cb + eigen_bg + eigen_cb:eigen_bg + 
                        frame + task + block + task:block + (task | subj_ID), mlm.ts.shift)
  ts.mlm.eigen.lag.sum = summary(ts.mlm.eigen.lag)
  
  # print(ts.mlm.eigen.lag.sum)
  # print(fixef(ts.mlm.eigen.lag) [c('eigen_cb', 'eigen_bg')])
}


###########
## average task blocks     
mlm.ts.avg.blks = read.csv(file=file.path(filepath, 'cort_mod_bgcb_eigen_ts_avg_blks.csv'), header=TRUE)
mlm.ts.avg.blks

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
mlm.ts.avg.blks.shift

df.cols = colnames(mlm.ts.avg.blks.shift)
shift.cols = tail(colnames(mlm.ts.avg.blks.shift), -9)
len.col = length(shift.cols)

cb.coeff.lst = as.vector(1:len.col)
bg.coeff.lst = as.vector(1:len.col)


ts.mlm.eigen.avg.blks.lag = lmer(mod_idx_lag1 ~ eigen_cb + eigen_bg + (1 | subj_ID), mlm.ts.avg.blks.shift)
ts.mlm.eigen.avg.blks.lag.sum = summary(ts.mlm.eigen.avg.blks.lag)
ts.mlm.eigen.avg.blks.lag.sum

# loop to do mlm for all lags
for (i in 9:length(colnames(mlm.ts.avg.blks.shift))) {
  col.data = mlm.ts.avg.blks.shift[,i]
  ts.mlm.eigen.avg.blks.lag = lmer(col.data ~ eigen_cb + eigen_bg + (1 | subj_ID), mlm.ts.avg.blks.shift)
  ts.mlm.eigen.avg.blks.lag.sum = summary(ts.mlm.eigen.avg.blks.lag)
  
  print(df.cols[i])
  print(ts.mlm.eigen.avg.blks.lag.sum)
  
  cb.coeff.lst[i-8] = fixef(ts.mlm.eigen.avg.blks.lag) ['eigen_cb'] 
  bg.coeff.lst[i-8] = fixef(ts.mlm.eigen.avg.blks.lag) ['eigen_bg']
  # confint(ts.mlm.eigen.avg.blks.lag)
  
  # print(fixef(ts.mlm.eigen.avg.blks.lag) ['eigen_bg'])
  # print(fixef(ts.mlm.eigen.avg.blks.lag) ['eigen_bg'])
}

# cb.coeff.lst
# bg.coeff.lst

plot(seq(-35,35), cb.coeff.lst, col='red', type='o', ylim=c(-0.05,0.18))
lines(seq(-35,35), bg.coeff.lst, col='blue')

# cor(cb.coeff.lst, bg.coeff.lst)






#### multi-level vector autoregression
# install.packages('mlVAR')
# depend = igraph, qgraph, graphicalVAR
# library(mlVAR)
# mlVARsim()

# source('/home/kimberlynestor/gitrepo/int_seg/bg_cb/mlVAR.R')
# source('/home/kimberlynestor/gitrepo/int_seg/bg_cb/mlVARmodel.R')
# source('/home/kimberlynestor/gitrepo/int_seg/bg_cb/lmer_murmur.R')

# mlVARsim(nPerson=50, nNode=3, nTime=50, lag=1)
# fit = mlVAR(mlm.ts.avg.blks, vars=c('mod_idx', 'eigen_cb', 'eigen_bg'), idvar='subj_ID', lags=1, temporal='correlated' )
# summary(fit)
