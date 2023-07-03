""""
Functions made by Javi Rasero
Source:
https://github.com/CoAxLab/cofluctuating-task-connectivity/blob/main/notebooks/03-analyse_RSS.ipynb
https://github.com/CoAxLab/cofluctuating-task-connectivity/blob/main/scripts/05-compute_rss.py
"""

import os
from os.path import join as opj

from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# ^^^ pyforest auto-imports - don't write above this line
import matplotlib.pyplot as plt

# plt.rcParams["font.family"] = "Arial"

from nilearn.image import load_img
from scipy.spatial.distance import squareform
import tqdm


def extract_edge_ts(img):
    """This function loads nii.gz volume for a subject, of efc pairwise matrix
    for 268 parcellations based on shen atlas. Then extracts triangle vectors
    from symmetrical matrix."""
    edge_ts_img = load_img(img)
    edge_ts_data = np.squeeze(edge_ts_img.get_fdata()) # get rid of 1d in z
    n_vols = edge_ts_data.shape[-1] # no. of points in timeseries ?
    edge_ts_data = np.array([squareform(edge_ts_data[:, :, ii], \
                                        checks=False) for ii in range(n_vols)]) # remove symmetric matrix
    return edge_ts_data


def compute_rss(edge_ts):
    """This function computes the residual sum of squares, based on a get_fdata
    loaded nii.gz input.
    efc_vec = jr.extract_edge_ts(os.path.join(d_path, subj_file))
    print(jr.compute_rss(efc_vec))
    """
    rss = np.sqrt(np.sum(edge_ts ** 2, axis=1))
    return rss


def load_rss_task(data_dir, task_id, subj_lst):
    """This function takes a nii.gz, makes efc vector, then rss for all subjects"""
    pattern = data_dir + "/task-%s" % task_id + "/" + "sub-%d_ses-01_" + \
              "task-%s_space-MNI152NLin2009cAsym_desc-edges_bold.nii.gz" % task_id

    efc_lst = [extract_edge_ts(pattern % subj) for subj in tqdm.tqdm(subj_lst)] # loading bar
    rss_lst = np.array(list(map(compute_rss, efc_lst)))
    return rss_lst

def get_efc_trans(data_dir, task_id, subj_lst):
    """Get pairwise eFC matrix transpose so have full matrix for each timepoint"""
    pattern = data_dir + "/task-%s" % task_id + "/" + "sub-%d_ses-01_" + \
              "task-%s_space-MNI152NLin2009cAsym_desc-edges_bold.nii.gz" % task_id
    efc_mat_allsub = [np.squeeze(load_img(pattern % subj).get_fdata()).T for subj in \
               tqdm.tqdm(subj_lst)]
    return(np.array(efc_mat_allsub))


def get_efc_trans_sing(data_dir, task_id, subj_id):
    """Get pairwise eFC matrix transpose so have full matrix for each timepoint"""
    pattern = data_dir + "/task-%s" % task_id + "/" + "sub-%d_ses-01_" + \
              "task-%s_space-MNI152NLin2009cAsym_desc-edges_bold.nii.gz" % task_id
    efc_mat_sing_sub = np.squeeze(load_img(pattern % subj_id).get_fdata()).T
    return(np.array(efc_mat_sing_sub))



"""
def compute_rss_npy(bold_img_file, conf_file, atlas_file):
    run_img = load_img(bold_img_file)

    regex_conf = "trans|rot|white_matter$|csf$|global_signal$"
    conf_df = pd.read_csv(conf_file, sep="\t").filter(regex=regex_conf).fillna(0)

    edge_atlas = NiftiEdgeAtlas(atlas_file=atlas_file,
                                high_pass=1 / 187., t_r=2.0)

    edge_img = edge_atlas.fit_transform(run_img=run_img, confounds=conf_df)
    edge_ts = extract_edge_ts(edge_img)
    rss = np.sqrt(np.sum(edge_ts ** 2, axis=1))
    return rss
"""