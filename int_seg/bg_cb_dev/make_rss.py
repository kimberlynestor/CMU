"""


"""

import sys
import os
from os.path import join as opj

import numpy as np

import jr_funcs as jr

# paths to data and dependencies
main_dir = '/home/kimberlynestor/gitrepo/int_seg/data/'
d_path = 'pip_edge_ts/shen/'
dep_path = 'depend/'
at_path = 'atlas/'

d_path_ss = 'task-msit/'
subj_file = 'sub-4285_ses-01_task-msit_space-MNI152NLin2009cAsym_desc-edges_bold.nii.gz'


#get efc for single subject
efc_vec = jr.extract_edge_ts(opj(main_dir, d_path, d_path_ss, subj_file))

# all subjs list and atlas
subj_lst = np.loadtxt(opj(main_dir, dep_path, 'subjects_intersect_motion_035.txt'))
atlas_file = opj(main_dir, at_path, "atlases/shen_2mm_268_parcellation.nii.gz")


sys.exit()




for task_id in ["stroop", "msit", "rest"]:
    bold_pattern = opj(project_dir, "data/preproc_bold/task-%s" % task_id + "/" + \
                       "sub-%d_ses-01_" + "task-%s_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" % task_id)

    conf_pattern = opj(project_dir, "data/confounders/task-%s" % task_id + "/" + \
                       "sub-%d_ses-01_" + "task-%s_desc-confounds_regressors.tsv" % task_id)

    output_dir = opj(project_dir, "results", "rss_w_task", "task-%s" % task_id)
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    for subj in tqdm(final_subjects):
        rss = compute_rss(bold_img_file=bold_pattern % subj, conf_file=conf_pattern % subj,
                          atlas_file=atlas_file)

        filename = Path(bold_pattern % subj).name
        filename = filename.replace("preproc_bold.nii.gz", "rss.npy")
        np.save(opj(output_dir, filename), rss)
sys.exit()