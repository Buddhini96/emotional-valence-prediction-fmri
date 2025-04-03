import os
from collections import defaultdict

import numpy as np
import nibabel as nib
import pandas as pd


def get_nii_data(file_path):
    nii_image = nib.load(file_path)
    beta_data = nii_image.get_fdata()
    beta_data = np.nan_to_num(beta_data)
    flattened_beta_data = beta_data.flatten()
    return flattened_beta_data

def extract_time_confounds(no_of_subjects, no_of_runs, base_path):
    subjects = [str(i).zfill(2) for i in range(1, no_of_subjects+1)]

    for subject in subjects:
        for run in range(1, no_of_runs+1):
            tsv_file = f"{base_path}\\sub-{subject}\\sub-{subject}_task-view_run-{str(run).zfill(2)}_desc-confounds_timeseries.tsv"
            df = pd.read_csv(tsv_file, sep='\t')
            df = df[["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]]
            with open(f'{base_path}\\sub-{subject}\\regressors_run{run}_output.txt', 'w') as f:
                for index, row in df.iterrows():
                    f.write(
                        f"  {row['trans_x']}   {row['trans_y']}    {row['trans_z']}   {row['rot_x']}    {row['rot_y']}   {row['rot_z']}    \n")

#extract_time_confounds(11, 2, "E:\\ds000205")

def read_ds00025_event_file(split_runs=False):
    tsv_folder_path = f"E:\\ds000205-fmriprep\\timing_files\\event_files\\"
    Y = []
    for filename in os.listdir(tsv_folder_path):
        file_path = os.path.join(tsv_folder_path, filename)
        data = np.loadtxt(file_path, delimiter='\t', dtype=str, skiprows=1)
        numeric_column = data[:, 3]
        y = [1 if x == "Positive" else 0 for x in list(numeric_column)]
        if split_runs:
            Y.append(y)
        else:
            Y.extend(y)
    return Y

def read_ds003507_event_file(subject_name, split_runs=False):
    tsv_folder_path = f"F:\\ds003507\\ds003507_tsv_files\\{subject_name}\\"
    Y = []
    for filename in os.listdir(tsv_folder_path):
        file_path = os.path.join(tsv_folder_path, filename)
        data = np.loadtxt(file_path, delimiter='\t', dtype=str, skiprows=1)
        numeric_column = data[:, 2]
        y = [int(x) for x in list(numeric_column)]
        if split_runs:
            Y.append(y)
        else:
            Y.extend(y)
    return Y


def get_ds_00025_stimulus_occurance_matrix():
    stimulus_id_onset = defaultdict(list)
    tsv_folder_path = f"E:\\ds000205-fmriprep\\timing_files\\event_files\\"
    for filename in os.listdir(tsv_folder_path):
        file_path = os.path.join(tsv_folder_path, filename)
        df = pd.read_csv(file_path, delimiter="\t")
        for i, row in df.iterrows():
            stimulus_id_onset[row["StimulusID"]].append(i)
    return stimulus_id_onset

