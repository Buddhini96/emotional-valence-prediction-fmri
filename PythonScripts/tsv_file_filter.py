import pandas as pd

tsv_files = [
     "F:\\GroupICATv4.0c_standalone_Win64\\ds003507_tsv_files\\ds003507_tsv_files\\sub-21_task-affect_run-1_events.tsv",
     "F:\\GroupICATv4.0c_standalone_Win64\\ds003507_tsv_files\\ds003507_tsv_files\\sub-21_task-affect_run-2_events.tsv",
     "F:\\GroupICATv4.0c_standalone_Win64\\ds003507_tsv_files\\ds003507_tsv_files\\sub-21_task-affect_run-3_events.tsv"
]

new_file_path = "F:\\GroupICATv4.0c_standalone_Win64\\ds003507_tsv_files\\ds003507_required_filtered_files\\sub-21\\"

for tsv_file in tsv_files:
    df = pd.read_csv(tsv_file, sep="\t")
    df_filtered = df[df["trial_type"].isin({1, 4})]
    tsv_file_name = tsv_file.split("\\")[-1]
    df_filtered.to_csv(new_file_path+tsv_file_name, sep="\t", index=False)
