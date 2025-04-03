import pandas as pd

# List of TSV files to combine
file1 = "F:\\GroupICATv4.0c_standalone_Win64\\ds003507_dat_files\\convolved_sub-01_task-affect_run-1_events.dat"
file2 = "F:\\GroupICATv4.0c_standalone_Win64\\ds003507_dat_files\\convolved_sub-01_task-affect_run-2_events.dat"
file3 = "F:\\GroupICATv4.0c_standalone_Win64\\ds003507_dat_files\\convolved_sub-01_task-affect_run-3_events.dat"

output_file = "F:\\GroupICATv4.0c_standalone_Win64\\ds003507_dat_files\\convolved_sub-01_task-affect_events.dat"

# Read and combine all TSV files
files = [file1, file2, file3]

dataframes = [pd.read_csv(file, delim_whitespace=True) for file in files]
print(dataframes[0].shape)
print(dataframes[1].shape)
print(dataframes[2].shape)
combined_df = pd.concat(dataframes, axis=1, ignore_index=True)

# Save the combined data to a new TSV file
combined_df.to_csv(output_file, sep="\t", index=False)
print(f"Combined file saved as {output_file}")
