import pandas as pd

TOTAL_TIME = 590

tsv_files = [
     "F:\\GroupICATv4.0c_standalone_Win64\\ds003507_tsv_files\\ds003507_required_files\\sub-01_task-affect_run-3_events.tsv",
     # "F:\\GroupICATv4.0c_standalone_Win64\\ds003507_tsv_files\\ds003507_required_files\\sub-01_task-affect_run-2_events.tsv",
     # "F:\\GroupICATv4.0c_standalone_Win64\\ds003507_tsv_files\\ds003507_required_files\\sub-01_task-affect_run-3_events.tsv"
]

with open("F:\GroupICATv4.0c_standalone_Win64\ds003507_new_dat_files\sub-01_task-affect_run-3_events.dat", "w") as file:
    for file_no, tsv_file in enumerate(tsv_files):
        df = pd.read_csv(tsv_file, sep='\t')
        events_vs_time = [0 for _ in range(TOTAL_TIME)]

        for index, row in df.iterrows():
            onset = int(row["onset"])
            for i in range(int(row["duration"])):
                events_vs_time[onset+i-1] = int(row["trial_type"])

        event_1_vs_time = [1 if event == 1 else 0 for event in events_vs_time ]
        event_2_vs_time = [1 if event == 2 else 0 for event in events_vs_time ]
        event_3_vs_time = [1 if event == 3 else 0 for event in events_vs_time ]
        event_4_vs_time = [1 if event == 4 else 0 for event in events_vs_time ]
        event_5_vs_time = [1 if event == 5 else 0 for event in events_vs_time ]

        for entry in range(TOTAL_TIME):
            file.write(f"{event_1_vs_time[entry]} {event_2_vs_time[entry]}  {event_3_vs_time[entry]}  {event_4_vs_time[entry]}\n")

