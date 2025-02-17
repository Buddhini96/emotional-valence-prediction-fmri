% %---------------------%
% % Convert Onset Times %
% %---------------------%
% 
% % Converts timing files from BIDS format into a two-column format that can
% % be read by SPM
% 
% % The columns are:
% % 1. Onset (in seconds); and
% % 2. Duration (in seconds
% 
% 
% % Run this script from the directory that contains all of your subjects
% % (i.e., the Flanker directory)

% subjects = [01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26]; % Replace with a list of all of the subjects you wish to analyze

subjects = [21];

for subject=subjects
    
    subject = num2str(subject, '%02d'); % Zero-pads each number so that the subject ID is 2 characters long

    cd(['../sub-' subject '/func']) % Navigate to the subject's directory

    Run1_onsetTimes = tdfread(['sub-' subject '_task-affect_run-1_events.tsv'], '\t'); % Read onset times file
    Run1_onsetTimes.trial_type = string(Run1_onsetTimes.trial_type); % Convert char array to string array, to make logical comparisons easier

    Run1_1 = [];
    Run1_2 = [];
    Run1_3 = [];
    Run1_4 = [];

    for i = 1:length(Run1_onsetTimes.onset)
        if strtrim(Run1_onsetTimes.trial_type(i,:)) == '1'
            Run1_1 = [Run1_1; Run1_onsetTimes.onset(i,:) Run1_onsetTimes.duration(i,:)];
        elseif strtrim(Run1_onsetTimes.trial_type(i,:)) == '2'
            Run1_2 = [Run1_2; Run1_onsetTimes.onset(i,:) Run1_onsetTimes.duration(i,:)];
        elseif strtrim(Run1_onsetTimes.trial_type(i,:)) == '3'
            Run1_3 = [Run1_3; Run1_onsetTimes.onset(i,:) Run1_onsetTimes.duration(i,:)];
        elseif strtrim(Run1_onsetTimes.trial_type(i,:)) == '4'
            Run1_4 = [Run1_4; Run1_onsetTimes.onset(i,:) Run1_onsetTimes.duration(i,:)];
        end
    end

    Run2_onsetTimes = tdfread(['sub-' subject '_task-affect_run-2_events.tsv'], '\t');
    Run2_onsetTimes.trial_type = string(Run2_onsetTimes.trial_type);

    Run2_1 = [];
    Run2_2 = [];
    Run2_3 = [];
    Run2_4 = [];

    for i = 1:length(Run2_onsetTimes.onset)
        if strtrim(Run2_onsetTimes.trial_type(i,:)) == '1'
            Run2_1 = [Run2_1; Run2_onsetTimes.onset(i,:) Run2_onsetTimes.duration(i,:)];
        elseif strtrim(Run2_onsetTimes.trial_type(i,:)) == '2'
            Run2_2 = [Run2_2; Run2_onsetTimes.onset(i,:) Run2_onsetTimes.duration(i,:)];
        elseif strtrim(Run2_onsetTimes.trial_type(i,:)) == '3'
            Run2_3 = [Run2_3; Run2_onsetTimes.onset(i,:) Run2_onsetTimes.duration(i,:)];
        elseif strtrim(Run2_onsetTimes.trial_type(i,:)) == '4'
            Run2_4 = [Run2_4; Run2_onsetTimes.onset(i,:) Run2_onsetTimes.duration(i,:)];
        end
    end

    Run3_onsetTimes = tdfread(['sub-' subject '_task-affect_run-3_events.tsv'], '\t');
    Run3_onsetTimes.trial_type = string(Run3_onsetTimes.trial_type);

    Run3_1 = [];
    Run3_2 = [];
    Run3_3 = [];
    Run3_4 = [];

    for i = 1:length(Run3_onsetTimes.onset)
        if strtrim(Run3_onsetTimes.trial_type(i,:)) == '1'
            Run3_1 = [Run3_1; Run3_onsetTimes.onset(i,:) Run3_onsetTimes.duration(i,:)];
        elseif strtrim(Run3_onsetTimes.trial_type(i,:)) == '2'
            Run3_2 = [Run3_2; Run3_onsetTimes.onset(i,:) Run3_onsetTimes.duration(i,:)];
        elseif strtrim(Run3_onsetTimes.trial_type(i,:)) == '3'
            Run3_3 = [Run3_3; Run3_onsetTimes.onset(i,:) Run3_onsetTimes.duration(i,:)];
        elseif strtrim(Run3_onsetTimes.trial_type(i,:)) == '4'
            Run3_4 = [Run3_4; Run3_onsetTimes.onset(i,:) Run3_onsetTimes.duration(i,:)];
        end
    end


    % Save timing files into text files

    save('Run1_1.txt', 'Run1_1', '-ASCII');
    save('Run1_2.txt', 'Run1_2', '-ASCII');
    save('Run1_3.txt', 'Run1_3', '-ASCII');
    save('Run1_4.txt', 'Run1_4', '-ASCII');

    save('Run2_1.txt', 'Run2_1', '-ASCII');
    save('Run2_2.txt', 'Run2_2', '-ASCII');
    save('Run2_3.txt', 'Run2_3', '-ASCII');
    save('Run2_4.txt', 'Run2_4', '-ASCII');

    save('Run3_1.txt', 'Run3_1', '-ASCII');
    save('Run3_2.txt', 'Run3_2', '-ASCII');
    save('Run3_3.txt', 'Run3_3', '-ASCII');
    save('Run3_4.txt', 'Run3_4', '-ASCII');

    % Go back to Flanker directory

    cd ../..

end
