
% This script created condition .mat files, which define the structure of
% your experiment - the names, onsets, and durations of each experimental
% condition for a given run of a given subject. 
% These can be used in first-level GLM specification (under
% "Multiple conditions") to simplify design matrix definition. That is,
% conditions need not be entered individually; only one condition .mat file
% is necessary for each run of each subject.
% The script assumes that events.tsv files following the BIDS format (https://bids.neuroimaging.io/) 
% have already been created and are stored in each subject's functional folder.
%
% Written by Philipp Kuhnke (2022)
% kuhnke@cbs.mpg.de

%% Clear the workspace
clear all
close all

%% Setup


curr_func_folder = 'E:\ds000205-fmriprep\timing_files';

event_files = dir([curr_func_folder '/*events.txt']);

for iRun = 1:numel(event_files)

    curr_event_file = event_files(iRun)

    event_file_path = [curr_event_file.folder '/' curr_event_file.name]

        %% Import events file
        % this depends on the structure of your events.tsv files -> define
        % new via "Home"->"Import Data", export as script, and paste here
    opts = delimitedTextImportOptions("NumVariables", 11);

    % Specify range and delimiter
    opts.DataLines = [2, Inf];
    opts.Delimiter = "\t";
    
    % Specify column names and types
    opts.VariableNames = ["onset", "duration", "parametricLoss", "distanceFromIndifference", "parametricGain", "gain", "loss", "PTval", "respnum", "respcat", "response_time"];
    opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];
    
    % Specify file level properties
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "read";
        
        % Import the data
    events = readtable(event_file_path, opts);
    
    % Clear temporary variables
    clear opts

        %%
%     conds = {'Negative'; 'Positive'} % hard coded conditions: may be best/easiest option, especially if you want a specific order of condition regressors
    %conds = unique(events.trial_type); % automatically determine experimental conditions

    conds = size(events, 1);
    
    names = {};
    onsets = {};
    durations = {};

    for iCond = 1:conds
        
        curr_onsets = events.onset(iCond);

        curr_durations = events.duration(iCond);
            
        names(iCond) = {iCond}
        onsets(iCond) = {curr_onsets}
        durations(iCond) = {curr_durations}

        fprintf('Processing condition: %s\n', iCond);

    end

    save([ 'ds000205' '_run-0' num2str(iRun) '.mat'], ...
        'names','onsets','durations')

end

