% List of open inputs
nrun = 2; % enter the number of runs here
jobfile = {'F:\fMRIDataProcessing\emotional-valence-prediction-fmri\MatlabScripts\create_SPM_files_job.m'};
jobs = repmat(jobfile, 1, nrun);
inputs = cell(0, nrun);
for crun = 1:nrun
end
spm('defaults', 'FMRI');
spm_jobman('run', jobs, inputs{:});
