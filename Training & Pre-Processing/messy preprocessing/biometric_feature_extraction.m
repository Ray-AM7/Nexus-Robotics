% This script processes converted .mat files using fileread.m.
% Modified to exclude feature extraction while retaining preprocessing and segmentation.

clear
addpath(genpath([pwd filesep 'Output BM']))

% Define parameters
fs = 2048;                   % Sampling frequency
NSUB = length(dir([pwd filesep 'Output BM' filesep 'Session1_converted'])) - 2; % Number of participants
NSESSION = length(dir([pwd filesep 'Output BM'])) - 2; % Number of sessions
NGESTURE = 17;              % Total number of gestures
NTRIALS = 7;                % Total number of trials

% Variables for processing
a1 = [];                      % Temporary variable to merge trials of a specific contraction
a2 = [];                      % Temporary variable for gesture-wise data
CompleteSet = [];             % Store final gestures and subjects for each session

%% Define output folder
if ~exist('Feature Extracted BM', 'dir')
    mkdir('Feature Extracted BM')
else
    disp('Overwriting')
    rmdir('Feature Extracted BM', 's')
    mkdir('Feature Extracted BM')
end

%% Flatten the trials to obtain Gestures x Subjects
for isession = 1:NSESSION
    for isub = 1:NSUB
        fileName = ['session' num2str(isession) '_participant', num2str(isub), '.mat'];
        temp_load_forearm = load(fileName, 'DATA_FOREARM');
        datafile = temp_load_forearm.DATA_FOREARM;

        for igesture = 1:NGESTURE + 1 % +1 to include REST gesture
            for itrial = 1:NTRIALS
                a1 = [a1; datafile{itrial, igesture}];
            end
            a2 = [a2, {a1}];
            a1 = [];
        end

        CompleteSet = [CompleteSet; a2];
        a2 = [];
        disp(['Loaded: ' num2str(isession) ' ' num2str(isub)])
    end
end

rmpath(genpath([pwd filesep 'Output BM']))         % Save memory

%% Segmentation and Processing
count = 0;
for isession = 1:NSESSION
    FeatSet = {}; % Store segmented raw data
    for isub = 1:NSUB
        for igesture = 1:NGESTURE
            OneSet = CompleteSet{isub, igesture}';       % Shape = 16 x TotalSamples
            post_process = [];
            
            % Monopolar average referencing
            for ichannel_2 = 1:8
                temp2 = OneSet(1:8, :);
                post_process(ichannel_2, :) = OneSet(ichannel_2, :) - mean(temp2, 1);
            end
            
            % Segment the EMG Data
            segData = segmentEMG(post_process', 0.2, 0.15, NTRIALS * 5, fs, 1);  % post_process' is NSamp x 8
            
            % Save segmented raw data (instead of extracted features)
            FeatSet(isub, igesture) = {segData};
        end
        count = count + 1;
        disp(['Processed: ', num2str(count), ' of ', num2str(NSUB * NSESSION), ' files'])
    end

    % Save segmented data for the session
    disp(['Saving: Session ', num2str(isession), ' biometric data'])
    save(['Feature Extracted BM' filesep 'Forearm_Session' num2str(isession) '.mat'], 'FeatSet', '-v7.3')

end
