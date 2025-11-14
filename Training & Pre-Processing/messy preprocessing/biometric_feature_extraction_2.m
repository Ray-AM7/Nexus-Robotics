% This script processes converted .mat files using fileread.m.
% Modified to process only the first gesture and first participant with non-overlapping windows
% and format the output as desired.

clear
addpath(genpath([pwd filesep 'Output BM']))

% Define parameters
fs = 2048;                   % Sampling frequency
NSUB = length(dir([pwd filesep 'Output BM' filesep 'Session1_converted'])) - 2; % Number of participants
NSESSION = length(dir([pwd filesep 'Output BM'])) - 2; % Number of sessions
NGESTURE = 17;              % Total number of gestures w/ rest
NTRIALS = 7;                % Total number of trials

% Variables for processing
a1 = [];                     % Temporary variable to merge trials of a specific contraction
a2 = [];                     % Temporary variable for gesture-wise data
CompleteSet = [];            % Store final gestures and subjects for the session

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
    addpath(genpath([pwd filesep 'Output BM']))
    CompleteSet = [];  
    for isub = 1:NSUB
        fileName = ['session' num2str(isession) '_participant', num2str(isub), '.mat'];
        temp_load_forearm = load(fileName, 'DATA_FOREARM');
        datafile = temp_load_forearm.DATA_FOREARM;

        for igesture = 1:NGESTURE % Process only the first gesture
            for itrial = 1:NTRIALS
                a1 = [a1; datafile{itrial, igesture}];
            end
            a2 = [a2, {a1}];
            a1 = [];
        end

        CompleteSet = [CompleteSet; a2];
        a2 = [];
        disp(['Loaded: Session ', num2str(isession), ', Participant ', num2str(isub)])
    end
    
    rmpath(genpath([pwd filesep 'Output BM']))         % Save memory

    %% Segmentation and Processing
    FeatSet = {}; % Store segmented raw data
    for isub = 1:NSUB
        for igesture = 1:NGESTURE
            OneSet = CompleteSet{isub, igesture}';       % accessed indexed cell w/ array w/ Shape = samples in window x channels
            post_process = [];
            
            % Monopolar average referencing
            for ichannel_2 = 1:8
                temp2 = OneSet(1:8, :); % Proximal sensor channels, more important for amputees
                post_process(ichannel_2, :) = OneSet(ichannel_2, :) - mean(temp2, 1);
            end
            
            % Segment the EMG Data (Non-Overlapping)
            segData = segmentEMG(post_process', 0.2, 0.15, NTRIALS * 5, fs, 1);  % stride = window size for no overlap
            
            % Adjust dimensions: 175 x 409 x 8
            segData = permute(segData, [3, 2, 1]); % From 8 x 409 x 175 to 175 x 409 x 8
            
            % Store each window as a separate 2D array in a 1D cell array
            windows = cell(size(segData, 1), 1);
            for i = 1:size(segData, 1)
                windows{i} = squeeze(segData(i, :, :)); % Each window as 409 x 8
            end
            
            % Save the 1D cell array into FeatSet
            FeatSet(isub, igesture) = {windows};
        end
    end
    
    % Save segmented data for inspection
    disp(['saving: Session ',num2str(isession),' biometric data'])
    save(['Feature Extracted BM' filesep 'Forearm_Session' num2str(isession) '.mat'], 'FeatSet', '-v7.3')

end