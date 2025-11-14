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
CombinedFeatSet = cell(1, NGESTURE);

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
        for igesture = 1:NGESTURE %switch to gesture major for easier labels

            OneSet = CompleteSet{isub, igesture}';       % accessed indexed cell w/ array w/ Shape = samples x channels
            post_process = [];
            
            % Monopolar average referencing
            for ichannel_2 = 1:8
                temp2 = OneSet(1:8, :); % Proximal sensor channels, more important for amputees
                post_process(ichannel_2, :) = OneSet(ichannel_2, :) - mean(temp2, 1);
            end
            
            % Bipolar average referencing with circular adjacency
            %for ichannel_2 = 1:8
             %   if ichannel_2 == 1
              %      temp2 = mean([OneSet(8, :); OneSet(2, :)], 1); % Channels 8 and 2 for circular adjacency
               % elseif ichannel_2 == 8
                %    temp2 = mean([OneSet(7, :); OneSet(1, :)], 1); % Channels 7 and 1 for circular adjacency
                %else
                 %   temp2 = mean([OneSet(ichannel_2 - 1, :); OneSet(ichannel_2 + 1, :)], 1); % Adjacent channels
                %end
                %post_process(ichannel_2, :) = OneSet(ichannel_2, :) - temp2;
            %end

            %add z score normalization, here we are still looking at one
            %specific emg sample so dist. can be remembered for dynamic
            %labeling right after windowing.


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

            % add labeling code after and windowing, accessing the
            % z-score dist for this specific emg sample from earlier to
            % assign any windows w/ ALL CHANNELS indivisually 70% z-score normalizes values under 25th
            % percentile of the z-normalized dist.
            
            % labels added to labels array or wtv each loop so no more code
            % needed

        end
    end
    
    %

    disp(['Finished Processing: Session ',num2str(isession),' biometric data'])
    
    
    CompleteSet = [];  % Freeing memory from previous things

    %Smush participants across all rows and sessions into 1 x 17 row with
    %30,057 windows in each gesture/cell
    for igesture = 1:NGESTURE
        combined_windows = {};  % Temporary storage to gather data from all participants
        
        for isub = 1:NSUB
            % Extract the cell containing the window data for the participant and gesture
            participant_windows = FeatSet{isub, igesture};  % 233x1 cell array
            
            % Append participant data to combined list
            combined_windows = [combined_windows; participant_windows];
        end
        
        % Store concatenated data into the 1x17 matrix
        if isempty(CombinedFeatSet{1, igesture}) 
            CombinedFeatSet{1, igesture} = combined_windows;
        else % Combining concatenated data w/ previous session in CombinedFeatSet, thats why we need if else here.
            CombinedFeatSet{1, igesture} = [CombinedFeatSet{1, igesture}; combined_windows]; 
        end
        
        disp(['Gesture ', num2str(igesture), ' combined across all participants.'])
    end
    FeatSet = {}; % Freeing memory from previous things before processing new loop

end

% Save the combined dataset from all sessions into 1 file (if u want
% seperate files move this into session loop & add session variable into 
% file name below and get rid of conditional above).
output_file = ['Feature Extracted BM' filesep 'Combined_Forearm_Data.mat'];
disp(['Saving combined data to ', output_file]);
save(output_file, 'CombinedFeatSet', '-v7.3');

disp('Data processing and combination into 1 row complete.');