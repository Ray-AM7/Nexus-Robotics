% This script processes converted .mat files using fileread.m.
% Modified to process only the first gesture and first participant with non-overlapping windows
% and format the output as desired.

clear
addpath(genpath([pwd filesep 'Output BM']))

clear
addpath(genpath([pwd filesep 'Output BM']))

% Parameters
fs_original = 2048;  % Original sampling frequency
fs_downsampled = 512;  % Target sampling frequency after downsampling
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
    
    for isub = 1:NSUB
        fileName = ['session' num2str(isession) '_participant', num2str(isub), '.mat'];
        temp_load_forearm = load(fileName, 'DATA_FOREARM');
        datafile = temp_load_forearm.DATA_FOREARM;

        for igesture = 1:NGESTURE
            % **Concatenate Trials into One Long Sequence**
            concatenated_signal = [];
            for itrial = 1:NTRIALS
                concatenated_signal = [concatenated_signal; datafile{itrial, igesture}];
            end

            % Monopolar average referencing
            %for ichannel_2 = 1:8
             %   temp2 = OneSet(1:8, :); % Proximal sensor channels, more important for amputees
              %  post_process(ichannel_2, :) = OneSet(ichannel_2, :) - mean(temp2, 1);
            %end

            % **Bipolar Average Referencing**
            for ichannel = 1:8
                if ichannel == 1
                    temp_ref = mean([concatenated_signal(8, :); concatenated_signal(2, :)], 1);
                elseif ichannel == 8
                    temp_ref = mean([concatenated_signal(7, :); concatenated_signal(1, :)], 1);
                else
                    temp_ref = mean([concatenated_signal(ichannel-1, :); concatenated_signal(ichannel+1, :)], 1);
                end
                concatenated_signal(ichannel, :) = concatenated_signal(ichannel, :) - temp_ref;
            end

            % **Downsample to 512 Hz**
            downsampled_signal = resample(concatenated_signal', fs_downsampled, fs_original)';

            % **Segment EMG into 128ms Windows with 32ms Overlap**
            segmented_data = segmentEMG(downsampled_signal', 0.128, 0.096, fs_downsampled, 1);
            segmented_data = permute(segmented_data, [3, 2, 1]); % From 8 x 409 x 175 to 175 x 409 x 8

            % **Compute Welch’s PSD Per Window (Instead of Full Spectrogram)**
            window_length = round(0.128 * fs_downsampled); % 128ms
            overlap = round(0.032 * fs_downsampled); % 32ms
            num_windows = size(segmented_data, 1);
            
            psd_windows = cell(num_windows, 1); % Store PSDs separately per window
            
            for i = 1:num_windows
                [pxx, f] = pwelch(segmented_data(i, :)', hamming(window_length), overlap, window_length, fs_downsampled);
                % From GPT: The Hamming window is a tapering function that smooths the edges of signal segments to reduce spectral leakage.
                % Without a window, the Fourier Transform assumes the signal repeats infinitely, leading to discontinuities.
                
                psd_windows{i} = pxx; % Save individual PSD per window
            end
            
            % Store feature set
            FeatSet{isub, igesture} = psd_windows;
        end
        disp(['Processed: Participant ', num2str(isub)])

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