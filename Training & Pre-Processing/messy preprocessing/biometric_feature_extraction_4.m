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
num_windows_per_seq = 233;
% Variables for processing
a1 = [];                     % Temporary variable to merge trials of a specific contraction
a2 = [];                     % Temporary variable for gesture-wise data
CompleteSet = [];            % Store final gestures and subjects for the session
CombinedFeatSet = cell(1, 1);
labels = [];

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
    for igesture = 1:NGESTURE
        for isub = 1:NSUB % switched to gesture major for easier labels

            OneSet = CompleteSet{isub, igesture}';       % accessed indexed cell w/ array w/ Shape = samples x channels
            post_process = [];

            % Replace NaN values in OneSet with zeros before normalization
            OneSet(isnan(OneSet)) = 0;
            
            % Monopolar average referencing
            for ichannel_2 = 1:8
                temp2 = OneSet(1:8, :); % Proximal sensor channels, more important for amputees
                post_process(ichannel_2, :) = OneSet(ichannel_2, :) - mean(temp2, 1);
            end
            

            % Apply Z-Score Normalization
            %for ichannel_2 = 1:8
               % mean_val = mean(post_process(ichannel_2, :));
                %std_val = std(post_process(ichannel_2, :));

                % Avoid division by zero
                %if std_val == 0
                 %   std_val = 1;
                %end

                % Apply Z-score normalization
                %post_process(ichannel_2, :) = (post_process(ichannel_2, :) - mean_val) / std_val;
            %end


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

            % add labeling code after windowing, accessing the
            % z-score dist for this specific emg sample from earlier to
            % assign any windows w/ ALL CHANNELS indivisually 70% z-score normalizes values under 25th
            % percentile of the z-normalized dist.
            
            % labels added to labels array or wtv each loop so no more code
            % needed           
            
            % Create labels (gesture indices are 1 to 17)
            labels = [labels; repmat(igesture, num_windows_per_seq, 1)];
           
            % Extract windows for this participant & gesture
            windowed_data = FeatSet{isub, igesture};  % Cell array of size (num_windows_per_seq x 1)
            
            % Compute 25th percentile for each channel across ALL windows of this gesture-trial sequence
           % all_windows_matrix = cell2mat(windowed_data); % Convert cell array to matrix (num_windows * 409 x 8)
          %  percentile_25 = prctile(all_windows_matrix, 25, 1); % Shape: (1 x 8) -> 25th percentile per channel
            
            % Iterate through each window for rest classification
            %for i = 1:length(windowed_data)  
               % window = windowed_data{i}; % Shape: (409 x 8)
            
                % Check % of values below the 25th percentile per channel
                % Matrix/array operations
               % below_threshold = sum(window < percentile_25, 1) / 409;  % Fraction of values per channel
            
                % If **ALL** 8 channels have at least 60% of values below their own 25th percentile → mark as `Rest`
             %   if all(below_threshold >= 0.6)
             %       labels((end - num_windows_per_seq) + i) = 17;  % Overwrite label with `Rest`
             %       disp(['overwrote in ', num2str(igesture)]);
              %  end
          %  end

            disp(['Labeled a seq ', num2str(igesture)]);

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
        if isempty(CombinedFeatSet{1, 1}) 
            CombinedFeatSet{1, 1} = combined_windows;
        else % Combining concatenated data w/ previous session in CombinedFeatSet, thats why we need if else here.
            CombinedFeatSet{1, 1} = [CombinedFeatSet{1, 1}; combined_windows]; 
        end
        
        disp(['Gesture ', num2str(igesture), ' combined across all participants.'])
    end
    FeatSet = {}; % Freeing memory from previous things before processing new loop

end

% Save the combined dataset from all sessions into 1 file (if u want
% seperate files move this into session loop & add session variable into 
% file name below and get rid of conditional above).
output_file = ['Feature Extracted BM' filesep 'Combined_Forearm_Data.mat'];
output_labs = ['Feature Extracted BM' filesep 'labels.mat'];
disp(['Saving combined data to ', output_file]);
save(output_file, 'CombinedFeatSet', '-v7.3');
save(output_labs, 'labels', '-v7.3');

disp('Data processing and combination into 1 row complete.');
