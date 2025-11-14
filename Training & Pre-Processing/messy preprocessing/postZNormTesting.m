clear
addpath(genpath([pwd filesep 'Output BM']))

% Define parameters
fs = 2048;                   % Sampling frequency
NSUB = length(dir([pwd filesep 'Output BM' filesep 'Session1_converted'])) - 2; % Number of participants
NSESSION = length(dir([pwd filesep 'Output BM'])) - 2; % Number of sessions
NGESTURE = 17;              % Total number of gestures w/ rest
NTRIALS = 3;                % Total number of trials
num_windows_per_seq = 233;
% Variables for processing
a1 = [];                     % Temporary variable to merge trials of a specific contraction
a2 = [];                     % Temporary variable for gesture-wise data
CompleteSet = [];            % Store final gestures and subjects for the session
CombinedFeatSet = cell(1, NGESTURE);
labels = [];

%% Define output folder
if ~exist('Feature Extracted BM', 'dir')
    mkdir('Feature Extracted BM')
else
    disp('nah')
end

CompleteSet = [];  
for isub = 30:30
    fileName = ['session' num2str(1) '_participant', num2str(isub), '.mat'];
    temp_load_forearm = load(fileName, 'DATA_FOREARM');
    datafile = temp_load_forearm.DATA_FOREARM;

    for igesture = 9:9 % Process only the first gesture
        for itrial = 2:NTRIALS
            a1 = [a1; datafile{itrial, igesture}];
        end
        a2 = [a2, {a1}];
        a1 = [];
    end

    CompleteSet = [CompleteSet; a2];
    a2 = [];
    disp(['Loaded: Session ', num2str(1), ', Participant ', num2str(isub)])
end

rmpath(genpath([pwd filesep 'Output BM']))         % Save memory

%% Segmentation and Processing
FeatSet = {}; % Store segmented raw data
for igesture = 9:9
    for isub = 30:30 % switched to gesture major for easier labels

        OneSet = CompleteSet{1, 1}';       % accessed indexed cell w/ array w/ Shape = samples x channels
        post_process = [];

        % Replace NaN values in OneSet with zeros before normalization
        OneSet(isnan(OneSet)) = 0;
        
        % Monopolar average referencing
        for ichannel_2 = 1:8
            temp2 = OneSet(1:8, :); % Proximal sensor channels, more important for amputees
            post_process(ichannel_2, :) = OneSet(ichannel_2, :) - mean(temp2, 1);
        end
        
        % Apply Z-Score Normalization
        for ichannel_2 = 1:8
            mean_val = mean(post_process(ichannel_2, :));
            std_val = std(post_process(ichannel_2, :));

            % Avoid division by zero
            if std_val == 0
                std_val = 1;
            end

            % Apply Z-score normalization
            post_process(ichannel_2, :) = (post_process(ichannel_2, :) - mean_val) / std_val;
        end

    end
end

figure; % Open a new figure window
plot(post_process'); % Each column represents a channel
xlabel('Time Points');
ylabel('Amplitude (mV)');
title('EMG Signal for Trial 1, Gesture 1 (Forearm)');
legend('Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', ...
       'Channel 5', 'Channel 6', 'Channel 7', 'Channel 8');
grid on;


figure; % Open a new figure window
for channel = 1:8 % Loop through the first 8 channels
    subplot(8, 1, channel); % Create an 8-row, 1-column subplot layout
    plot(post_process(channel, :)); % Plot each row (now a channel)
    title(['Channel ', num2str(channel)]);
    xlabel('Time Points');
    ylabel('Amplitude (mV)');
    ylim([-4, 4]); % Adjust based on the range of your data
    grid on;
end
