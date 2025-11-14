% labeling data script, integrate into above code after the windowing where I commented.
% Load the combined dataset
% load('Feature Extracted BM/Combined_Forearm_Data.mat');

% Define split indices
train_size = 20038;  % Number of windows for training
test_size = 10019;   % Number of windows for testing

% Initialize storage for train and test data
train_data = cell(1, 17);
test_data = cell(1, 17);
train_labels = [];
test_labels = [];

for igesture = 1:17
    % Extract the 30,057 windows for this gesture
    all_windows = CombinedFeatSet{1, igesture}; % Shape: 30057 x 1 (each cell/element withiin is 409x8)
    

    % Split the data into train (first 20038) and test (last 10019).
    % Could've done cat here or in windowing step but whatever it all works
    train_data{igesture} = all_windows(1:train_size);
    test_data{igesture} = all_windows(train_size+1:end);
    
    % Create labels (gesture indices are 1 to 17)
    train_labels = [train_labels; repmat(igesture, train_size, 1)];
    test_labels = [test_labels; repmat(igesture, test_size, 1)];

    disp(['Processed Gesture ', num2str(igesture)]);
end

% Convert the cell arrays to matrices for export to Python
% This concatenates the windows of each gesture in a set into one thing 
% btw. So just bringing everything out Could've done this before.
train_data_matrix = cat(1, train_data{:});  % Each element Shape: 409 x 8
test_data_matrix = cat(1, test_data{:});    % Each element Shape: 409 x 8
disp('Brought window arrays out and made huge matrices for train and test sets')

% Reshape each cell array into the correct format for TensorFlow: (samples, time_steps, channels)
train_data_reshaped = zeros(train_size * 17, 409, 8);
test_data_reshaped = zeros(test_size * 17, 409, 8);
disp('Created new 3D arrays for bringing out raw data into 1 array for each set')

for i = 1:(train_size * 17)
    % Convert cell to matrix
    train_data_reshaped(i, :, :) = train_data_matrix{i}; 
    % Each element is now raw data in 3D array of size 340646 x 409 x 8
end
disp('Training set 3D array reshaped (340646 x 409 x 8)')

for i = 1:(test_size * 17)
    % Convert cell to matrix
    test_data_reshaped(i, :, :) = test_data_matrix{i}; % Convert cell to matrix
    % Each element is now raw data in 3D array of size 170323 x 409 x 8
end
disp('Testing set 3D array reshaped (170323 x 409 x 8)')


% Save the reshaped data and labels as .mat files

% Might feel like forever or it broke or something, but patience
% for some reason file explorer isn't showing data being written until the 
% whole file is done, pretty sure this didn't happen w/ the other mat files 
% so idk what's up.
% depending on hardware and specifically ur hardrive, time may vary.
save('16_train_data.mat', 'train_data_reshaped', '-v7.3');
disp('X_train saved successfully.');

save('16_test_data.mat', 'test_data_reshaped', '-v7.3');
disp('X_test saved successfully.');

save('16_train_labels.mat', 'train_labels', '-v7.3');
disp('Y_train saved successfully.');

save('16_test_labels.mat', 'test_labels', '-v7.3');
disp('Y_test saved successfully.');

disp('Train/Test split & labels saved successfully.');