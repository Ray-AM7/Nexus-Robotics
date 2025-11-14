% Load the combined dataset
load('Feature Extracted BM/Combined_Forearm_Data.mat');
load('Feature Extracted BM/labels.mat');  % Load your labels

% Define split ratio (70% training, 30% testing, or any other ratio)
train_size = 340646;
test_size = 170323;

% Initialize storage for train and test data
train_data = [];
test_data = [];
train_labels = [];
test_labels = [];

% train/test split

windows = CombinedFeatSet{1, 1}
train_data = [train_data; windows(1:train_size)]; 
train_labels = [train_labels; labels(1:train_size)]; 

test_data = [test_data; windows(train_size+1:train_size+test_size)]; 
test_labels = [test_labels; labels(train_size+1:train_size+test_size)];  
disp(test_labels(1:20));

% Convert cell arrays into matrices for TensorFlow compatibility
train_data_matrix = cell2mat(train_data);
test_data_matrix = cell2mat(test_data);

% Reshape data into a TensorFlow-friendly format: (samples, time_steps, channels)
train_data_reshaped = reshape(train_data_matrix, train_size, 409, 8);
test_data_reshaped = reshape(test_data_matrix, test_size, 409, 8);

% Save the reshaped data and labels as .mat files
save('train_data.mat', 'train_data_reshaped', '-v7.3');
save('test_data.mat', 'test_data_reshaped', '-v7.3');
save('train_labels.mat', 'train_labels', '-v7.3');
save('test_labels.mat', 'test_labels', '-v7.3');

disp('Train/Test split & labels saved successfully.');
