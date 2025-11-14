import numpy as np
import h5py


# Function to load HDF5-based .mat files with memory-efficient processing
def load_hdf5_mat_lazy(file_path, variable_name):
    with h5py.File(file_path, 'r') as f:
        dataset = f[variable_name]
        shape = dataset.shape  # Get shape without loading entire data
        dtype = dataset.dtype  # Get datatype
        
        # Read data in chunks to save memory and convert to numpy
        data = np.empty(shape, dtype=dtype)
        dataset.read_direct(data)  # Read directly to avoid memory duplication
        
        # MATLAB stores as column-major, so transpose to row-major (NumPy default)
        data = np.transpose(data)
        
        # Replace NaN values with 0.0 (handling in-place to save memory)
        np.nan_to_num(data, copy=False, nan=0.0)
        
    return data

# Load labels with flattening
def load_labels_lazy(file_path, variable_name):
    with h5py.File(file_path, 'r') as f:
        dataset = f[variable_name]
        data = np.empty(dataset.shape, dtype=dataset.dtype)
        dataset.read_direct(data)
    return data.flatten()

# Load training and testing datasets lazily
dataset = None
train_data = load_hdf5_mat_lazy('train_data.mat', 'train_data_reshaped')
np.save('train_data.npy', train_data)
print("Train Data Loaded. Shape:", train_data.shape)  # Expected (340646, 409, 8)
del dataset
del train_data

dataset = None
test_data = load_hdf5_mat_lazy('test_data.mat', 'test_data_reshaped')
np.save('test_data.npy', test_data)
print("Test Data Loaded. Shape:", test_data.shape)    # Expected (170323, 409, 8)
del dataset
del test_data

dataset = None
train_labels = load_labels_lazy('train_labels.mat', 'train_labels')
test_labels = load_labels_lazy('test_labels.mat', 'test_labels')

# Convert labels from 1-17 to 0-16 (for TensorFlow sparse categorical crossentropy)
train_labels -= 1
test_labels -= 1

np.save('train_labels.npy', train_labels)
np.save('test_labels.npy', test_labels)

# Clear memory by deleting unnecessary variables
del dataset
del train_labels, test_labels

print("Data processed and saved as NumPy files successfully.")
