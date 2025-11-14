import os
import shutil
import numpy as np
import pandas as pd  # For interpolation
from scipy.io import loadmat
from scipy.signal import decimate, butter, filtfilt
import pywt

# ----- Helper Functions for Filtering & Transformations -----
def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply a low-pass Butterworth filter.
    
    Parameters:
        data : np.ndarray
            Input data (assumed to be in shape (channels, timesteps)).
        cutoff : float
            Cutoff frequency in Hz.
        fs : float
            Sampling frequency in Hz.
        order : int
            Filter order.
            
    Returns:
        y : np.ndarray
            Filtered data, same shape as input.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Apply filter along the time axis (axis=1)
    y = filtfilt(b, a, data, axis=1)
    return y

def segmentEMG(Sig, WTime, STime, MTime, SFreq, MLabel):
    """
    Segment signal into windows and assign labels.
    
    Parameters:
        Sig : np.ndarray
            Signal with rows as samples and columns as channels.
        WTime : float
            Window time in seconds.
        STime : float
            Sliding window time in seconds.
        MTime : float
            Motion duration in seconds.
        SFreq : int or float
            Sampling frequency.
        MLabel : array-like
            Motion labels for each motion.
            
    Returns:
        ESig : np.ndarray
            3D array with shape (channels, winLen, total_segments).
            (Each segment is produced as channels x winLen.)
        ELabel : np.ndarray
            2D array with shape (total_segments, 1) of labels.
    """
    winLen = int(np.floor(WTime * SFreq))
    sldLen = int(np.floor(STime * SFreq))
    monLen = int(np.floor(MTime * SFreq))
    
    k1 = len(MLabel)  # number of motions
    k2 = int(np.floor((monLen - winLen) / sldLen))
    
    segments = []  # Each segment: channels x winLen
    labels = []
    
    for i in range(k1):
        for j in range(k2 + 1):
            start_idx = int(round(i * monLen + j * sldLen))
            end_idx = int(round(i * monLen + j * sldLen + winLen))
            # Sig is assumed to be (samples, channels); transpose to (channels, winLen)
            segment = Sig[start_idx:end_idx, :].T  
            segments.append(segment)
            labels.append(MLabel[i])
    
    ESig = np.stack(segments, axis=2)  # shape: (channels, winLen, total_segments)
    ELabel = np.array(labels).reshape(-1, 1)
    return ESig, ELabel

# ----- Optional Transformation Functions -----
def compute_fft(window):
    """
    Compute the FFT magnitude for a window.
    
    Parameters:
        window : np.ndarray
            2D array of shape (winLen, channels) in time domain.
    Returns:
        fft_window_mag : np.ndarray
            2D array of shape (winLen_fft, channels), where winLen_fft = winLen//2+1.
    """
    fft_window = np.fft.rfft(window, axis=0)
    fft_window_mag = np.abs(fft_window)
    return fft_window_mag

def compute_stft(window, fs, nperseg=64):
    """
    Compute the STFT magnitude for a window.
    
    Parameters:
        window : np.ndarray
            2D array of shape (winLen, channels).
        fs : float
            Sampling frequency.
        nperseg : int
            Length of each segment for STFT.
    Returns:
        stft_results : np.ndarray
            3D array of shape (frequencies, times, channels).
    """
    import scipy.signal as sps
    stft_results = []
    for ch in range(window.shape[1]):
        f, t, Zxx = sps.stft(window[:, ch], fs=fs, nperseg=nperseg)
        stft_results.append(np.abs(Zxx))
    stft_results = np.stack(stft_results, axis=-1)
    return stft_results

def compute_wavelet(window, fs, scales=np.arange(1,31), wavelet='morl'):
    """
    Compute the continuous wavelet transform (CWT) magnitude for a window.
    
    Parameters:
        window : np.ndarray
            2D array of shape (winLen, channels).
        fs : float
            Sampling frequency.
        scales : array-like
            Scales to use for CWT.
        wavelet : str
            Wavelet name.
    Returns:
        coeffs_all : np.ndarray
            3D array of shape (num_scales, winLen, channels).
    """
    coeffs_all = []
    for ch in range(window.shape[1]):
        coeffs, freqs = pywt.cwt(window[:, ch], scales, wavelet, sampling_period=1/fs)
        coeffs_all.append(np.abs(coeffs))
    coeffs_all = np.stack(coeffs_all, axis=-1)
    return coeffs_all

# ----- Label Mapping ----- 
# New mapping: 
# [0,15] -> 0; [1,3,12,14] -> 1; [2,6,8] -> 2; [4,5,9] -> 3; [10] -> 4; [11] -> 5; [13] -> 6; [7] -> 7; [16] -> 8
# The purpose of this is to be able to combine similar gestures and determine correct bionic output using other alg's & data
# while still having high AI accuracy, rather than having low accuracies on similar gestures when using only proximal forearm data.
# These remappings were deemed necessary specifically for this dataset. (multi-day emg etc etc. paper)

label_mapping = {
    0: 0, 15: 0,
    1: 12, 3: 13, 12: 8, 14: 10,
    2: 1, 6: 15, 8: 4,
    4: 14, 5: 2, 9: 5,
    10: 6,
    11: 7,
    13: 9,
    7: 3,
    16: 11
}

# 0,2,5,7,8,9,10,11,12,13,14,16, 1, 3, 4, 6
# 0,1,2,3,4,5,6, 7, 8, 9, 10,11,12,13,14,15


# label_mapping = {
#     0: 0, 15: 0,
#     1: 1, 3: 1, 12: 1, 14: 1,
#     2: 2, 6: 2, 8: 2,
#     4: 3, 5: 3, 9: 3,
#     10: 4,
#     11: 5,
#     13: 6,
#     7: 7,
#     16: 8
# }

# ---------------------------
# Main Processing Code
# ---------------------------
def main():
    # ---------------------------
    # Define directories and parameters
    # ---------------------------
    base_dir = os.path.join(os.getcwd(), "gesture-recognition-and-biometrics-electromyogram-grabmyo-1.0.1", "Output BM")
    session1_converted_dir = os.path.join(base_dir, "Session1_converted")
    fs = 2048      # Original sampling frequency
    dfs = 1024   # Downsampled frequency
    q = fs // dfs  # Decimation factor
    
    # Count sessions and subjects (based on folder contents)
    NSESSION = len(os.listdir(base_dir))          # Total number of sessions
    NSUB = len(os.listdir(session1_converted_dir))  # Total number of subjects
    NGESTURE = 17  # Gestures (0-indexed: 0 to 16)
    NTRIALS = 7    # Trials per gesture
    nCh = 8        # Use first 8 channels
    
    print(f"Found {NSUB} participants and {NSESSION} sessions.")
    
    # ---------------------------
    # Prepare combined structures per gesture.
    # Each element (index g) will accumulate segmented windows for gesture g.
    CombinedFeatSet = [ [] for _ in range(NGESTURE) ]
    CombinedLabelSet = [ [] for _ in range(NGESTURE) ]
    
    # ---------------------------
    # Create (or overwrite) the output folder
    # ---------------------------
    output_folder = os.path.join(os.getcwd(), "gesture-recognition-and-biometrics-electromyogram-grabmyo-1.0.1", "Feature Extracted BM")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        print("Overwriting existing output folder.")
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)
    
    # ---------------------------
    # Process each session:
    # 1. Flatten 7 trials per gesture (with interpolation and downsampling)
    # 2. Perform referencing, envelope extraction, and segmentation.
    # CompleteSet: 2D list (subjects x gestures).
    # ---------------------------
    for isession in range(1, NSESSION + 1):
        CompleteSet = []  # Reset for each session.
        for isub in range(NSUB):
            file_name = f"session{isession}_participant{isub+1}.mat"
            file_path = os.path.join(base_dir, f"Session{isession}_converted", file_name)
            mat_data = loadmat(file_path)
            datafile = mat_data["DATA_FOREARM"]
            a2 = []  # Will store concatenated trials per gesture for this subject.
            for igesture in range(NGESTURE):
                a1_list = []  # Accumulate the 7 trials for this gesture.
                for itrial in range(NTRIALS):
                    trial_data = np.array(datafile[itrial, igesture])
                    # Interpolate NaNs for each channel.
                    for ch in range(trial_data.shape[1]):
                        series = pd.Series(trial_data[:, ch])
                        series_interp = series.interpolate(method='linear')
                        series_interp = series_interp.fillna(method='bfill').fillna(method='ffill')
                        trial_data[:, ch] = series_interp.values
                    # Downsample trial from fs to dfs.
                    trial_data = decimate(trial_data, q, axis=0, zero_phase=True)
                    a1_list.append(trial_data)
                # Vertically concatenate the 7 trials.
                a1 = np.vstack(a1_list)
                a2.append(a1)
            CompleteSet.append(a2)
            print(f"Loaded: Session {isession}, Participant {isub+1}")
        
        # For each subject and each gesture, perform referencing, envelope extraction, and segmentation.
        for subj in range(NSUB):
            for gesture in range(NGESTURE):
                # Get concatenated data for this subject and gesture.
                # Original data: shape (samples, channels); transpose to (channels, samples)
                OneSet = CompleteSet[subj][gesture].T
                # Restrict to first nCh channels.
                OneSet = OneSet[:nCh, :]
                # post_process = np.zeros((nCh, OneSet.shape[1])) # uncomment for average referencing
                post_process = OneSet # comment out for average referencing

                # Monopolar average referencing.
                # for ichannel in range(nCh):
                #     temp2 = OneSet[:nCh, :]
                #     post_process[ichannel, :] = OneSet[ichannel, :] - np.mean(temp2, axis=0)

                # Bipolar average referencing:
                # for i in range(nCh):
                #     if i == 0:
                #         # For channel 0, use the last channel and channel 1
                #         temp2 = np.mean(np.vstack([OneSet[-1, :], OneSet[1, :]]), axis=0)
                #     elif i == nCh - 1:
                #         # For the last channel, use the second last channel and channel 0
                #         temp2 = np.mean(np.vstack([OneSet[nCh - 2, :], OneSet[0, :]]), axis=0)
                #     else:
                #         # For other channels, use the previous and next channels
                #         temp2 = np.mean(np.vstack([OneSet[i - 1, :], OneSet[i + 1, :]]), axis=0)
                    
                #     post_process[i, :] = OneSet[i, :] - temp2
                
                # ----- Envelope Extraction -----
                # Rectify (absolute value) the referenced signal.
                # envelope = np.abs(post_process)

                # # Apply a low-pass Butterworth filter to obtain the envelope.
                # cutoff = 30  # Cutoff frequency in Hz (adjust as needed)
                # envelope = butter_lowpass_filter(envelope, cutoff, dfs, order=4)
                # --------------------------------
                
                # Define motion label (0-indexed already).
                new_label = label_mapping[gesture]
                MLabel = [new_label]

                #MLabel = [gesture]

                # Segment the envelope signal.
                # The segmentation function expects input in shape (samples, channels); envelope.T passes (samples, channels).
                # segData, segLabels = segmentEMG(envelope.T, 0.2, 0.1, NTRIALS * 5, dfs, MLabel) # if getting envelope
                segData, segLabels = segmentEMG(post_process.T, 0.2, 0.1, NTRIALS * 5, dfs, MLabel) # if u want to use raw-ish data without envelope
                # segData: shape (channels, winLen, num_windows). Transpose to (num_windows, winLen, channels)
                segData = np.transpose(segData, (2, 1, 0))
                windows = [segData[i, :, :] for i in range(segData.shape[0])]
                labels = segLabels.flatten().tolist()
                
                # Accumulate windows and labels for this gesture.
                CombinedFeatSet[gesture].extend(windows)
                CombinedLabelSet[gesture].extend(labels)
    
    # ---------------------------
    # Optional: Apply a transformation to each window.
    # Choose by setting transform_type.
    # Options:
    #   "fft"      → FFT magnitude: each window becomes (winLen_fft, channels)
    #   "stft"     → STFT magnitude: each window becomes (frequencies, times, channels)
    #   "wavelet"  → Wavelet transform: each window becomes (num_scales, winLen, channels)
    #   "none"     → Use original time-domain window: shape (winLen, channels)
    # ---------------------------
    transform_type = "none"  # Change this to "stft", "wavelet", or "none" as desired.
    CombinedFeatSet_transformed = [ [] for _ in range(NGESTURE) ]
    
    if transform_type == "fft": # time steps will get roughly halved btw because of how fft works
        for g in range(NGESTURE):
            for window in CombinedFeatSet[g]:
                # window shape: (winLen, channels)
                fft_window_mag = compute_fft(window)  # shape: (winLen_fft, channels)
                CombinedFeatSet_transformed[g].append(fft_window_mag)
    elif transform_type == "stft":
        for g in range(NGESTURE):
            for window in CombinedFeatSet[g]:
                stft_window = compute_stft(window, dfs, nperseg=64)  # shape: (frequencies, times, channels)
                CombinedFeatSet_transformed[g].append(stft_window)
    elif transform_type == "wavelet":
        for g in range(NGESTURE):
            for window in CombinedFeatSet[g]:
                wavelet_window = compute_wavelet(window, dfs, scales=np.arange(1,31), wavelet='morl')  # shape: (num_scales, winLen, channels)
                CombinedFeatSet_transformed[g].append(wavelet_window)
    else:
        CombinedFeatSet_transformed = CombinedFeatSet

    # Use transformed features.
    CombinedFeatSet = CombinedFeatSet_transformed
    del CombinedFeatSet_transformed


            # ----- Exclusion & Combination -----


    # First: combine gestures, so relabel all of the windows in gesture 15 as gesture 0 
    
    # Second: Exclude unwanted gestures. (e.g., if you want to remove gestures 1, 3, 4, and 6)

    excludeGestures = [1,3,4,6]

    # Define which gesture indices to keep (for instance, remove indices 5 and 6)
    keep_indices = [i for i in range(NGESTURE) if i not in excludeGestures]
    # print("Keeping gestures:", keep_indices)

    # Filter out the unwanted gestures:
    CombinedFeatSet = [CombinedFeatSet[i] for i in keep_indices]
    CombinedLabelSet = [CombinedLabelSet[i] for i in keep_indices]

    # Update NGESTURE to reflect the new number of gestures.
    NGESTURE = len(keep_indices)



    # ---------------------------
    # At this point, CombinedFeatSet and CombinedLabelSet are lists (length = NGESTURE).
    # Each element is a list containing all windows (shape: timesteps x channels)
    # and corresponding labels from all sessions/participants for that gesture.
    # Now, for each gesture, optionally shuffle the windows, then split into train and test.
    # ---------------------------
    train_x_list, train_y_list = [], []
    test_x_list, test_y_list = [], []
    
    for g in range(NGESTURE):
        gesture_windows = np.array(CombinedFeatSet[g])  # shape: (num_windows, timesteps, channels)
        gesture_labels = np.array(CombinedLabelSet[g]).reshape(-1)  # shape: (num_windows,)
        
        # Optionally shuffle (uncomment if desired):
        # Uncomment the following lines to shuffle before splitting:
        perm = np.random.permutation(gesture_windows.shape[0])
        gesture_windows = gesture_windows[perm]
        gesture_labels = gesture_labels[perm]
        
        # Split indices (adjust numbers if needed)
        # Taking the first 20038 windows for training and the next 10019 for testing.
        train_windows = gesture_windows[:20038]
        train_labels = gesture_labels[:20038]
        test_windows = gesture_windows[20038:20038+10019]
        test_labels = gesture_labels[20038:20038+10019]
        
        train_x_list.append(train_windows)
        train_y_list.append(train_labels)
        test_x_list.append(test_windows)
        test_y_list.append(test_labels)
    
    X_train = np.concatenate(train_x_list, axis=0)
    Y_train = np.concatenate(train_y_list, axis=0)
    X_test = np.concatenate(test_x_list, axis=0)
    Y_test = np.concatenate(test_y_list, axis=0)
    
    # Optionally, shuffle the combined training and testing sets.
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    indices = np.arange(len(X_test))
    np.random.shuffle(indices)
    X_test = X_test[indices]
    Y_test = Y_test[indices]

    print(f"Training data shape: {X_train.shape}, Labels: {Y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, Labels: {Y_test.shape}")
    print("Unique training labels:", np.unique(Y_train))
    print("Unique testing labels:", np.unique(Y_test))
    
    # ---------------------------
    # Save the final datasets as NumPy files.
    # ---------------------------
    np.save(os.path.join(output_folder, "X_train.npy"), X_train)
    np.save(os.path.join(output_folder, "Y_train.npy"), Y_train)
    np.save(os.path.join(output_folder, "X_test.npy"), X_test)
    np.save(os.path.join(output_folder, "Y_test.npy"), Y_test)
    
    print("Data processing, transformation, splitting, and saving complete. NumPy files saved.")

if __name__ == "__main__":
    main()
