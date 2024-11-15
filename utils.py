import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

cuda = torch.cuda.is_available()

#GPU Setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    cuda = True
    print('Using: ' + str(torch.cuda.get_device_name(device)))
else:
    cuda = False
    print('Using: CPU')

#function to select the tsv file
def select_random_tsv(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.tsv'):
                return os.path.join(root, file)
    return None

def plot_eeg_signals(epoch, noisy_signal, denoised_signal, clean_signal):

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(noisy_signal.cpu().detach().numpy().squeeze(), label='Noisy')
    plt.title(f"Noisy EEG (Epoch {epoch})")
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')

    plt.subplot(1, 3, 2)
    plt.plot(denoised_signal.cpu().detach().numpy().squeeze(), label='Denoised')
    plt.title(f"Denoised EEG (Epoch {epoch})")
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')

    plt.subplot(1, 3, 3)
    plt.plot(clean_signal.cpu().detach().numpy().squeeze(), label='Clean')
    plt.title(f"Clean EEG (Epoch {epoch})")
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()
    
    # Pause to allow the plot to update without blocking execution
    plt.pause(0.5)  # Adjust the pause duration if needed

    # Optional: Close the current figure if you want to free up memory after plotting
    #plt.close()

# Iterate through each file in the directory
def MyDataset_unhealthy(directory):
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    dataframes = []
    labels = []

    for file in csv_files:
        # Extract label from the filename
        label = os.path.splitext(file)[0]
        read_csv = os.path.join(directory, file)

        # Read the csv file including the last column
        csv_file = pd.read_csv(read_csv, header=None)

        # Take a random tsv file from the directory
        random_tsv_file = select_random_tsv(directory)

        # Read the tsv file
        tsv_df = pd.read_csv(random_tsv_file, sep='\t', header=None)

        # Exclude the first row and take the first column
        first_column_tsv = tsv_df.iloc[1:, 0]
        first_column_tsv_truncated = first_column_tsv[:len(csv_file.iloc[0, :])]

        # Replace the first row of the CSV DataFrame with the first column of the TSV DataFrame
        csv_file.iloc[0, :] = first_column_tsv_truncated.values

        # Filter rows where the last column contains 'S1', 'S4', 'S8', 'S10'
        filtered_csv_file = csv_file[csv_file.iloc[:, -1].isin(['S 1', 'S 4', 'S 8', 'S 10'])]

        # Ensure the first row is included in the filtered DataFrame
        filtered_csv_file = pd.concat([csv_file.iloc[[0]], filtered_csv_file])

        # Remove the last column
        filtered_csv_file = filtered_csv_file.iloc[:, :-1]

        if len(filtered_csv_file) >= 2:
            dataframes.append(filtered_csv_file)
            labels.append(label)

    return dataframes, labels

def MyDataset_healthy(directory):
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    dataframes = []
    labels = []

    for file in csv_files:
        # Extract label from the filename
        label = os.path.splitext(file)[0]
        read_csv = os.path.join(directory, file)

        # Read the csv file including the last column
        csv_file = pd.read_csv(read_csv, header=None)

        # Take a random tsv file from the directory
        random_tsv_file = select_random_tsv(directory)

        # Read the tsv file
        tsv_df = pd.read_csv(random_tsv_file, sep='\t', header=None)

        # Exclude the first row and take the first column
        first_column_tsv = tsv_df.iloc[1:, 0]
        first_column_tsv_truncated = first_column_tsv[:len(csv_file.iloc[0, :])]

        # Replace the first row of the CSV DataFrame with the first column of the TSV DataFrame
        csv_file.iloc[0, :-1] = first_column_tsv_truncated.values

        # Filter rows where the last column contains 'TASK1T1' or 'TASK1T2'
        filtered_csv_file = csv_file[csv_file.iloc[:, -1].isin(['TASK1T1', 'TASK1T2'])]

        # Ensure the first row is included in the filtered DataFrame
        filtered_csv_file = pd.concat([csv_file.iloc[[0]], filtered_csv_file])

        # Remove the last column
        filtered_csv_file = filtered_csv_file.iloc[:, :-1]

        if len(filtered_csv_file) >= 2:
            dataframes.append(filtered_csv_file)
            labels.append(label)

    return dataframes, labels

def print_dataframes_with_labels(dataframes, labels):
    #Print each dataframe along with its corresponding label.
    if len(dataframes) != len(labels):
        raise ValueError("The number of dataframes and labels must be equal.")

    for dataframe, label in zip(dataframes, labels):
        print("Label:", label)
        print("Dataframe:")
        print(dataframe)
        print("------------------")
        
def score_and_find_common(list1, list2):
    common_elements = []

    # Convert all elements in list2 to lowercase for case-insensitive comparison
    lower_list2 = [element.lower() for element in list2]

    for element in list1:
        if element.lower() in lower_list2:
            common_elements.append(element)

    return common_elements

def filter_dataframe(df, common_channels):
    # Convert the common channels list to lowercase
    common_channels_lower = [channel.lower() for channel in common_channels]

    # Filter columns based on lowercase channel names
    df_filtered = df.loc[:, df.iloc[0].str.lower().isin(common_channels_lower)]

    return df_filtered

def df_to_tensor_with_skip(df, label=None):
    # Skip the first row of the DataFrame
    df = df.iloc[1:]
    # Convert DataFrame to NumPy array with float32 dtype
    array = np.array(df, dtype=np.float32)
    # Convert NumPy array to tensor
    tensor = torch.tensor(array, device=device)

    return tensor, label

#Convert tensor with label
def convert_to_tensors_with_labels_with_skip(dataframes, labels):
    tensors_with_labels = []
    for df, label in zip(dataframes, labels):
        tensor, labeled_tensor = df_to_tensor_with_skip(df, label=label)
        tensors_with_labels.append((tensor, labeled_tensor))
    return tensors_with_labels

def df_to_tensor(df, label=None):
    # Convert DataFrame to NumPy array with float32 dtype
    array = np.array(df, dtype=np.float32)

    # Convert NumPy array to tensor
    tensor = torch.tensor(array)

    return tensor, label

#Convert tensor with label
def convert_to_tensors_with_labels(dataframes, labels):
    tensors_with_labels = []
    for df, label in zip(dataframes, labels):
        tensor, labeled_tensor = df_to_tensor(df, label=label)
        tensors_with_labels.append((tensor, labeled_tensor))
    return tensors_with_labels

#function in order to normalize the data
def normalize_dataframes(dataframes_list):

    normalized_dataframes = []

    for df in dataframes_list:
        # Apply Min-Max scaling to normalize values between [0, 1]
        normalized_df = (df - df.min()) / (df.max() - df.min())
        normalized_dataframes.append(normalized_df)

    return normalized_dataframes

Fs = 128
Ts = 1.0/Fs

def plot_random_signal(tensor_list, title):
    num_signals = len(tensor_list)
    num_cols = 5  # Number of columns in each row
    num_rows = (num_signals + num_cols - 1) // num_cols  # Calculate number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5*num_rows))
    fig.suptitle(title)

    for i, (tensor, label) in enumerate(tensor_list):
        row = i // num_cols
        col = i % num_cols
        sig_len = len(tensor[:, 0].detach().cpu().numpy()) / Fs
        t = np.arange(0, sig_len, Ts)

        ax = axes[row, col]
        ax.plot(t, tensor[:, 0].detach().cpu().numpy())
        ax.set_title(label)
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()
    
def filter_tensors(labeled_tensors, threshold_shape):
    filtered_tensors = []
    for tensor, label in labeled_tensors:
        if tensor.shape[0] >= threshold_shape[0] and tensor.shape[1] >= threshold_shape[1]:
            # If tensor shape is larger than threshold, resize or crop it
            tensor_filtered = tensor[:threshold_shape[0], :threshold_shape[1]]

            filtered_tensors.append((tensor_filtered, label))

    return filtered_tensors