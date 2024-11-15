import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mne
from utils import *

cuda = torch.cuda.is_available()

#GPU Setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    cuda = True
    print('Using: ' + str(torch.cuda.get_device_name(device)))
else:
    cuda = False
    print('Using: CPU')

#Loading Dataset
dataset_unhealthy=r'C:\Users\franc\Desktop\AI & Robotics\Elective in AI\BCI_Signal_Reconstruction\Dataset\EEG_Dataset_Final\Unhealthy'
dataset_healthy=r'C:\Users\franc\Desktop\AI & Robotics\Elective in AI\BCI_Signal_Reconstruction\Dataset\EEG_Dataset_Final\Healthy'

print(dataset_unhealthy)
print(dataset_healthy)


"""Healthy & Unhealthy Dataframe"""

#Transform the dataset in dataframe
unhealthy_dataframe,labels_unhealthy = MyDataset_unhealthy(dataset_unhealthy)
healthy_dataframe,labels_healthy = MyDataset_healthy(dataset_healthy)

#Print each Dataframe
print_dataframes_with_labels(unhealthy_dataframe, labels_unhealthy)
print_dataframes_with_labels(healthy_dataframe, labels_healthy)

"""Filter dataframe with the common channels in each Dataset"""

def remove_channels(dataframe):
  dataframes=[]
  tsv_healthy=select_random_tsv(dataset_healthy)
  tsv_unhealthy= select_random_tsv(dataset_unhealthy)
  common_channels=[]
  #Read the tsv file
  tsv_df_healthy = pd.read_csv(tsv_healthy, sep='\t', header=None)
  tsv_df_unhealthy = pd.read_csv(tsv_unhealthy, sep='\t', header=None)
  # Exclude the first row
  first_column_healthy = tsv_df_healthy.iloc[1:, 0]
  first_column_unhealthy = tsv_df_unhealthy.iloc[1:, 0]
  common_channels= score_and_find_common(first_column_healthy, first_column_unhealthy)
  for df in dataframe:
    # Filter the DataFrame to keep only the desired columns
    df_filtered = filter_dataframe(df, common_channels)
    dataframes.append(df_filtered)

  return dataframes

unhealthy_dataframe=remove_channels(unhealthy_dataframe)
healthy_dataframe=remove_channels(healthy_dataframe)

print_dataframes_with_labels(unhealthy_dataframe, labels_unhealthy)
print_dataframes_with_labels(healthy_dataframe, labels_healthy)

"""Plot the normal signal"""

unhealthy_tensors = convert_to_tensors_with_labels_with_skip(unhealthy_dataframe, labels_unhealthy[:len(unhealthy_dataframe)])
healthy_tensors = convert_to_tensors_with_labels_with_skip(healthy_dataframe, labels_healthy[:len(healthy_dataframe)])

plot_random_signal(unhealthy_tensors, 'Unhealthy signal')
plot_random_signal(healthy_tensors, 'Healthy signal')

"""Preprocessing signals"""
fs = 128  # Sampling frequency in Hz
low_freq = 8  #Hz
high_freq = 30  #Hz

# Function to apply bandpass filter to EEG signals
def apply_bandpass_filter(df, low_freq, high_freq, fs):
    # Extract channel names from the first row
    ch_names = df.iloc[0].tolist()
    #remove the first row contain the channel row
    df = df.iloc[1:]
    # Convert DataFrame to MNE Raw object
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(df.values.T, info)

    # Apply bandpass filter using MNE-Python
    raw.filter(l_freq=low_freq, h_freq=high_freq)

    # Append the filtered channel names after the first row
    df = pd.concat([df.head(1), pd.DataFrame([raw.ch_names], columns=df.columns), df[1:]], ignore_index=True)

    #extract filtered data
    filtered_data = raw.get_data().T
    # Convert back to DataFrame
    df_filtered = pd.DataFrame(data=filtered_data, columns=ch_names)

    return df_filtered

# Apply the bandpass filter to each DataFrame
unhealthy_dataframe = [apply_bandpass_filter(df, low_freq, high_freq, fs) for df in unhealthy_dataframe]
healthy_dataframe = [apply_bandpass_filter(df, low_freq, high_freq, fs) for df in healthy_dataframe]

#Print each dataframe
print_dataframes_with_labels(unhealthy_dataframe, labels_unhealthy)
print_dataframes_with_labels(healthy_dataframe, labels_healthy)

#Normalization
unhealthy_dataframe=normalize_dataframes(unhealthy_dataframe)
healthy_dataframe=normalize_dataframes(healthy_dataframe)

#Print each dataframe
print(f"---------UNHEALTHY:{len(unhealthy_dataframe)} Dataframe ---------")
print_dataframes_with_labels(unhealthy_dataframe, labels_unhealthy)
print("\n")
print(f"---------HEALTHY:{len(healthy_dataframe)} Dataframe ---------")
print_dataframes_with_labels(healthy_dataframe, labels_healthy)

"""Taking only important channel of the signal"""
#Keep only column 'C3' in each DataFrame
unhealthy_dataframe_c3 = [df[['C3']] for df in unhealthy_dataframe]
healthy_dataframe_c3 = [df[['C3']] for df in healthy_dataframe]

#Print each dataframe
print(f"---------UNHEALTHY:{len(unhealthy_dataframe_c3)} Dataframe ---------")
print_dataframes_with_labels(unhealthy_dataframe_c3, labels_unhealthy)
print("\n")
print(f"---------HEALTHY:{len(healthy_dataframe_c3)} Dataframe ---------")
print_dataframes_with_labels(healthy_dataframe_c3, labels_healthy)

#Transform to tensor
unhealthy_tensors = convert_to_tensors_with_labels(unhealthy_dataframe_c3, labels_unhealthy[:len(unhealthy_dataframe)])
healthy_tensors = convert_to_tensors_with_labels(healthy_dataframe_c3, labels_healthy[:len(healthy_dataframe)])

#Print the tensors with labels and shapes
print(f"---------UNHEALTHY:{len(unhealthy_tensors)} Tensor---------")
for tensor, label in unhealthy_tensors:
    print("Label:", label)
    print("Tensor Shape:", tensor.shape)
    print("Tensor:")
    print(tensor)
    print("------------------")

print("\n")

print(f"---------HEALTHY:{len(healthy_tensors)} Tensor---------")
for tensor, label in healthy_tensors:
    print("Label:", label)
    print("Tensor Shape:", tensor.shape)
    print("Tensor:")
    print(tensor)
    print("------------------")

#Plot the signal after apply Band Pass filter

plot_random_signal(unhealthy_tensors, 'Unhealthy signal')
plot_random_signal(healthy_tensors, 'Healthy signal')

#Filter tensor which have less common shape and set a common shape for this tensor in order to make similarity from each other
threshold_unhealthy = (150, 1)
threshold_healthy = (150, 1)

unhealthy_tensors_train = filter_tensors(unhealthy_tensors, threshold_unhealthy)
healthy_tensors_train = filter_tensors(healthy_tensors, threshold_healthy)

# Print the tensors with labels and shapes
print(f"---------UNHEALTHY:{len(unhealthy_tensors)} Tensor---------")
for tensor, label in unhealthy_tensors_train:
    print("Label:", label)
    print("Tensor Shape:", tensor.shape)
    print("Tensor:")
    print(tensor)
    print("------------------")

print("\n")

print(f"---------HEALTHY:{len(healthy_tensors)} Tensor---------")
for tensor, label in healthy_tensors_train:
    print("Label:", label)
    print("Tensor Shape:", tensor.shape)
    print("Tensor:")
    print(tensor)
    print("------------------")


# Saving the tensors into file
path=r'C:\Users\franc\Desktop\AI & Robotics\Elective in AI\BCI_Signal_Reconstruction\Signal_processed'

#torch.save(unhealthy_tensors_train, path + r'\unhealthy_tensors.pt')
#torch.save(healthy_tensors_train, path + r'\healthy_tensors.pt')