import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mne
import random
from utils import *
from scipy import signal as sig
from sklearn.metrics import mutual_info_score
from fastdtw import fastdtw
from GAN_model import Generator

#Loading Dataset
dataset_unhealthy=r'C:\Users\franc\Desktop\AI & Robotics\Elective in AI\BCI_Signal_Reconstruction\Dataset\EEG_Dataset_Final\Unhealthy'
dataset_healthy=r'C:\Users\franc\Desktop\AI & Robotics\Elective in AI\BCI_Signal_Reconstruction\Dataset\EEG_Dataset_Final\Healthy'

#Transform the dataset in dataframe
unhealthy_dataframe,labels_unhealthy = MyDataset_unhealthy(dataset_unhealthy)
healthy_dataframe,labels_healthy = MyDataset_healthy(dataset_healthy)

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

#Normalization
unhealthy_dataframe=normalize_dataframes(unhealthy_dataframe)
healthy_dataframe=normalize_dataframes(healthy_dataframe)

#Taking different channel for testing
#Keep only column with 'C' in each DataFrame
unhealthy_dataframe_test = [df[['Cz']] for df in unhealthy_dataframe]
healthy_dataframe_test = [df[['Cz']] for df in healthy_dataframe]

unhealthy_tensors_test = convert_to_tensors_with_labels(unhealthy_dataframe_test, labels_unhealthy[:len(unhealthy_dataframe_test)])
healthy_tensors_test = convert_to_tensors_with_labels(healthy_dataframe_test, labels_healthy[:len(healthy_dataframe_test)])

# Define the threshold shape
threshold_unhealthy = (150, 1)
threshold_healthy = (150, 1)

unhealthy_tensors_test = filter_tensors(unhealthy_tensors_test, threshold_unhealthy)
healthy_tensors_test = filter_tensors(healthy_tensors_test, threshold_healthy)


#Create a DataLoader for the training unhealthy and healthy data
batch_size = 64
unhealthy_loader_test = DataLoader(unhealthy_tensors_test, batch_size=batch_size, shuffle=True)
healthy_loader_test = DataLoader(healthy_tensors_test, batch_size=batch_size, shuffle=True)

"""Test GAN model with loss"""

def test_EEGgan(generator, noisy_dataloader, clean_dataloader, num_epochs=20, plot_interval=5):
    
    generator.load_state_dict(torch.load(save_path + r'\GAN_generator_final_epoch.pth', weights_only=True))
    generator.eval()
    loss_fn = nn.MSELoss()

    # Lists to store signals for later plotting
    clean_signals = []
    noisy_signals = []
    denoised_signals = []
    generator_losses = []
    
    for epoch in range(num_epochs):
        print(len(noisy_dataloader))
        
        for (noisy, _), (clean, _) in zip(noisy_dataloader, clean_dataloader):
            clean_eeg = clean[0].to(device)
            noisy_eeg = noisy[0].to(device)

            # Denoise the EEG signal
            denoised_eeg = generator(noisy_eeg.unsqueeze(2)).squeeze(0).detach()

            # Calculate the loss
            generator_loss = loss_fn(denoised_eeg, clean_eeg.unsqueeze(2))

            # Store the signals and loss
            clean_signals.append(clean_eeg.cpu())
            noisy_signals.append(noisy_eeg.cpu())
            denoised_signals.append(denoised_eeg.cpu())
            generator_losses.append(generator_loss.item())

        print(f"Epoch [{epoch}/{num_epochs}], Generator Loss:", generator_loss.item())
        
        # Plot signals at specified intervals
        if epoch % plot_interval == 0:
            plot_eeg_signals(epoch, noisy_eeg, denoised_eeg, clean_eeg)

            # Normalize the signals
            noisy_normalized = (noisy_eeg - torch.min(noisy_eeg)) / (torch.max(noisy_eeg) - torch.min(noisy_eeg))
            denoised_normalized = (denoised_eeg - torch.min(denoised_eeg)) / (torch.max(denoised_eeg) - torch.min(denoised_eeg))

            # Plot the normalized signals overlaid
            plt.figure(figsize=(8, 2))
            plt.plot(noisy_normalized[:128, 0].detach().cpu().numpy(), label='Noisy Signal')
            plt.plot(denoised_normalized[:128, 0].detach().cpu().numpy(), label='Denoised Signal')
            plt.title('Overlaid Noisy and Denoised EEG')
            plt.xlabel('Time (samples)')
            plt.ylabel('Amplitude(Normalized)')
            plt.legend()
            plt.show()

    return clean_signals, noisy_signals, denoised_signals, generator_losses

input_size = 1
output_size = 1
seq_len= 150

save_path=r'C:\Users\franc\Desktop\AI & Robotics\Elective in AI\BCI_Signal_Reconstruction\Model_weights'
generator = Generator(input_size, output_size, seq_len).to(device)
# generator.load_state_dict(torch.load(save_path+'\Gen\generator_epoch_24.pt', weights_only=True))

clean_signals, noisy_signals, denoised_signals, generator_losses = test_EEGgan(generator, unhealthy_loader_test, healthy_loader_test, num_epochs=15)
'''
def plot_signals(clean_signals, noisy_signals, denoised_signals):
    # Set the number of signals to plot (you can adjust this as needed)
    num_signals = min(len(clean_signals), len(noisy_signals), len(denoised_signals))

    # Calculate number of rows needed for two columns
    num_rows = (num_signals + 1) // 2  # Using integer division to round up if odd

    plt.figure(figsize=(12, 3 * num_rows))

    for i in range(num_signals):
        # Normalize the signals
        noisy_normalized = (noisy_signals[i] - torch.min(noisy_signals[i])) / (torch.max(noisy_signals[i]) - torch.min(noisy_signals[i]))
        denoised_normalized = (denoised_signals[i] - torch.min(denoised_signals[i])) / (torch.max(denoised_signals[i]) - torch.min(denoised_signals[i]))
        clean_normalized = (clean_signals[i] - torch.min(clean_signals[i])) / (torch.max(clean_signals[i]) - torch.min(clean_signals[i]))

        # Plot the normalized signals in a 2-column layout
        plt.subplot(num_rows, 2, i + 1)
        plt.plot(clean_normalized[:128, 0].detach().cpu().numpy(), label='Clean Signal', color='green')
        plt.plot(noisy_normalized[:128, 0].detach().cpu().numpy(), label='Noisy Signal', color='red', alpha=0.5)
        plt.plot(denoised_normalized[:128, 0].detach().cpu().numpy(), label='Denoised Signal', color='blue')
        plt.title(f'EEG Signal {i + 1}')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude (Normalized)')
        plt.legend()

    plt.tight_layout()
    plt.show()

# Call the function with your lists of signals
plot_signals(clean_signals, noisy_signals, denoised_signals)
'''  

''' 
def plot_random_signal(clean_signals, noisy_signals, denoised_signals):
    # Check that the input lists are not empty
    if not clean_signals or not noisy_signals or not denoised_signals:
        print("One or more signal lists are empty.")
        return

    while True:  # Loop until the user decides to stop
        # Randomly select an index
        random_index = random.randint(0, min(len(clean_signals), len(noisy_signals), len(denoised_signals)) - 1)

        # Get the selected signals
        clean_signal = clean_signals[random_index]
        noisy_signal = noisy_signals[random_index]
        denoised_signal = denoised_signals[random_index]

        # Normalize the signals
        clean_normalized = (clean_signal - torch.min(clean_signal)) / (torch.max(clean_signal) - torch.min(clean_signal))
        noisy_normalized = (noisy_signal - torch.min(noisy_signal)) / (torch.max(noisy_signal) - torch.min(noisy_signal))
        denoised_normalized = (denoised_signal - torch.min(denoised_signal)) / (torch.max(denoised_signal) - torch.min(denoised_signal))

        # Plot the selected signals
        plt.figure(figsize=(10, 6))
        plt.plot(noisy_normalized[:128, 0].detach().cpu().numpy(), label='Noisy Signal', color='blue')
        plt.plot(denoised_normalized[:128, 0].detach().cpu().numpy(), label='Denoised Signal', color='orange')
        plt.title(f'Randomly Selected EEG Signal (Index: {random_index})')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude (Normalized)')
        plt.legend()
        plt.show()

        # Ask the user if they want to continue
        user_input = input("Do you want to plot another random signal? (yes/no): ").strip().lower()
        if user_input != 'yes':
            print("Exiting the plotting process.")
            break

# Call the function with your lists of signals
plot_random_signal(clean_signals, noisy_signals, denoised_signals)
'''

"""Evaluation of denoised signal

*   SNR (ground truth): Measure of the level of the desired signal compared to the level of background noise
*   RRMSE: Measure of the error between the original signal and the denoised signal

"""

def calculate_snr(signal, noise):
    # Calculate the power of the signal
    signal_power = torch.mean(signal ** 2)
    # Calculate the power of the noise
    noise_power = torch.mean(noise ** 2)
    # Calculate SNR
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr

def calculate_rrmse(original_signal, denoised_signal):
    rmse = torch.sqrt(torch.mean((original_signal - denoised_signal) ** 2))
    rrmse = rmse / torch.std(original_signal)  # Normalizing by the std deviation
    return rrmse

# MAE measures the average magnitude of the errors between the denoised and original signals.
def calculate_mae(original_signal, denoised_signal):
    return torch.mean(torch.abs(original_signal - denoised_signal))

# Higher PSNR values generally indicate better signal fidelity
def calculate_psnr(original_signal, denoised_signal):
    mse = torch.mean((original_signal - denoised_signal) ** 2)
    max_val = torch.max(original_signal)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr

# Quantify whether frequency components are preserved during denoising, especially crucial for EEG.
def calculate_spectral_distortion(original_signal, denoised_signal):
    original_spectrum = torch.fft.fft(original_signal)
    denoised_spectrum = torch.fft.fft(denoised_signal)
    spectral_distortion = torch.norm(original_spectrum - denoised_spectrum) / torch.norm(original_spectrum)
    return spectral_distortion.item()

# Captures non-linear dependencies and can reveal whether the denoising process preserves the informational content of the EEG signal
def calculate_mutual_information(original_signal, denoised_signal):
    # Flatten the signals if they're multidimensional, as MI expects 1D arrays
    original_flat = original_signal.flatten().cpu().numpy()
    denoised_flat = denoised_signal.flatten().cpu().numpy()

    # Calculate the mutual information
    mi = mutual_info_score(original_flat, denoised_flat)
    return mi
def calculate_dtw(original_signal, denoised_signal):
    # Flatten the signals to 1D
    original_flat = original_signal.flatten().cpu().numpy()
    denoised_flat = denoised_signal.flatten().cpu().numpy()

    # Compute the DTW distance and path
    distance, path = fastdtw(original_flat, denoised_flat)
    return distance, path

# Cross-Correlation function
def calculate_cross_correlation(original_signal, denoised_signal):
    original_signal = original_signal.view(1, 1, -1)
    denoised_signal = denoised_signal.view(1, 1, -1)

    flipped_signal = torch.flip(denoised_signal, [2])
    correlation = F.conv1d(original_signal, flipped_signal)

    max_correlation = torch.max(correlation)
    lag = torch.argmax(correlation) - (len(denoised_signal) - 1)

    return max_correlation.item(), lag.item()

'''Print the metrics for all the signal denoised'''
'''
def compute_metrics(clean_signals, noisy_signals, denoised_signals):
    snr_values = []
    rrmse_values = []

    # Check that the input lists have the same length
    if not (len(clean_signals) == len(noisy_signals) == len(denoised_signals)):
        print("Error: All signal lists must have the same length.")
        return snr_values, rrmse_values

    # Iterate over the lists to compute SNR and RRMSE for each signal
    for clean, noisy, denoised in zip(clean_signals, noisy_signals, denoised_signals):
        noise = noisy - denoised
        snr_value = calculate_snr(clean, noise)
        rrmse_value = calculate_rrmse(clean, denoised)

        snr_values.append(snr_value.item())  # Convert tensor to a Python float
        rrmse_values.append(rrmse_value.item())  # Convert tensor to a Python float

    return snr_values, rrmse_values

# Call the function with your lists of signals
snr_values, rrmse_values = compute_metrics(clean_signals, noisy_signals, denoised_signals)

# Print the computed metrics
for i in range(len(snr_values)):
    print(f"Signal {i + 1} - SNR: {snr_values[i]:.2f} dB, RRMSE: {rrmse_values[i]:.4f}")
'''
def compute_metrics(clean_signals, noisy_signals, denoised_signals): 
    snr_values = []
    rrmse_values = []
    mi_values = []
    sd_values = []
    mae_values = []
    psnr_values = []
    cc_values = []  # For Cross-Correlation
    dtw_values = []  # For Dynamic Time Warping

    # Check that the input lists have the same length
    if not (len(clean_signals) == len(noisy_signals) == len(denoised_signals)):
        print("Error: All signal lists must have the same length.")
        return (snr_values, rrmse_values, mi_values, sd_values, 
                mae_values, psnr_values, cc_values, dtw_values)

    # Iterate over the lists to compute each metric for each signal
    for clean, noisy, denoised in zip(clean_signals, noisy_signals, denoised_signals):
        noise = noisy - denoised
        snr_value = calculate_snr(clean, noise)
        rrmse_value = calculate_rrmse(clean, denoised)
        mi_value = calculate_mutual_information(clean, denoised)
        sd_value = calculate_spectral_distortion(clean, denoised)
        mae_value = calculate_mae(clean, denoised)
        psnr_value = calculate_psnr(clean, denoised)

        # Calculate Cross-Correlation and DTW
        cc_value, lag_value = calculate_cross_correlation(clean, denoised)  # You might want to use denoised and clean instead
        dtw_distance, _ = calculate_dtw(clean, denoised)

        # Append each calculated metric to its respective list
        snr_values.append(snr_value.item())
        rrmse_values.append(rrmse_value.item())
        mi_values.append(mi_value)
        sd_values.append(sd_value)
        mae_values.append(mae_value)
        psnr_values.append(psnr_value)
        cc_values.append(cc_value)  # Store cross-correlation result
        dtw_values.append(dtw_distance)  # Store DTW distance

    return (snr_values, rrmse_values, mi_values, sd_values, 
            mae_values, psnr_values, cc_values, dtw_values)

# Call the function with your lists of signals
metrics = compute_metrics(clean_signals, noisy_signals, denoised_signals)

# Print the computed metrics including CC and DTW
for i, (snr, rrmse, mi, sd, mae, psnr, cc, dtw) in enumerate(zip(*metrics)):
    print(f"Signal {i + 1} - SNR: {snr:.2f} dB, RRMSE: {rrmse:.4f}, MI: {mi:.4f}, "
          f"SD: {sd:.4f}, MAE: {mae:.4f}, PSNR: {psnr:.2f} dB, CC: {cc:.4f}, DTW: {dtw:.4f}")

'''Plot random signal by computing metrics'''
'''
def plot_random_signal_with_metrics(clean_signals, noisy_signals, denoised_signals):
    # Check that the input lists are not empty
    if not clean_signals or not noisy_signals or not denoised_signals:
        print("One or more signal lists are empty.")
        return

    while True:  # Loop until the user decides to stop
        # Randomly select an index
        random_index = random.randint(0, min(len(clean_signals), len(noisy_signals), len(denoised_signals)) - 1)

        # Get the selected signals
        clean_signal = clean_signals[random_index]
        noisy_signal = noisy_signals[random_index]
        denoised_signal = denoised_signals[random_index]

        # Normalize the signals
        clean_normalized = (clean_signal - torch.min(clean_signal)) / (torch.max(clean_signal) - torch.min(clean_signal))
        noisy_normalized = (noisy_signal - torch.min(noisy_signal)) / (torch.max(noisy_signal) - torch.min(noisy_signal))
        denoised_normalized = (denoised_signal - torch.min(denoised_signal)) / (torch.max(denoised_signal) - torch.min(denoised_signal))

        # Plot the selected signals
        plt.figure(figsize=(10, 6))
        plt.plot(noisy_normalized[:128, 0].detach().cpu().numpy(), label='Noisy Signal')
        plt.plot(denoised_normalized[:128, 0].detach().cpu().numpy(), label='Denoised Signal')
        plt.title(f'Randomly Selected EEG Signal (Index: {random_index})')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude (Normalized)')
        plt.legend()

        # Calculate noise and metrics
        noise = noisy_signal - denoised_signal
        snr_value = calculate_snr(clean_signal, noise)
        rrmse_value = calculate_rrmse(clean_signal, denoised_signal)

        # Display metrics on the plot
        metrics_text = f"SNR: {snr_value.item():.2f} dB\nRRMSE: {rrmse_value.item():.4f}"
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12, 
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        plt.show()

        # Ask the user if they want to continue
        user_input = input("Do you want to plot another random signal? (yes/no): ").strip().lower()
        if user_input != 'yes':
            print("Exiting the plotting process.")
            break

# Call the function with your lists of signals
plot_random_signal_with_metrics(clean_signals, noisy_signals, denoised_signals)
'''

def plot_random_signal_with_metrics(clean_signals, noisy_signals, denoised_signals):
    if not clean_signals or not noisy_signals or not denoised_signals:
        print("One or more signal lists are empty.")
        return

    while True:
        random_index = random.randint(0, min(len(clean_signals), len(noisy_signals), len(denoised_signals)) - 1)

        clean_signal = clean_signals[random_index]
        noisy_signal = noisy_signals[random_index]
        denoised_signal = denoised_signals[random_index]

        clean_normalized = (clean_signal - torch.min(clean_signal)) / (torch.max(clean_signal) - torch.min(clean_signal))
        noisy_normalized = (noisy_signal - torch.min(noisy_signal)) / (torch.max(noisy_signal) - torch.min(noisy_signal))
        denoised_normalized = (denoised_signal - torch.min(denoised_signal)) / (torch.max(denoised_signal) - torch.min(denoised_signal))

        plt.figure(figsize=(10, 6))
        plt.plot(noisy_normalized[:128, 0].detach().cpu().numpy(), label='Noisy Signal')
        plt.plot(denoised_normalized[:128, 0].detach().cpu().numpy(), label='Denoised Signal')
        plt.title(f'Randomly Selected EEG Signal (Index: {random_index})')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude (Normalized)')
        plt.legend()

        # Calculate metrics for display
        noise = noisy_signal - denoised_signal
        snr_value = calculate_snr(clean_signal, noise)
        rrmse_value = calculate_rrmse(clean_signal, denoised_signal)
        mi_value = calculate_mutual_information(clean_signal, denoised_signal)
        sd_value = calculate_spectral_distortion(clean_signal, denoised_signal)
        mae_value = calculate_mae(clean_signal, denoised_signal)
        psnr_value = calculate_psnr(clean_signal, denoised_signal)
        
        # Calculate cross-correlation and DTW
        cc_value, lag_value = calculate_cross_correlation(clean_signal, denoised_signal)
        dtw_distance, _ = calculate_dtw(clean_signal, denoised_signal)

        metrics_text = (f"SNR: {snr_value.item():.2f} dB\nRRMSE: {rrmse_value.item():.4f}\n"
                        f"MI: {mi_value.item():.4f}\nSD: {sd_value:.4f}\n"
                        f"MAE: {mae_value.item():.4f}\nPSNR: {psnr_value.item():.2f} dB\n"
                        f"Cross-Correlation: {cc_value:.4f}\nDTW Distance: {dtw_distance:.4f}")
        
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12, 
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        plt.show()

        user_input = input("Do you want to plot another random signal? (yes/no): ").strip().lower()
        if user_input != 'yes':
            print("Exiting the plotting process.")
            break

# Call the function with your lists of signals
plot_random_signal_with_metrics(clean_signals, noisy_signals, denoised_signals)
