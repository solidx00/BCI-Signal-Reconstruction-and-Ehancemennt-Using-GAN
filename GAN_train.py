import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import KLDivLoss
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from utils import plot_eeg_signals
from GAN_model import Generator, Discriminator

cuda = torch.cuda.is_available()

#GPU Setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    cuda = True
    print('Using: ' + str(torch.cuda.get_device_name(device)))
else:
    cuda = False
    print('Using: CPU')

path=r'C:\Users\franc\Desktop\AI & Robotics\Elective in AI\BCI_Signal_Reconstruction\Signal_processed'

#Load the tensors already pre-processed
unhealthy = torch.load(path + r'\unhealthy_tensors.pt')
healthy = torch.load(path+ r'\healthy_tensors.pt')

"""Dataloader"""

#Create a DataLoader for the training unhealthy and healthy data
batch_size = 32

unhealthy_loader_train = DataLoader(unhealthy, batch_size=batch_size, shuffle=True)
healthy_loader_train = DataLoader(healthy, batch_size=batch_size, shuffle=True)

"""Gan Training"""

def train_EEGgan(generator, discriminator, noisy_dataloader, clean_dataloader, num_epochs=100, lr=0.01, plot_interval=5):

    adversarial_loss = nn.BCELoss()
    content_loss = nn.MSELoss()
    kl_div_loss = nn.KLDivLoss(reduction="batchmean", log_target=False)  # Optional KL-divergence

    optimizer_g = optim.Adam(generator.parameters(), lr=0.00002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.002, betas=(0.5, 0.999))

    generator.train()
    discriminator.train()

    generator_losses = []
    discriminator_losses = []

    for epoch in range(num_epochs):
        for (noisy, _), (clean, _) in zip(noisy_dataloader, clean_dataloader):

            clean_eeg = clean[0].to(device)
            noisy_eeg = noisy[0].to(device)

            real_labels = torch.ones(clean_eeg.size(0), 1).to(device)
            fake_labels = torch.zeros(clean_eeg.size(0), 1).to(device)

            # ====================
            # Train Discriminator
            # ====================
            optimizer_d.zero_grad()

            real_outputs = discriminator(clean_eeg.unsqueeze(2)).to(device)
            d_loss_real = adversarial_loss(real_outputs, real_labels)

            denoised_data = generator(noisy_eeg.unsqueeze(2)).to(device)
            fake_outputs = discriminator(denoised_data.squeeze(0)).to(device)
            d_loss_fake = adversarial_loss(fake_outputs, fake_labels)

            loss_D = (d_loss_real + d_loss_fake) / 2
            loss_D.backward()
            optimizer_d.step()

            # ====================
            # Train Generator
            # ====================
            optimizer_g.zero_grad()

            denoised_data = generator(noisy_eeg.unsqueeze(2))
            fake_outputs = discriminator(denoised_data.squeeze(0)).to(device)

            # Adversarial loss
            loss_G_adv = adversarial_loss(fake_outputs, real_labels)

            # Content loss (MSE)
            loss_G_content = content_loss(denoised_data, real_labels)

            # Optional KL-divergence loss
            denoised_data_log = F.log_softmax(denoised_data, dim=1)
            clean_eeg_softmax = F.softmax(clean_eeg, dim=1)
            loss_G_kl = kl_div_loss(denoised_data_log, clean_eeg_softmax)

            # Combined generator loss
            loss_G = loss_G_kl #loss_G_adv + 0.5 * loss_G_content #+ 0.1 * loss_G_kl

            loss_G.backward()
            optimizer_g.step()

        generator_losses.append(loss_G.item())
        discriminator_losses.append(loss_D.item())

        print(f"Epoch [{epoch}/{num_epochs}], Generator Loss: {loss_G.item():.4f}, Discriminator Loss: {loss_D.item():.4f}")

        # Plot signals at specified intervals
        if epoch % plot_interval == 0:
            plot_eeg_signals(epoch, noisy_eeg, denoised_data, clean_eeg)

            # Normalize the signals
            noisy_normalized = (noisy_eeg - torch.min(noisy_eeg)) / (torch.max(noisy_eeg) - torch.min(noisy_eeg))
            denoised_normalized = (denoised_data - torch.min(denoised_data)) / (torch.max(denoised_data) - torch.min(denoised_data))

            # Plot the normalized signals overlaid
            plt.figure(figsize=(8, 2))
            plt.plot(noisy_normalized[:128, 0].detach().cpu().numpy(), label='Noisy Signal')
            plt.plot(denoised_normalized[:128, 0].detach().cpu().numpy(), label='Denoised Signal')
            plt.title('Overlaid Noisy and Denoised EEG')
            plt.xlabel('Time (samples)')
            plt.ylabel('Amplitude(Normalized)')
            plt.legend()
            plt.show()

    return generator_losses, discriminator_losses


input_size = 1 #channel of the each signal
output_size = 1
seq_len=150
num_epochs = 25

generator = Generator(input_size, output_size, seq_len).to(device)
discriminator = Discriminator(input_size, seq_len).to(device)

#Train the model
generator_losses, discriminator_losses= train_EEGgan(generator, discriminator, unhealthy_loader_train, healthy_loader_train, num_epochs=num_epochs)

"""Plot the loss"""
plt.figure(figsize=(10, 5))
plt.plot(generator_losses, label='Generator Loss')
plt.plot(discriminator_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('GAN Training Loss')
plt.show()