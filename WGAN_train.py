import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from utils import plot_eeg_signals
from WGAN_model import Generator, Discriminator

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
batch_size = 64
unhealthy_loader_train = DataLoader(unhealthy, batch_size=batch_size, shuffle=True)
healthy_loader_train = DataLoader(healthy, batch_size=batch_size, shuffle=True)

def generator_loss(fake_output):
    return -torch.mean(fake_output)  # We want to maximize the critic's output for fake samples.

def critic_loss(real_output, fake_output, gradient_penalty):
    return torch.mean(fake_output) - torch.mean(real_output) + gradient_penalty  # Minimize the difference

save_path=r'C:\Users\franc\Desktop\AI & Robotics\Elective in AI\BCI_Signal_Reconstruction\Model_weights'

def compute_gradient_penalty(critic, real_samples, fake_samples):
    batch_size = real_samples.size(0)
    epsilon = torch.rand(batch_size, 1, 1).to(real_samples.device)  # Random weight for interpolation
    interpolated = (epsilon * real_samples + (1 - epsilon) * fake_samples).requires_grad_(True)
    
    critic_interpolated = critic(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(critic_interpolated.size()).to(real_samples.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)  # Flatten gradients
    gradient_norm = gradients.norm(2, dim=1)  # Compute L2 norm
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)  # Penalty term for gradient norm
    return gradient_penalty
    
def train_EEGwgan(generator, discriminator, noisy_dataloader, clean_dataloader, num_epochs=100, lr=0.00002, plot_interval=5, n_critic=5, lambda_gp = 10):

    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    generator.train()
    discriminator.train()

    generator_losses = []
    critic_losses = []

    for epoch in range(num_epochs):
        for (noisy, _), (clean, _) in zip(noisy_dataloader, clean_dataloader):

            clean_eeg = clean[0].to(device)
            noisy_eeg = noisy[0].to(device)

            # ====================
            # Train Discriminator
            # ====================
            for _ in range(n_critic):
                optimizer_d.zero_grad()

                # Fake EEG input
                denoised_data = generator(noisy_eeg.unsqueeze(2)).to(device)
                
                # Real EEG input
                real_output = discriminator(clean_eeg.unsqueeze(2)).to(device)
                fake_output = discriminator(denoised_data.detach().squeeze(0)).to(device)
                
                # Compute gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, clean_eeg.unsqueeze(2), denoised_data.squeeze(0))
                
                loss_c =  critic_loss(real_output, fake_output, lambda_gp * gradient_penalty)
                loss_c.backward()
                optimizer_d.step()
                
            # ======= Train Generator =======
            optimizer_g.zero_grad()

            # Generate fake EEG signals and calculate loss
            denoised_data = generator(noisy_eeg.unsqueeze(2)).to(device)
            fake_output = discriminator(denoised_data.detach().squeeze(0)).to(device)


            # Combined generator loss
            loss_g = generator_loss(fake_output)
            loss_g.backward()
            optimizer_g.step()

        generator_losses.append(loss_g.item())
        critic_losses.append(loss_c.item())

        print(f"Epoch [{epoch}/{num_epochs}], Generator Loss: {loss_g.item():.4f}, Critic Loss: {loss_c.item():.4f}")

        '''
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
        '''
        
    # Save the model weights after the final epoch
    torch.save(generator.state_dict(), save_path + r'\generator_final_epoch.pth')

    return generator_losses, critic_losses

input_size = 1 #channel of the each signal
output_size = 1
seq_len=150
num_epochs = 25

generator = Generator(input_size, output_size,seq_len).to(device)
discriminator = Discriminator(input_size, seq_len).to(device)

#Train the model
generator_losses, discriminator_losses= train_EEGwgan(generator, discriminator, unhealthy_loader_train, healthy_loader_train, num_epochs=num_epochs)

"""Plot the loss"""
plt.figure(figsize=(10, 5))
plt.plot(generator_losses, label='Generator Loss')
plt.plot(discriminator_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('GAN Training Loss')
plt.show()