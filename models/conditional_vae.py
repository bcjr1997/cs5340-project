import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, input_channels, latent_dim=32, num_classes=15):
        super(VAE, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # Encoder q(z|x)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_classes, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Mean
        self.mu = nn.Linear(128 * 4 * 4, self.latent_dim)
        # Log Variance
        self.logvar = nn.Linear(128 * 4 * 4, self.latent_dim)

        # 

        # Decoder p(x|z)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mean, log_variance):
        standard_deviation = torch.exp(0.5 * log_variance)
        epsilon = torch.randn_like(standard_deviation)
        return mean + epsilon * standard_deviation
    
    def forward(self, x, labels):
        batch_size = x.shape[0]
        labels = labels.view(batch_size, self.num_classes, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat([x, labels], dim=1) # Concatenate the Tensor
        enc = self.encoder(x)
        enc = enc.view(batch_size, -1) # Reshape the Tensor to 2D (Batch Size x D)
        mean, log_variance = self.mu(enc), self.logvar(enc)
        z = self.reparameterize(mean, log_variance) # Approximate Z via Reparameterization
        dec = self.decoder(z)
        return dec, mean, log_variance

    def loss_function(self, denoised_x, clean_x, mean, log_variance):
        reconstruction_loss = F.mse_loss(denoised_x, clean_x)
        kl_divergence_loss = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
        return reconstruction_loss + kl_divergence_loss


    