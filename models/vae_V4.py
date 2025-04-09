import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, image_dim, input_channels=1, latent_dim=256, dropout_rate = 0.5):
        super(VAE, self).__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.image_dim = image_dim
        self.feature_map_size = self.image_dim // (2 * 5)  # 2 * 3 Conv Layers
        self.flattened_dim = 25088

        # Encoder q(z|x)
        self.encoder_l1 = nn.Sequential(nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.Dropout(dropout_rate))
        self.encoder_l2 = nn.Sequential(nn.Conv2d(32, 32 * 2, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.Dropout(dropout_rate))
        self.encoder_l3 = nn.Sequential(nn.Conv2d(32 * 2, 32 * 4, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.Dropout(dropout_rate))
        self.encoder_l4 = nn.Sequential(nn.Conv2d(32 * 4, 32 * 8, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.Dropout(dropout_rate))
        self.encoder_l5 = nn.Sequential(nn.Conv2d(32 * 8, 32 * 16, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.Dropout(dropout_rate))

        # Mean and Log Variance
        self.mu = nn.Linear(self.flattened_dim, self.latent_dim)
        self.logvar = nn.Linear(self.flattened_dim, self.latent_dim)

        # Decoder Input
        self.decoder_input = nn.Linear(self.latent_dim, self.flattened_dim)

        # Decoder p(x|z)
        self.decoder_l1 = nn.Sequential(nn.ConvTranspose2d(32 * 16, 32 * 8, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.Dropout(dropout_rate))
        self.decoder_l2 = nn.Sequential(nn.ConvTranspose2d(32 * 16, 32 * 4, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.Dropout(dropout_rate))
        self.decoder_l3 = nn.Sequential(nn.ConvTranspose2d(32 * 8, 32 * 2, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.Dropout(dropout_rate))
        self.decoder_l4 = nn.Sequential(nn.ConvTranspose2d(32 * 4, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.Dropout(dropout_rate))
        self.decoder_l5 = nn.Sequential(nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1), nn.Sigmoid())

    def reparameterize(self, mean, log_variance):
        standard_deviation = torch.exp(0.5 * log_variance)
        epsilon = torch.randn_like(standard_deviation)
        return mean + epsilon * standard_deviation
    
    def forward(self, x):
        # Model Input
        model_input = x

        # Encoder
        enc_l1 = self.encoder_l1(model_input)
        enc_l2 = self.encoder_l2(enc_l1)
        enc_l3 = self.encoder_l3(enc_l2)
        enc_l4 = self.encoder_l4(enc_l3)
        enc_l5 = self.encoder_l5(enc_l4)

        # Latent Vector
        batch_size = x.shape[0]
        encoder = enc_l5.view(batch_size, -1)  # Reshape to 2D (Batch Size x D)
        mean, log_variance = self.mu(encoder), self.logvar(encoder)
        z = self.reparameterize(mean, log_variance)  # Approximate Z via Reparameterization
        
        dec_input = self.decoder_input(z)  # Expand latent vector to feature map size
        dec_input = dec_input.view(batch_size, 32 * 16, 7, 7)

        # Decoder
        dec_l1 = self.decoder_l1(dec_input)
        dec_l2 = self.decoder_l2(torch.cat([dec_l1, enc_l4], dim=1))
        dec_l3 = self.decoder_l3(torch.cat([dec_l2, enc_l3], dim=1))
        dec_l4 = self.decoder_l4(torch.cat([dec_l3, enc_l2], dim=1))
        dec_l5 = self.decoder_l5(torch.cat([dec_l4, enc_l1], dim=1))

        # Model Output
        dec_output = dec_l5
        
        return dec_output, mean, log_variance

    def sobel_filter(self, img):
        #(B, C, H, W)
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)

        grad_x = F.conv2d(img, sobel_x, padding=1, groups=1)
        grad_y = F.conv2d(img, sobel_y, padding=1, groups=1)

        edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        return edge_map


    def loss_function(self, denoised_x, clean_x, mean, log_variance, beta_weightage=1e-4, edge_weight=0.1):
        recon_edges = self.sobel_filter(denoised_x)
        target_edges = self.sobel_filter(clean_x)
        edge_loss = F.mse_loss(recon_edges, target_edges)

        reconstruction_loss = F.l1_loss(denoised_x, clean_x)
        kl_divergence_loss = -0.5 * torch.mean(1 + log_variance - mean.pow(2) - log_variance.exp())

        return reconstruction_loss + beta_weightage * kl_divergence_loss + edge_weight * edge_loss