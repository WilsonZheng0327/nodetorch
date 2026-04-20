"""Standalone diffusion model test — same U-Net architecture as the preset.
Trains on MNIST, generates sample images, saves to test_diffusion_output/"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math

# === Noise Schedule ===
class NoiseScheduler:
    def __init__(self, num_timesteps=100, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        self.alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas

    def add_noise(self, x, noise, t):
        sqrt_alpha = self.alpha_cumprod[t].sqrt().view(-1, 1, 1, 1).to(x.device)
        sqrt_one_minus = (1 - self.alpha_cumprod[t]).sqrt().view(-1, 1, 1, 1).to(x.device)
        return sqrt_alpha * x + sqrt_one_minus * noise

    def to(self, device):
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        return self


# === U-Net (matching the preset architecture) ===
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder level 1: 2 -> 32 channels, 28x28
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)  # 28 -> 14

        # Encoder level 2: 32 -> 64 channels, 14x14
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)  # 14 -> 7

        # Bottleneck: 64 -> 128 channels, 7x7
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )

        # Decoder level 2: upsample 7->14, concat with enc2 (128+64=192 -> 64)
        self.up1 = nn.Upsample(scale_factor=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )

        # Decoder level 1: upsample 14->28, concat with enc1 (64+32=96 -> 32)
        self.up2 = nn.Upsample(scale_factor=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        )

        # Output: 32 -> 1
        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # x: [B, 2, 28, 28] (noisy image + timestep channel)
        e1 = self.enc1(x)          # [B, 32, 28, 28]
        e2 = self.enc2(self.pool1(e1))  # [B, 64, 14, 14]
        b = self.bottleneck(self.pool2(e2))  # [B, 128, 7, 7]

        d2 = self.dec2(torch.cat([self.up1(b), e2], dim=1))  # [B, 64, 14, 14]
        d1 = self.dec1(torch.cat([self.up2(d2), e1], dim=1))  # [B, 32, 28, 28]

        return self.out_conv(d1)   # [B, 1, 28, 28]


# === Training ===
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = UNet().to(device)
    scheduler = NoiseScheduler(num_timesteps=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    for epoch in range(30):
        total_loss = 0
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device)
            batch_size = images.size(0)

            t = torch.randint(0, 100, (batch_size,), device=device)
            noise = torch.randn_like(images)
            noisy = scheduler.add_noise(images, noise, t)

            # Timestep channel
            t_ch = (t.float() / 100).view(-1, 1, 1, 1).expand(-1, 1, 28, 28)
            model_input = torch.cat([noisy, t_ch], dim=1)

            pred_noise = model(model_input)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/30: loss={avg_loss:.4f}")

    return model, scheduler, device


# === Sampling ===
@torch.no_grad()
def sample(model, scheduler, device, n=16):
    model.eval()
    x = torch.randn(n, 1, 28, 28, device=device)

    for t_val in reversed(range(scheduler.num_timesteps)):
        t_ch = torch.full((n, 1, 28, 28), t_val / scheduler.num_timesteps, device=device)
        model_input = torch.cat([x, t_ch], dim=1)
        pred_noise = model(model_input)

        alpha = scheduler.alphas[t_val]
        alpha_bar = scheduler.alpha_cumprod[t_val]
        beta = scheduler.betas[t_val]

        if t_val > 0:
            z = torch.randn_like(x)
        else:
            z = 0

        x = (1 / alpha.sqrt()) * (x - (beta / (1 - alpha_bar).sqrt()) * pred_noise) + beta.sqrt() * z

    # Denormalize from MNIST normalized space to [0, 1]
    x = x * 0.3081 + 0.1307
    return x.clamp(0, 1)


# === Save grid ===
def save_grid(images, path, nrow=4):
    """Save a grid of images as a single PNG."""
    from PIL import Image
    n = images.size(0)
    ncol = nrow
    nrows = (n + ncol - 1) // ncol
    h, w = images.shape[2], images.shape[3]
    grid = Image.new('L', (ncol * w, nrows * h), 0)
    for i in range(n):
        img = (images[i, 0].cpu() * 255).byte().numpy()
        grid.paste(Image.fromarray(img), ((i % ncol) * w, (i // ncol) * h))
    grid.save(path)
    print(f"Saved: {path}")


if __name__ == '__main__':
    os.makedirs('test_diffusion_output', exist_ok=True)

    model, scheduler, device = train()

    print("\nGenerating samples...")
    samples = sample(model, scheduler, device, n=16)
    save_grid(samples, 'test_diffusion_output/samples.png', nrow=4)

    print("Done!")
