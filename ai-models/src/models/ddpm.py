import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """Time embeddings for diffusion timesteps"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResnetBlock(nn.Module):
    """ResNet block with time embedding for U-Net"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # Use LayerNorm instead of GroupNorm to avoid channel issues
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        # First block
        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        
        # Second block
        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.residual_conv(x)

class UNet(nn.Module):
    """Simplified U-Net architecture for DDPM"""
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        time_emb_dim=128,
    ):
        super().__init__()
        
        # Channel sizes
        self.ch1 = 64   # First level
        self.ch2 = 128  # Second level  
        self.ch3 = 256  # Third level
        self.ch4 = 512  # Bottom level
        
        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4),
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, self.ch1, 3, padding=1)
        
        # Encoder
        self.down1 = ResnetBlock(self.ch1, self.ch2, time_emb_dim * 4)
        self.down2 = ResnetBlock(self.ch2, self.ch3, time_emb_dim * 4)  
        self.down3 = ResnetBlock(self.ch3, self.ch4, time_emb_dim * 4)
        
        # Downsampling
        self.downsample1 = nn.Conv2d(self.ch2, self.ch2, 4, 2, 1)
        self.downsample2 = nn.Conv2d(self.ch3, self.ch3, 4, 2, 1)
        self.downsample3 = nn.Conv2d(self.ch4, self.ch4, 4, 2, 1)
        
        # Bottleneck
        self.bottleneck1 = ResnetBlock(self.ch4, self.ch4, time_emb_dim * 4)
        self.bottleneck2 = ResnetBlock(self.ch4, self.ch4, time_emb_dim * 4)
        
        # Upsampling
        self.upsample3 = nn.ConvTranspose2d(self.ch4, self.ch4, 4, 2, 1)
        self.upsample2 = nn.ConvTranspose2d(self.ch4, self.ch3, 4, 2, 1)
        self.upsample1 = nn.ConvTranspose2d(self.ch3, self.ch2, 4, 2, 1)
        
        # Decoder (with skip connections) - FIXED CHANNEL CALCULATIONS
        self.up3 = ResnetBlock(self.ch4 + self.ch4, self.ch4, time_emb_dim * 4)  # 512 + 512 = 1024 -> 512
        self.up2 = ResnetBlock(self.ch3 + self.ch3, self.ch3, time_emb_dim * 4)  # 256 + 256 = 512 -> 256
        self.up1 = ResnetBlock(self.ch2 + self.ch2, self.ch2, time_emb_dim * 4)  # 128 + 128 = 256 -> 128
        
        # Output
        self.conv_out = nn.Sequential(
            nn.BatchNorm2d(self.ch2),
            nn.SiLU(),
            nn.Conv2d(self.ch2, out_channels, 3, padding=1)
        )

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_emb(t)
        
        # Initial convolution
        x1 = self.conv_in(x)  # [B, 64, H, W]
        
        # Encoder
        x2 = self.down1(x1, t_emb)  # [B, 128, H, W]
        x2_down = self.downsample1(x2)  # [B, 128, H/2, W/2]
        
        x3 = self.down2(x2_down, t_emb)  # [B, 256, H/2, W/2]
        x3_down = self.downsample2(x3)  # [B, 256, H/4, W/4]
        
        x4 = self.down3(x3_down, t_emb)  # [B, 512, H/4, W/4]
        x4_down = self.downsample3(x4)  # [B, 512, H/8, W/8]
        
        # Bottleneck
        x = self.bottleneck1(x4_down, t_emb)  # [B, 512, H/8, W/8]
        x = self.bottleneck2(x, t_emb)  # [B, 512, H/8, W/8]
        
        # Decoder with skip connections
        x = self.upsample3(x)  # [B, 512, H/4, W/4]
        x = torch.cat([x, x4], dim=1)  # [B, 1024, H/4, W/4]
        x = self.up3(x, t_emb)  # [B, 512, H/4, W/4]
        
        x = self.upsample2(x)  # [B, 256, H/2, W/2]
        x = torch.cat([x, x3], dim=1)  # [B, 512, H/2, W/2]
        x = self.up2(x, t_emb)  # [B, 256, H/2, W/2]
        
        x = self.upsample1(x)  # [B, 128, H, W]
        x = torch.cat([x, x2], dim=1)  # [B, 256, H, W]
        x = self.up1(x, t_emb)  # [B, 128, H, W]
        
        return self.conv_out(x)

class DDPM:
    """Denoising Diffusion Probabilistic Model"""
    def __init__(self, model, device, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.model = model.to(device)
        self.device = device
        self.timesteps = timesteps
        
        # Beta schedule - FIXED: torch.linspace instead of torch.linear
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def get_index_from_list(self, vals, t, x_shape):
        """Extract values from list at timestep t"""
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t):
        """Add noise to clean image (forward process)"""
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = UNet(in_channels=3, out_channels=3)
    ddpm = DDPM(model, device)
    
    # Test forward pass
    x = torch.randn(2, 3, 64, 64).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)
    
    noise_pred = model(x, t)
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {noise_pred.shape}")
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("🎉 DDPM model test passed!")