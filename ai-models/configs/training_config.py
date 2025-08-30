from dataclasses import dataclass
from typing import List

@dataclass
class DDPMConfig:
    # Model parameters
    image_size: int = 256
    in_channels: int = 3
    out_channels: int = 3
    time_emb_dim: int = 128
    down_channels: tuple = (64, 128, 256, 512)
    up_channels: tuple = (512, 256, 128, 64)
    
    # Training parameters
    timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 2e-4
    
    # Data parameters
    data_dir: str = "data"
    num_workers: int = 4

# Disease classes for ChestX-ray14
DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# Create default configuration
ddpm_config = DDPMConfig()