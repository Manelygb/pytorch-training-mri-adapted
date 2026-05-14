import torch
from unet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model with the same number of channels/classes
model = UNet(n_channels=3, n_classes=2)  
model.to(device)

# Load the checkpoint
checkpoint_path = "checkpoints/mri_checkpoint_epoch50.pth"
state_dict = torch.load(checkpoint_path, map_location=device)

# Remove 'mask_values' if present
if 'mask_values' in state_dict:
    del state_dict['mask_values']

model.load_state_dict(state_dict)
model.eval()


print(f"Checkpoint {checkpoint_path} loaded successfully!")
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
