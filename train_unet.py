import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from models.model_unet import UNet  # Assuming you have your UNet model defined
from datasets.unet_dataset import UnetDataset  # Assuming you have your dataset defined
from tqdm import tqdm  # Import tqdm for progress bar
from transform_helper import PadWhite, AddGaussianNoice

# Set the device (CPU or GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 50
learning_rate = 0.00005
batch_size = 1
log_interval = 5  # Interval for logging
validation_interval = 5  # Interval for validation
accumulation_steps = 8

# Initialize the model
model = UNet(in_channels=1, out_channels=1).to(device)

# Define Mean Squared Error (MSE) loss
criterion = nn.MSELoss().to(device)

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Transformation for adding noise (if needed)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor
    # Add any additional transformations here, e.g., noise addition
])

# Create UnetDataset instances for training and validation
train_dataset = UnetDataset("data/patch_dataset/patch_dataset_train", transform=transform)
val_dataset = UnetDataset("data/patch_dataset/patch_dataset_dev", transform=transform)

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)



def add_noise(imgs, noiser):
    noisy_imgs = []
    for img in imgs:
        noisy_imgs.append(noiser(img))
    return torch.stack(noisy_imgs)
# Training loop with progress bar and validation every 10 epochs
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_train_loss = 0.0
    noiser = AddGaussianNoice(std=5, is_stochastic=False)

    # Create tqdm progress bar for training batches
    train_progress = tqdm(enumerate(train_loader), total=len(train_loader))
    
    for batch_idx, data in train_progress:
        # Move data to device (GPU if available)
        noisy_data = add_noise(data, noiser)  # Ensure data is moved to the correct device

        # Forward pass: compute reconstructions
        reconstructions = model(noisy_data.to(device))

        # Compute loss: MSE loss between input and reconstructed output
        loss = criterion(reconstructions, data.to(device))  # Remove .detach()

        # Accumulate total loss for reporting
        total_loss = total_loss + loss
        total_train_loss += loss.item()

        # Backward pass: compute gradients and update weights
        # loss.backward()

        # Accumulate gradients and perform an optimization step after accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            avg_loss = total_loss / accumulation_steps
            avg_loss.backward()
            optimizer.step()  # Update weights
            optimizer.zero_grad()  # Reset gradients for the next step
            total_loss = 0.0  # Reset total_loss after each accumulation step

    # Print the average training loss for this epoch
    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")
    
    # Validation every `validation_interval` epochs
    if (epoch + 1) % validation_interval == 0:
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0.0

        with torch.no_grad():  # Disable gradient calculation for validation
            for val_data in val_loader:
                # Move validation data to device
                noisy_data = add_noise(val_data, noiser)  # Ensure data is moved to the correct device
                # Forward pass: compute reconstructions without adding noise
                val_reconstructions = model(noisy_data.to(device))

                # Compute validation loss: MSE loss between input and reconstructed output
                val_loss = criterion(val_reconstructions, val_data.to(device))

                # Accumulate validation loss
                total_val_loss += val_loss.item()

        # Calculate average validation loss
        avg_val_loss = total_val_loss / len(val_loader)

        # Print validation loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}")

# Optionally, save the trained model
torch.save(model.state_dict(), 'unet_autoencoder_pos_test.pth')

