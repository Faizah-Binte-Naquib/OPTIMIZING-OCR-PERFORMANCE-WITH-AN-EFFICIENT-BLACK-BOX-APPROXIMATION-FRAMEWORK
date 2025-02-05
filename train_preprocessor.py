import io
import json
import math
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm import tqdm

from torch import nn, optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
from transformers import (
    GenerationConfig,
    TrOCRProcessor,
    ViTConfig,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
    get_linear_schedule_with_warmup,
)
from transformers.models.vit.modeling_vit import ViTSelfAttention

from datasets.document_dataset import DocumentDataset
from models.model_trocr import TrocrResizer
from models.model_unet import UNet
from transform_helper import AddGaussianNoice, PadWhite
import properties
from utils import (
    compare_labels,
    custom_collate_fn,
    get_ocr_helper,
    levenshtein_distance,
    process_labels_trocr,
    resize_and_expand_channels,
    save_img,
)

from google.cloud import vision

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train Preprocessor Script")

parser.add_argument(
    "--ocr", 
    type=str, 
    choices=["Tesseract", "GoogleVisionAPI"], 
    default="Tesseract", 
    help="Choose OCR type: Tesseract or GoogleVisionAPI"
)

parser.add_argument(
    "--dataset", 
    type=str, 
    choices=["pos_text", "vgg_text", "patch", "sroie"], 
    required=True, 
    help="Choose dataset: pos_text, vgg_text, patch, or sroie"
)

args = parser.parse_args()


# ==============================
# Hyperparameters
# ==============================

# OCR Type (Options: "Tesseract", "GoogleVisionAPI")
OCR_TYPE = args.ocr

# Dataset Paths
TRAIN_DIR = getattr(properties, f"{args.dataset}_dataset_train")
VAL_DIR = getattr(properties, f"{args.dataset}_dataset_dev")

# Image Preprocessing
OPTIMAL_HEIGHT, OPTIMAL_WIDTH = getattr(properties, f"trocr_input_size_{args.dataset}", (384, 384))

# Training Parameters
NUM_EPOCHS = 100
SEC_LOSS_SCALAR = 0.5  # Scaling factor for secondary loss
if args.dataset == "sroie":
    ACCUMULATION_STEPS_PREP = 2
    ACCUMULATION_STEPS_APPROX = 4
elif args.dataset == "pos_text":
    ACCUMULATION_STEPS_PREP = 4
    ACCUMULATION_STEPS_APPROX = 4
else:
    ACCUMULATION_STEPS_PREP = 4  # Default for other datasets
    ACCUMULATION_STEPS_APPROX = 4
BATCH_SIZE = 1  # Dataloader batch size

# Model & Training Checkpoints
PREP_MODEL_CHECKPOINT = "trained_models/UNet/unet_autoencoder_pos.pth"
TROCR_CHECKPOINT = "trained_models/TrOCR/trocr_pretrained_pos.pth"
SAMPLE_SAVE_PATH = "outputs/img_out"  # Path to save sample predictions

# OCR Helper Initialization
ocr = get_ocr_helper(OCR_TYPE)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# Image Transformations
# ==============================

prep_transform = transforms.Compose([
    transforms.ToTensor()
])

trocr_transform = transforms.Compose([
    transforms.Resize((OPTIMAL_HEIGHT, OPTIMAL_WIDTH)),  # Resize to fixed dimensions
    transforms.ToTensor()
])

# ==============================
# Datasets and DataLoaders
# ==============================

prep_train_dataset = DocumentDataset(root_dir=TRAIN_DIR, transform=prep_transform)
trocr_train_dataset = DocumentDataset(root_dir=TRAIN_DIR, transform=prep_transform)
val_dataset = DocumentDataset(root_dir=VAL_DIR, transform=prep_transform)

train_dataloader_prep = torch.utils.data.DataLoader(
    prep_train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn
)
train_dataloader_approx = torch.utils.data.DataLoader(
    trocr_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn
)

# ==============================
# Load UNet Model (Preprocessing Model)
# ==============================

prep_model = UNet(in_channels=1, out_channels=1).to(device)
checkpoint = torch.load(PREP_MODEL_CHECKPOINT, map_location=torch.device('cpu'), weights_only=True)
prep_model.load_state_dict(checkpoint)

# ==============================
# Load TrOCR Model
# ==============================

checkpoint = torch.load(TROCR_CHECKPOINT)
loaded_model = VisionEncoderDecoderModel(config=checkpoint["config"])
loaded_model.load_state_dict(checkpoint["model_state_dict"])
approx_model = loaded_model.to(device)

# ==============================
# Processor
# ==============================

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

# ==============================
# Optimizers & Schedulers
# ==============================

optimizer_trocr = optim.AdamW(approx_model.parameters(), lr=5e-5, weight_decay=0)
optimizer_prep = optim.AdamW(prep_model.parameters(), lr=5e-5, weight_decay=0)

scheduler = get_linear_schedule_with_warmup(
    optimizer_trocr, num_warmup_steps=500, num_training_steps=len(train_dataloader_approx) * NUM_EPOCHS
)

# ==============================
# Loss Functions
# ==============================

secondary_loss_fn = MSELoss().to(device)

# ==============================
# Fixed Sample for Monitoring
# ==============================

fixed_images, fixed_targets = next(iter(train_dataloader_approx))
fixed_images = fixed_images.to(device)  # Move to device if necessary

# ==============================
# Training Settings
# ==============================

os.makedirs(SAMPLE_SAVE_PATH, exist_ok=True)


# Function to Add Noise
def add_noise(imgs, noiser):
    noisy_imgs = [noiser(img) for img in imgs]
    return torch.stack(noisy_imgs)



for epoch in range(NUM_EPOCHS):
    total_train_loss = 0.0
    total_prep_loss = 0.0

    approx_model.zero_grad()
    prep_model.zero_grad()

    accumulation_steps = 2
    total_loss = 0.0
    approx_model.eval()
    prep_model.train()
    # --------------------------
    # Loop 2: Train the prep model
    for step, (images, targets) in enumerate(tqdm(train_dataloader_prep)):
        images = images.to(device)
        img_preds = prep_model(images)

        labels = process_labels_trocr(targets['text'], processor, device)
        prim_loss = approx_model(pixel_values=resize_and_expand_channels(img_preds, OPTIMAL_HEIGHT, OPTIMAL_WIDTH), labels=labels).loss
        sec_loss = secondary_loss_fn(img_preds, torch.ones(img_preds.shape).to(device)) * SEC_LOSS_SCALAR
        combined_loss = prim_loss + sec_loss
        total_loss = total_loss + combined_loss

        if (step + 1) % accumulation_steps == 0:
            avg_loss = total_loss / accumulation_steps
            avg_loss.backward()
            optimizer_prep.step()
            prep_model.zero_grad()
            total_loss = 0.0

        total_prep_loss += combined_loss.item()
        

    avg_prep_loss = total_prep_loss / len(train_dataloader_prep)
    print(f"Epoch {epoch+1}, Prep Model Loss: {avg_prep_loss:.4f}")



    accumulation_steps = 4
    approx_model.train()
    prep_model.eval()
    total_loss = 0.0
    # --------------------------
    # Loop 1: Train the approx model
    for step, (images, targets) in enumerate(tqdm(train_dataloader_approx)):
        pixel_values = images.to(device)
        img_preds = prep_model(pixel_values)
        noiser = AddGaussianNoice(5, is_stochastic=False)
        img_preds = img_preds.detach().cpu()
        noisy_img_preds = add_noise(img_preds, noiser)
        labels = process_labels_trocr(targets['text'], processor, device)
        tesseract_labels   = ocr.get_text(noisy_img_preds)
        tesseract_labels = process_labels_trocr(tesseract_labels, processor, device)
        
        tesseract_outputs = approx_model(pixel_values=resize_and_expand_channels(noisy_img_preds.to(device), OPTIMAL_HEIGHT, OPTIMAL_WIDTH), labels=tesseract_labels)
        loss = tesseract_outputs.loss
        total_loss = total_loss + loss

        if (step + 1) % accumulation_steps == 0:
            avg_loss = total_loss / accumulation_steps
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(approx_model.parameters(), 1.0)
            optimizer_trocr.step()
            scheduler.step()
            approx_model.zero_grad()
            total_loss = 0.0

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader_approx)

    print(f"Epoch {epoch+1}, Approximation Train Loss: {avg_train_loss:.4f}")

    torch.save({
    "model_state_dict": approx_model.state_dict(),
    "config": approx_model.config
    }, f"trained_models/TrOCR/custom_trocr_model_state_and_config{epoch+1}.pth")



    # Instead of using the current batch, always use fixed_images to save
    with torch.no_grad():  # Ensures gradients are not computed during saving
        img_preds = prep_model(fixed_images)  # Pass the fixed images through the model
        save_img(img_preds.cpu(), 'outgray_' + str(epoch+1), SAMPLE_SAVE_PATH, 8)



    torch.save(prep_model, "trained_models/UNet/Prep_model_"+str(epoch+1))

    # Clear cache at the end of epoch
    torch.cuda.empty_cache()



