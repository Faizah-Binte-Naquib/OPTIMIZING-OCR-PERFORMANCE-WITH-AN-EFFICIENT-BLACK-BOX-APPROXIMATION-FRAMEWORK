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
NUM_EPOCHS = 50
ACCUMULATION_STEPS_APPROX = 2
BATCH_SIZE = 2 # Dataloader batch size

# Model & Training Checkpoints
PREP_MODEL_CHECKPOINT = "trained_models/UNet/unet_autoencoder_pos.pth"
TROCR_CHECKPOINT = "microsoft/trocr-base-printed"
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
    transforms.RandomRotation(degrees=5),  # Small rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust image properties
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small translations
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),  # Gaussian blur for noise simulation
    transforms.ToTensor()
])

# ==============================
# Datasets and DataLoaders
# ==============================
trocr_train_dataset = DocumentDataset(root_dir=TRAIN_DIR, transform=trocr_transform)
train_dataloader_approx = torch.utils.data.DataLoader(
    trocr_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn
)


# ==============================
# Load TrOCR Model
# ==============================


# Load the original TrOCR model from a checkpoint
approx_model = VisionEncoderDecoderModel.from_pretrained(TROCR_CHECKPOINT).to(device)
# Initialize the TrocrResizer with the model and new size parameters
resizer = TrocrResizer(approx_model, OPTIMAL_HEIGHT, OPTIMAL_WIDTH, 16, device)
# Create a new resized TrOCR model
approx_model = resizer.resize_and_create_model()


# ==============================
# Processor
# ==============================

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

# ==============================
# Set model configuration
# ==============================

approx_model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
approx_model.config.pad_token_id = processor.tokenizer.pad_token_id
approx_model.config.vocab_size = approx_model.config.decoder.vocab_size
approx_model.config.eos_token_id = processor.tokenizer.sep_token_id
approx_model.config.max_length = 512
approx_model.config.early_stopping = True
approx_model.config.no_repeat_ngram_size = 3
approx_model.config.length_penalty = 2.0
approx_model.config.num_beams = 4
approx_model.config.output_attentions = True



# ==============================
# Optimizers & Schedulers
# ==============================

optimizer_trocr = optim.AdamW(approx_model.parameters(), lr=5e-5, weight_decay=0)

scheduler = get_linear_schedule_with_warmup(
    optimizer_trocr, num_warmup_steps=500, num_training_steps=len(train_dataloader_approx) * NUM_EPOCHS
)

# ==============================
# Training Settings
# ==============================

os.makedirs(SAMPLE_SAVE_PATH, exist_ok=True)


# Function to Add Noise
def add_noise(imgs, noiser):
    noisy_imgs = [noiser(img) for img in imgs]
    return torch.stack(noisy_imgs)



for epoch in range(NUM_EPOCHS):
    total_train_loss = 0
    accumulation_steps = ACCUMULATION_STEPS_APPROX
    approx_model.train()
    total_loss = 0.0
    # --------------------------
    # Loop 1: Train the approx model
    for step, (images, targets) in enumerate(tqdm(train_dataloader_approx)):
        pixel_values = images.to(device)
        noiser = AddGaussianNoice(5, is_stochastic=False)
        noisy_img_preds = add_noise(pixel_values.detach().cpu(), noiser)
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


    # Clear cache at the end of epoch
    torch.cuda.empty_cache()



torch.save({
"model_state_dict": approx_model.state_dict(),
"config": approx_model.config
}, f"trained_models/TrOCR/trocr_pretrained_{args.dataset}.pth")



