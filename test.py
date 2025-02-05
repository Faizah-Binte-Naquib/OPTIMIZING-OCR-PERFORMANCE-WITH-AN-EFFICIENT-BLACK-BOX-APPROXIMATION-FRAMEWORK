import argparse
import json
import math
import os
import io

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tqdm import tqdm
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
    padder,
    compare_labels,
    custom_collate_fn,
    get_ocr_helper,
    levenshtein_distance,
    process_labels_trocr,
    resize_and_expand_channels,
    save_img,
    word_error_rate,
    f1_score_word_level,
)

from google.cloud import vision


ocr = get_ocr_helper("Tesseract")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate_model(dataloader, is_training=False):
    """
    Validate the model using the provided dataloader.
    """
    total_tesseract_cer_fullimage = 0.0
    total_tesseract_cer_patch = 0.0
    total_trocr_cer = 0.0
    correct_count = 0
    total_box_count = 0
    total_image_count = 0
    total_word_error_rate = 0
    total_f1_score = 0
    total_precision_score = 0
    total_recall_score = 0

    with torch.no_grad():
        for images, targets in dataloader:
            pixel_values = images.to(device)
            images = prep_model(pixel_values)

            # OCR Predictions
            tesseract_connected_text = ocr.get_text(images)
            ground_truth_connected_text = targets['text']
            temp, tesseract_cer = compare_labels(tesseract_connected_text, ground_truth_connected_text)
            total_tesseract_cer_fullimage += tesseract_cer

            # Compute Word Error Rate and F1 Score
            word_error = word_error_rate(tesseract_connected_text[0], ground_truth_connected_text[0])
            f1_score, precision, recall = f1_score_word_level(tesseract_connected_text[0], ground_truth_connected_text[0])
            
            total_word_error_rate += word_error
            total_f1_score += f1_score
            total_precision_score += precision
            total_recall_score += recall

            # Process each image and bounding box
            for i, image in enumerate(images):
                bounding_boxes = targets['boxes'][i].cpu().numpy()
                texts_for_boxes = targets['texts_for_boxes'][i]

                for j, box in enumerate(bounding_boxes):
                    if (box == [0, 0, 0, 0]).all():
                        continue

                    x_min, y_min, x_max, y_max = map(int, box)
                    patch = image[:, y_min:y_max, x_min:x_max]
                    patch = padder(patch, 32, 128)
                    
                    if patch.shape[1] > 0 and patch.shape[2] > 0:
                        predicted_text = ocr.get_labels(patch)
                    else:
                        continue

                    ground_truth_text = texts_for_boxes[j]
                    total_box_count += 1
                    correct, tesseract_cer_patch = compare_labels(predicted_text, [ground_truth_text])
                    correct_count += correct
                    total_tesseract_cer_patch += tesseract_cer_patch

                total_image_count += 1

    # Compute Averages
    avg_tesseract_cer_fullimage = total_tesseract_cer_fullimage / total_image_count if total_image_count > 0 else float('inf')
    avg_tesseract_cer_patch = total_tesseract_cer_patch / total_box_count if total_box_count > 0 else 0.0
    avg_word_error_rate = total_word_error_rate / total_image_count if total_image_count > 0 else float('inf')
    avg_f1_score = total_f1_score / total_image_count if total_image_count > 0 else float('inf')
    avg_precision_score = total_precision_score / total_image_count if total_image_count > 0 else float('inf')
    avg_recall_score = total_recall_score / total_image_count if total_image_count > 0 else float('inf')
    accuracy = correct_count / total_box_count if total_box_count > 0 else 0.0

    print(f"Total Images: {total_image_count}, Total Boxes: {total_box_count}")

    return (
        avg_tesseract_cer_fullimage, avg_tesseract_cer_patch,
        accuracy, avg_word_error_rate, avg_f1_score, avg_precision_score, avg_recall_score
    )

# Load Preprocessing Model
prep_model = torch.load("trained_models/UNet/Prep_model_sroie_35_best")
prep_model.eval()

# Image Preprocessing Transformation
prep_transform = transforms.Compose([
    transforms.ToTensor()
])

# Dataset Paths
val_dir = "data/patch_dataset/patch_dataset_dev"

# Datasets and DataLoaders
val_dataset = DocumentDataset(root_dir=val_dir, transform=prep_transform)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

# Run Validation
(
    avg_tesseract_cer, avg_tesseract_cer_patch,
    accuracy, avg_word_error_rate, avg_f1_score, avg_precision_score, avg_recall_score
) = validate_model(val_dataloader, is_training=False)

# Print Evaluation Results
print(
    f"Tesseract CER: {avg_tesseract_cer:.4f}, "
    f"Tesseract CER Patch: {avg_tesseract_cer_patch:.4f}, "
    f"Accuracy: {accuracy:.2f}%, "
    f"Word Error Rate: {avg_word_error_rate:.4f}, "
    f"F1 Score: {avg_f1_score:.4f}, "
    f"Precision: {avg_precision_score:.4f}, "
    f"Recall: {avg_recall_score:.4f}"
)
