import os
import gc
import json
import torch
import math
import optuna
from optuna.exceptions import TrialPruned
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, get_linear_schedule_with_warmup, GenerationConfig, ViTConfig, VisionEncoderDecoderConfig
from transformers.models.vit.modeling_vit import ViTSelfAttention
from peft import LoraConfig, get_peft_model
from PIL import Image
from tqdm import tqdm
from models.model_unet import UNet
from torch.nn import MSELoss
import torch.nn.functional as F
from PIL import Image, ImageOps
from utils import get_ocr_helper, compare_labels, save_img, levenshtein_distance
import editdistance
import matplotlib.pyplot as plt
from transform_helper import PadWhite, AddGaussianNoice



ocr = get_ocr_helper("Tesseract")


def resize_and_create_model(trocr_model, optimal_height, optimal_width, mini_patch_size, device):
    # Ensure the model is on the correct device
    trocr_model.to(device)

    # Access the encoder configuration
    encoder_config = trocr_model.config.encoder

    # Create a new ViT configuration with dynamic parameters
    new_vit_config = ViTConfig(
        hidden_size=768,
        num_hidden_layers=encoder_config.num_hidden_layers,
        num_attention_heads=12,  # Fixed number of attention heads
        intermediate_size=encoder_config.intermediate_size,
        hidden_act=encoder_config.hidden_act,
        layer_norm_eps=encoder_config.layer_norm_eps,
        hidden_dropout_prob=encoder_config.hidden_dropout_prob,  # Dropout rate for hidden layers
        attention_probs_dropout_prob=encoder_config.attention_probs_dropout_prob,  # Dropout rate for attention probs
        image_size=(optimal_height, optimal_width),  # Update image size
        patch_size=mini_patch_size,  # Modify this value as needed
        num_channels=encoder_config.num_channels,
        qkv_bias=encoder_config.qkv_bias,
    )

    # Calculate the new number of patches
    num_patches_height = optimal_height // mini_patch_size
    num_patches_width = optimal_width // mini_patch_size
    num_patches = num_patches_height * num_patches_width

    print(f"Number of patches (height x width): {num_patches_height} x {num_patches_width} = {num_patches}")

    # Create a new VisionEncoderDecoderModel with the updated configuration
    new_trocr_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config=new_vit_config,
        decoder_config=trocr_model.config.decoder  # Use existing decoder config
    )

    new_trocr_model = VisionEncoderDecoderModel(config=new_trocr_config).to(device)

    def update_position_embeddings(new_model, old_model, new_height, new_width, mini_patch_size):
        # Ensure the embeddings are on the correct device
        old_position_embeddings = old_model.encoder.embeddings.position_embeddings.data.to(device)
        num_old_patches = old_position_embeddings.size(1) - 1  # Exclude the class token
        old_grid_size = int(num_old_patches ** 0.5)

        # Calculate new number of patches
        num_new_patches_height = new_height // mini_patch_size
        num_new_patches_width = new_width // mini_patch_size
        new_num_patches = num_new_patches_height * num_new_patches_width

        # Initialize the new position embeddings with the new number of patches
        class_token = old_position_embeddings[:, 0:1, :].to(device)  # Class token
        new_position_embeddings = torch.zeros(1, new_num_patches + 1, old_position_embeddings.size(-1)).to(device)  # +1 for class token

        # Interpolate the old patch tokens to fit the new size
        old_patch_tokens = old_position_embeddings[:, 1:, :].reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2).to(device)
        interpolated_patch_tokens = nn.functional.interpolate(old_patch_tokens, size=(num_new_patches_height, num_new_patches_width), mode='bilinear', align_corners=False)
        interpolated_patch_tokens = interpolated_patch_tokens.permute(0, 2, 3, 1).reshape(1, new_num_patches, -1)

        # Combine class token and interpolated patch tokens
        new_position_embeddings[:, 0:1, :] = class_token
        new_position_embeddings[:, 1:, :] = interpolated_patch_tokens

        # Update the new model's position embeddings
        new_model.encoder.embeddings.position_embeddings = nn.Parameter(new_position_embeddings)

        return new_num_patches  # Return the new number of patches

    # Interpolate position embeddings and get the new number of patches
    new_num_patches = update_position_embeddings(new_trocr_model, trocr_model, optimal_height, optimal_width, mini_patch_size)

    # Resize patch embeddings in the encoder
    old_patch_embeddings = trocr_model.encoder.embeddings.patch_embeddings.projection.weight.data.to(device)
    new_patch_embeddings = nn.Parameter(torch.zeros((old_patch_embeddings.size(0), new_num_patches)).to(device))

    # Instead of using interpolate, let's reshape the weights to match the new patch size
    new_patch_embeddings.data = old_patch_embeddings[:, :new_num_patches]

    # Assign the new patch embeddings to the model
    new_trocr_model.encoder.embeddings.patch_embeddings.projection.weight = new_patch_embeddings

    # Load the weights from the original model (to retain decoder weights, etc.)
    encoder_state_dict = trocr_model.encoder.state_dict()
    encoder_state_dict.pop('embeddings.position_embeddings', None)  # Remove positional embeddings if they're not being resized
    encoder_state_dict.pop('embeddings.patch_embeddings.projection.weight', None)  # Remove patch embeddings if they're not being resized
    new_trocr_model.encoder.load_state_dict(encoder_state_dict, strict=False)

    # Load decoder weights
    new_trocr_model.decoder.load_state_dict(trocr_model.decoder.state_dict(), strict=False)

    # Return the resized model
    return new_trocr_model

class DocumentDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(512, 512)):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()
        self.target_size = target_size

    def _load_samples(self):
        samples = []
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    image_path = os.path.join(subdir, file)
                    # Set the initial path with a .json extension
                    json_path = image_path.replace(".jpg", ".json").replace(".png", ".json")

                    # Check if the .json file exists; if not, try the .txt file
                    if not os.path.exists(json_path):
                        json_path = image_path.replace(".jpg", ".txt").replace(".png", ".txt")
                    samples.append((image_path, json_path))

        # samples = samples[:100]
        return samples


    def sort_labels(self, labels, boxes, threshold=5):
        items_with_coords = [(label, box[1], box[0]) for label, box in zip(labels, boxes)]

        # Sort by y_min first
        items_with_coords.sort(key=lambda x: (x[1], x[2]))  # Sort primarily by y_min, secondarily by x_min

        final_sorted_labels = []
        
        # Group by y_min considering the threshold
        current_group = []
        current_y_min = None
        
        for item in items_with_coords:
            label, y_min, x_min = item
            
            if current_y_min is None:
                current_y_min = y_min
            
            # Check if the current item is within the threshold of the current group
            if abs(y_min - current_y_min) <= threshold:
                current_group.append(item)
            else:
                # Sort the current group by x_min and extend to final list
                current_group.sort(key=lambda x: x[2])  # Sort by x_min
                final_sorted_labels.extend([i[0] for i in current_group])
                # Start a new group
                current_group = [item]
                current_y_min = y_min  # Update the new baseline y_min

        # Don't forget to add the last group
        if current_group:
            current_group.sort(key=lambda x: x[2])  # Sort by x_min
            final_sorted_labels.extend([i[0] for i in current_group])

        return ' '.join(final_sorted_labels)


    def _load_json(self, json_path):
        # print("json name", json_path)
        labels, boxes = [], []
        if json_path.endswith('.json'):
            with open(json_path, 'r') as f:
                data = json.load(f)

            if not data:
                return [], []

            corrected_boxes = []  # Initialize corrected_boxes as an empty list

            # Process each item and only add both label and box if all are valid
            for item in data:
                # print("Processing item:", item)  # Debug print
                if 'label' in item:
                    # Initialize coordinates
                    coordinates = None

                    # Check for standard bounding box coordinates (x_min, y_min, x_max, y_max)
                    if all(k in item for k in ['x_min', 'y_min', 'x_max', 'y_max']):
                        coordinates = [item['x_min'], item['y_min'], item['x_max'], item['y_max']]
                        corrected_box = None  # No correction needed for standard bounding boxes

                    # Check for four corner coordinates (x1, y1, x2, y2, x3, y3, x4, y4)
                    elif all(k in item for k in ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']):
                        # Collect all four corners
                        x_coords = [item['x1'], item['x2'], item['x3'], item['x4']]
                        y_coords = [item['y1'], item['y2'], item['y3'], item['y4']]

                        # Calculate bounding box coordinates
                        x_min = item['x1']
                        x_max = item['x3']
                        y_min = item['y1']
                        y_max = item['y3']
                        coordinates = [x_min, y_min, x_max, y_max]


                    # Only append if we have valid coordinates
                    if coordinates is not None:
                        labels.append(item['label'])
                        boxes.append(coordinates)


        # Process TXT files
        elif json_path.endswith('.txt'):
            with open(json_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    
                    # Ensure there are at least 9 parts (8 coordinates and a label)
                    if len(parts) < 9:
                        continue
                    
                    # Parse coordinates
                    x_min, y_min = int(parts[0]), int(parts[1])
                    x_max, y_max = int(parts[4]), int(parts[5])  # Use third point as bottom-right corner
                    
                    # Combine elements from index 8 onward for the label
                    label = ' '.join(parts[8:]).strip()
                    
                    labels.append(label)
                    boxes.append([x_min, y_min, x_max, y_max])
                    

        return labels, boxes

    def _add_padding(self, image):
        # Get the width and height of the image
        w, h = image.size
        target_w, target_h = self.target_size

        # Calculate the padding needed to reach the target size
        delta_w = target_w - w
        delta_h = target_h - h
        pad_left = delta_w // 2
        pad_top = delta_h // 2
        pad_right = delta_w - pad_left
        pad_bottom = delta_h - pad_top

        # Create a tuple specifying the amount of padding for each side
        padding = (pad_left, pad_top, pad_right, pad_bottom)

        # Add padding (fill with 255 for a white background in grayscale)
        if delta_w > 0 or delta_h > 0:
            padded_image = ImageOps.expand(image, padding, fill=255)
        else:
            padded_image = image  # No padding needed


        return padded_image, pad_left, pad_top

    def _update_boxes(self, boxes, pad_left, pad_top):
        updated_boxes = [
            [x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top]
            for (x_min, y_min, x_max, y_max) in boxes
        ]
        return updated_boxes


    def _validate_boxes(self, boxes, image_size):
        w, h = image_size
        valid_boxes = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            # Ensure that boxes stay within image boundaries
            x_min, x_max = max(0, x_min), min(w, x_max)
            y_min, y_max = max(0, y_min), min(h, y_max)
            valid_boxes.append([x_min, y_min, x_max, y_max])
        return valid_boxes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, json_path = self.samples[idx]
        image = Image.open(image_path).convert("L")
        labels, boxes = self._load_json(json_path)

        # Add padding to make the image the same size (target_size)
        old_size = image.size
        # image, pad_left, pad_top = self._add_padding(image)

        # Update the box coordinates according to the padding
        # if boxes:
        #     boxes = self._update_boxes(boxes, pad_left, pad_top)

        # Apply any additional transformations (e.g., normalization)
        if self.transform:
            image = self.transform(image)

        # Create a list of dictionaries associating each label with its corresponding box
        label_box_pairs = []
        if boxes:
            label_box_pairs = [{'text': label, 'box': torch.tensor(box, dtype=torch.float)} 
                               for label, box in zip(labels, boxes)]
        else:
            label_box_pairs = [{'text': '', 'box': torch.empty((0, 4))}]  # Handle empty case

        # Create the connected string of all labels
        # connected_text = ' '.join(labels)
        connected_text =  self.sort_labels(labels, boxes, 5)

        # Return both the connected string and the label-box pairs
        target = {
            'connected_text': connected_text,
            'label_box_pairs': label_box_pairs
        }

        return image, target


# Custom collate function for DataLoader
def custom_collate_fn(batch):
    images, targets = zip(*batch)
    
    # Stack images into a single tensor
    images = torch.stack(images)
    
    # Extract the connected strings and the label-box pairs
    connected_texts = [target['connected_text'] for target in targets]
    
    # Extract the boxes and ensure padding for unequal box lengths across the batch
    label_box_pairs = [target['label_box_pairs'] for target in targets]
    
    # Handle box padding
    all_boxes = []
    max_boxes = max(len(pairs) for pairs in label_box_pairs)  # Find the maximum number of boxes in any sample

    for pairs in label_box_pairs:
        boxes = []
        for pair in pairs:
            box = pair['box']
            # Ensure that box is not empty and has the correct shape
            if box.numel() == 0 or box.shape[0] != 4:  # Check if it's empty or incorrectly shaped
                box = torch.zeros(4)  # Use a dummy zero tensor for empty/malformed boxes
            boxes.append(box)
        
        # Stack the boxes
        boxes = torch.stack(boxes)

        # Check if padding is needed
        if len(boxes) < max_boxes:
            # Create padding with the same dimensions as boxes
            padded_boxes = torch.cat([boxes, torch.zeros((max_boxes - len(boxes), 4))], dim=0)
        else:
            padded_boxes = boxes

        all_boxes.append(padded_boxes)

    # Stack padded boxes into a tensor
    boxes_tensor = torch.stack(all_boxes)

    # Also return the individual text for each box in the pairs
    texts_for_boxes = [[pair['text'] for pair in pairs] for pairs in label_box_pairs]

    # Return images and a dictionary containing both the connected text and the label-box data
    return images, {'text': connected_texts, 'texts_for_boxes': texts_for_boxes, 'boxes': boxes_tensor}


# Custom decoder forward method for LoRA
def custom_decoder_forward(self, encoder_hidden_states=None, **kwargs):
    return super(type(self), self).forward(encoder_hidden_states=encoder_hidden_states, **kwargs)



def get_tesseract_text(image_tensor):
    texts = []
    # Iterate over each image in the batch (assuming shape is [batch_size, channels, height, width])
    for img in image_tensor:   
        # Perform OCR on the image using Tesseract (or any OCR engine)
        tesseract_result = ocr.get_string(img)  # This returns a list of lists of text
        
        # Flatten and join the result, assuming it is a list of lists
        flattened_text = ' '.join([' '.join(sublist) for sublist in tesseract_result])
        
        texts.append(flattened_text)
    
    return texts  # Return a list of texts for the batch

def padder(crop, h, w):
    _, c_h, c_w = crop.shape
    pad_left = (w - c_w)//2
    pad_right = w - pad_left - c_w
    pad_top = (h - c_h)//2
    pad_bottom = h - pad_top - c_h
    pad = torch.nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 1)
    return pad(crop)



def trocr_pred_to_string(model, processor, src_img):
    generated_ids = model.generate(src_img.to(device))
    # print("shape of generated ids",generated_ids.shape)
    
    # Decode the generated IDs to get the OCR text for each image
    ocr_texts = [processor.decode(ids, skip_special_tokens=True) for ids in generated_ids]

    return ocr_texts

def word_error_rate(tesseract_output: str, ground_truth: str):
    # Tokenize the two strings into words
    tesseract_words = tesseract_output.split()
    ground_truth_words = ground_truth.split()

    # Calculate Levenshtein distance (word-level edit distance)
    distance = editdistance.eval(tesseract_words, ground_truth_words)

    # Normalize by the number of words in the ground truth
    wer = distance / len(ground_truth_words) if len(ground_truth_words) > 0 else 0

    return wer

def f1_score_word_level(tesseract_output: str, ground_truth: str) -> tuple:
    # Tokenize the two strings into words
    tesseract_words = tesseract_output.split()
    ground_truth_words = ground_truth.split()

    # Calculate the number of correct predictions (words present in both Tesseract output and ground truth)
    correct_predictions = len(set(tesseract_words) & set(ground_truth_words))

    # Calculate precision and recall
    precision = correct_predictions / len(tesseract_words) if len(tesseract_words) > 0 else 0
    recall = correct_predictions / len(ground_truth_words) if len(ground_truth_words) > 0 else 0

    # Calculate F1 score
    if precision + recall == 0:
        return 0.0, 0.0, 0.0  # If both precision and recall are zero

    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score, precision, recall  # Always return a tuple



def validate_model(model, prep_model, dataloader, processor, is_training=False, antialias_bool=True):
    model.eval()
    prep_model.eval()
    total_loss = 0.0
    total_tesseract_cer_fullimage = 0.0  # For CER of the connected string
    total_tesseract_cer_patch = 0.0
    total_trocr_cer = 0.0  # For CER of the connected string
    correct_count = 0
    total_box_count = 0  # To count total words for CER calculation (per box)
    total_connected_count = 0  # To count total connected string comparisons
    total_image_count = 0
    total_word_error_rate = 0
    total_f1_score = 0
    total_precision_score = 0
    total_recall_score = 0

    with torch.no_grad():
        for images, targets in dataloader:
            # Move images to the correct device
            pixel_values = images.to(device)
            labels = process_labels(targets['text'], processor, device)

            # Prep model predictions
            images = prep_model(pixel_values)

            # Primary TrOCR loss
            prim_loss = model(pixel_values=resize_image(images,  antialias_bool), labels=labels).loss

            # Secondary MSE loss for the prep model
            sec_loss = secondary_loss_fn(images, torch.ones(images.shape).to(device)) * sec_loss_scalar

            # Combine both losses (as in training loop)
            combined_loss = prim_loss + sec_loss
            total_loss += combined_loss.item()
            tesseract_connected_text = get_tesseract_text(images) 
            # print("tesseract:", tesseract_connected_text)
            trocr_connected_text = trocr_pred_to_string(model, processor, resize_image(images, antialias_bool))
            # print("trocr:", trocr_connected_text)
            # Calculate CER for the entire connected text string
            ground_truth_connected_text = targets['text']
            # print("ground truth text:", ground_truth_connected_text)
            # print("----------------------------------------------------")
            temp, tesseract_cer = compare_labels(tesseract_connected_text, ground_truth_connected_text)
            temp, trocr_cer = compare_labels(trocr_connected_text, ground_truth_connected_text)
            total_tesseract_cer_fullimage += tesseract_cer
            total_trocr_cer += trocr_cer

            word_error = word_error_rate(tesseract_connected_text[0], ground_truth_connected_text[0])
            total_word_error_rate += word_error

            f1_score, precision, recall = f1_score_word_level(tesseract_connected_text[0], ground_truth_connected_text[0])
            total_f1_score += f1_score
            total_precision_score += precision
            total_recall_score += recall
            # If not in training mode, calculate OCR performance metrics
            for i, (image) in enumerate(images):
                image = image.unsqueeze(0)
                bounding_boxes = targets['boxes'][i].cpu().numpy()  # Get the bounding boxes for the image
                texts_for_boxes = targets['texts_for_boxes'][i]  # Get ground truth texts for each box


                # Process individual bounding boxes for OCR
                for j, box in enumerate(bounding_boxes):
                    # Only process if the box is valid (skip padded boxes)
                    if not (box == [0, 0, 0, 0]).all():
                        x_min, y_min, x_max, y_max = box.astype(int)
                        patch = image[0][:, y_min:y_max, x_min:x_max]
                        patch = padder(patch, 32,128)
                        if patch.shape[1] > 0 and patch.shape[2] > 0:  # Check height and width
                            # Perform OCR on the patch (using Tesseract or another OCR function)
                            # predicted_text = get_tesseract_text(patch.unsqueeze(0))
                            predicted_text = ocr.get_labels(patch)
                        else:
                            continue

                        # Get the ground truth text for the corresponding box
                        ground_truth_text = texts_for_boxes[j]

                        total_box_count += 1
                        tesseract_correct_count, tesseract_cer_patch = compare_labels(predicted_text, [ground_truth_text])
                        correct_count +=tesseract_correct_count
                        total_tesseract_cer_patch += tesseract_cer_patch

                total_image_count += 1  # Increment image count

    avg_loss = total_loss / len(dataloader)
    # Calculate the average CER for Tesseract and TrOCR
    avg_tesseract_cer_fullimage = total_tesseract_cer_fullimage / total_image_count if total_image_count > 0 else float('inf')
    avg_trocr_cer = total_trocr_cer / total_image_count if total_image_count > 0 else float('inf')

    avg_word_error_rate = total_word_error_rate / total_image_count if total_image_count > 0 else float('inf')
    avg_f1_score = total_f1_score / total_image_count if total_image_count > 0 else float('inf')
    avg_precision_score = total_precision_score / total_image_count if total_image_count > 0 else float('inf')
    avg_recall_score = total_recall_score / total_image_count if total_image_count > 0 else float('inf')

    # Calculate accuracy (correct count) based on total number of words (bounding boxes)
    accuracy = correct_count / total_box_count if total_box_count > 0 else 0.0
    avg_tesseract_cer_patch = total_tesseract_cer_patch / total_box_count if total_box_count > 0 else 0.0


    return avg_loss, avg_tesseract_cer_fullimage, avg_trocr_cer, avg_tesseract_cer_patch, accuracy, avg_word_error_rate, avg_f1_score, avg_precision_score, avg_recall_score


# Dataset paths
train_dir = "data/patch_dataset/patch_dataset_train"
val_dir = "data/patch_dataset/patch_dataset_dev"

# train_dir = "data/SROIE2019/SROIE2019/train/cropped_img"
# val_dir = "data/SROIE2019/SROIE2019/eval/cropped_img"

# Transformations
prep_transform = transforms.Compose([
    transforms.ToTensor()
])
optimal_height , optimal_width = 384, 384
trocr_transform = transforms.Compose([
    transforms.Resize((optimal_height, optimal_width)),   # Resize to 384x384
    # transforms.RandomRotation(degrees=5),  # Small rotation to handle slight tilts
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust brightness, contrast, etc.
    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation for slight shifts
    # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),  # Apply Gaussian blur to simulate varying image quality
    transforms.ToTensor()  # Convert to tensor
])

# Datasets and DataLoaders
prep_train_dataset = DocumentDataset(root_dir=train_dir, transform=prep_transform)
trocr_train_dataset = DocumentDataset(root_dir=train_dir, transform=prep_transform)
val_dataset = DocumentDataset(root_dir=val_dir, transform=prep_transform)
train_dataloader_prep =  torch.utils.data.DataLoader(prep_train_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
train_dataloader_approx =  torch.utils.data.DataLoader(trocr_train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
val_dataloader =  torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# Training loop
num_epochs = 100
patience_counter = 0
secondary_loss_fn = MSELoss().to(device)

def process_labels(texts, processor, device, max_length=512):
    """
    Process the input texts to generate labels for training.

    Args:
        texts (list): A list of text strings (targets['text']).
        processor (TrOCRProcessor): The processor that handles tokenization.
        device (torch.device): The device to which the labels will be moved.
        max_length (int): The maximum length for tokenization. Default is 512.

    Returns:
        torch.Tensor: A tensor of processed labels, where padding tokens are replaced with -100.
    """
    # Tokenize the texts
    tokenized = processor.tokenizer(
        texts, 
        padding='max_length',  # Pad to the max length
        max_length=max_length,  # Maximum length of the sequences
        truncation=True,        # Truncate sequences longer than max_length
        return_tensors="pt"     # Return as PyTorch tensors
    ).input_ids

    # Replace padding tokens with -100 (ignored in loss calculation)
    labels = tokenized.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Move labels to the specified device
    return labels.to(device)

def resize_image(image, antialias):
    filtered_image = image.repeat(1, 3, 1, 1)
    antialias_bool =  antialias
    resized_image = F.interpolate(filtered_image, size=(optimal_height,optimal_width), mode='bicubic', align_corners=False, antialias= antialias_bool)
    return resized_image


def add_noise(imgs, noiser):
    noisy_imgs = []
    for img in imgs:
        noisy_imgs.append(noiser(img))
    return torch.stack(noisy_imgs)

def plot_img_preds(img_pred):
    # Remove the batch dimension and channel dimension
    img_pred = img_pred.squeeze().detach().cpu().numpy()  # Now the shape will be (350, 500)

    # Plot the grayscale image
    plt.imshow(img_pred, cmap='gray')  # Use grayscale colormap
    plt.axis('off')  # Hide axes for cleaner display




fixed_images, fixed_targets = next(iter(train_dataloader_approx))
fixed_images = fixed_images.to(device)  # Move to device if necessary
sec_loss_scalar = 1
accumulation_steps = 4  # Set accumulation steps
sample_save_path = "outputs/img_out"  # Path to save sample img_preds
os.makedirs(sample_save_path, exist_ok=True)
total_tesseract_calls = 0

# Objective function for Optuna
def objective(trial):

    # Explicit deletion
    global model, prep_model, optimizer_trocr, optimizer_prep  # If these exist globally
    try:
        del model
        del prep_model
        del optimizer_trocr
        del optimizer_prep
    except NameError:
        pass

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    # Hyperparameter suggestions
    lr_trocr = trial.suggest_float("lr_trocr", 1e-6, 1e-4, log=True)
    lr_prep = trial.suggest_float("lr_prep", 1e-4, 1e-2, log=True)
    sec_loss_scalar = trial.suggest_float("sec_loss_scalar", 0.1, 5.0)
    antialias = trial.suggest_categorical("antialias", [True, False])
    weight_decay_trocr = trial.suggest_float("weight_decay_trocr", 1e-5, 1e-2, log=True)
    weight_decay_prep = trial.suggest_float("weight_decay_prep", 1e-5, 1e-2, log=True)


    # Load the UNet model (prep model)
    prep_model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint_path = 'unet_autoencoder_pos.pth'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'),weights_only=True)
    prep_model.load_state_dict(checkpoint)
    del checkpoint  # Free memory


    # checkpoint = torch.load("trained_models/custom_trocr_model_state_and_config_10_save.pth")
    # loaded_model = VisionEncoderDecoderModel(config=checkpoint["config"])
    # # Load weights
    # loaded_model.load_state_dict(checkpoint["model_state_dict"])
    # model = loaded_model.to(device)

    model = VisionEncoderDecoderModel.from_pretrained("trained_models/TROCR_model_62_save").to(device)
    # del checkpoint  # Free memory

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

    # # Set model configuration
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 512
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    model.config.output_attentions = True

    num_epochs = 5  # Optuna runs with a limited number of epochs

    # Define optimizers with suggested hyperparameters
    optimizer_trocr = optim.AdamW(model.parameters(), lr=lr_trocr, weight_decay=weight_decay_trocr)
    optimizer_prep = optim.AdamW(prep_model.parameters(), lr=lr_prep, weight_decay=weight_decay_prep)
    scheduler = get_linear_schedule_with_warmup(optimizer_trocr, 
                                            num_warmup_steps=500, 
                                            num_training_steps=len(train_dataloader_approx) * num_epochs)

    best_composite_score = float('inf')

    for epoch in range(num_epochs):
        model.zero_grad()
        prep_model.zero_grad()
        # Training the prep model
        prep_model.train()
        model.eval()
        total_prep_loss = 0.0
        total_loss = 0.0
        accumulation_steps = 4
        for step, (images, targets) in enumerate(train_dataloader_prep):
            images = images.to(device)
            img_preds = prep_model(images)

            # Compute primary and secondary losses
            labels = process_labels(targets['text'], processor, device)
            prim_loss = model(pixel_values=resize_image(img_preds, antialias=antialias), labels=labels).loss
            sec_loss = secondary_loss_fn(img_preds, torch.ones_like(img_preds).to(device)) * sec_loss_scalar
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

        # Training the TrOCR model
        prep_model.eval()
        model.train()
        total_trocr_loss = 0.0
        total_loss = 0.0
        accumulation_steps = 4
        for step, (images, targets) in enumerate(train_dataloader_approx):
            images = images.to(device)
            img_preds = prep_model(images)
            noisy_imgs = add_noise(img_preds.detach().cpu(), AddGaussianNoice(5, is_stochastic=False)).to(device)

            # Compute loss with Tesseract guidance
            labels = process_labels(targets['text'], processor, device)
            tesseract_labels = get_tesseract_text(noisy_imgs)
            tesseract_labels = process_labels(tesseract_labels, processor, device)

            outputs = model(pixel_values=resize_image(noisy_imgs, antialias=antialias), labels=tesseract_labels)
            loss = outputs.loss
            total_trocr_loss = total_trocr_loss + loss.item() 
            total_loss = total_loss + loss

            if (step + 1) % accumulation_steps == 0:
                avg_loss = total_loss / accumulation_steps
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_trocr.step()
                # scheduler.step()
                model.zero_grad()
                total_loss = 0.0

        avg_trocr_loss = total_trocr_loss / len(train_dataloader_approx)

        # Validation
        (
            val_loss,
            avg_tesseract_cer_fullimage,
            avg_trocr_cer,
            avg_tesseract_cer_patch,
            accuracy,
            avg_word_error_rate,
            avg_f1_score,
            avg_precision_score,
            avg_recall_score
        ) = validate_model(model, prep_model, val_dataloader, processor, 'False', antialias_bool=antialias)

        # Composite score calculation
        composite_score = val_loss * 0.4 + avg_trocr_loss *0.1 + avg_tesseract_cer_fullimage *0.2 + avg_word_error_rate *0.3

        # Log metrics for analysis
        trial.set_user_attr("train_loss",  avg_prep_loss)
        trial.set_user_attr("val_loss", val_loss)
        trial.set_user_attr("aproximation_loss", avg_trocr_loss)
        trial.set_user_attr("Tesseract_CER_FullImage", avg_tesseract_cer_fullimage)
        trial.set_user_attr("TrOCR_CER", avg_trocr_cer)
        trial.set_user_attr("Tesseract_CER_Patch", avg_tesseract_cer_patch)
        trial.set_user_attr("Accuracy", accuracy)
        trial.set_user_attr("WER", avg_word_error_rate)
        trial.set_user_attr("F1_Score", avg_f1_score)
        trial.set_user_attr("Precision", avg_precision_score)
        trial.set_user_attr("Recall", avg_recall_score)

        # Print validation metrics
        print(f"Epoch {epoch + 1} Validation Metrics:")
        print(f"  Train Loss: { avg_prep_loss:.4f}, Approx Loss: {avg_trocr_loss:.4f}, Val Loss: {val_loss:.4f}, Tesseract CER FullImage: {avg_tesseract_cer_fullimage:.4f}, "
              f"TrOCR CER: {avg_trocr_cer:.4f}, WER: {avg_word_error_rate:.4f}, F1: {avg_f1_score:.4f}, "
              f"Precision: {avg_precision_score:.4f}, Recall: {avg_recall_score:.4f}")

        # Save the best model
        if composite_score < best_composite_score:
            best_composite_score = composite_score
            torch.save({"model_state_dict": model.state_dict(), "config": model.config}, "best_model_trocr.pth")
            torch.save(prep_model.state_dict(), "best_model_prep.pth")
        # Clear cache at the end of epoch
        torch.cuda.empty_cache()

    return best_composite_score


# Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Save best parameters
with open("best_hyperparameters.json", "w") as f:
    json.dump(study.best_trial.params, f)

# Reload best models and train for additional epochs
checkpoint = torch.load("best_model_trocr.pth")
model.load_state_dict(checkpoint["model_state_dict"])
prep_model.load_state_dict(torch.load("best_model_prep.pth"))


best_trial = study.best_trial
best_hyperparameters = best_trial.params

# Extract the hyperparameters for optimizers
lr_trocr = best_hyperparameters["lr_trocr"]
lr_prep = best_hyperparameters["lr_prep"]
weight_decay_trocr = best_hyperparameters["weight_decay_trocr"]
weight_decay_prep = best_hyperparameters["weight_decay_prep"]

# Reinitialize optimizers with best hyperparameters
optimizer_trocr = torch.optim.AdamW(
    model.parameters(),
    lr=lr_trocr,
    weight_decay=weight_decay_trocr
)

optimizer_prep = torch.optim.AdamW(
    prep_model.parameters(),
    lr=lr_prep,
    weight_decay=weight_decay_prep
)
best_composite_score = float('inf')  # Keep track of best composite score
for epoch in range(10, 80):  # Resume from epoch 30
    # Training the prep model
    prep_model.train()
    model.eval()
    total_prep_loss = 0.0
    accumulation_steps = 2
    for step, (images, targets) in enumerate(train_dataloader_prep):
        images = images.to(device)
        img_preds = prep_model(images)

        # Compute losses
        labels = process_labels(targets['text'], processor, device)
        prim_loss = model(pixel_values=resize_image(img_preds, antialias=antialias), labels=labels).loss
        sec_loss = secondary_loss_fn(img_preds, torch.ones_like(img_preds).to(device)) * sec_loss_scalar
        combined_loss = prim_loss + sec_loss

        # Backpropagation with gradient accumulation
        combined_loss.backward()
        if (step + 1) % accumulation_steps == 0:
            optimizer_prep.step()
            optimizer_prep.zero_grad()

        total_prep_loss += combined_loss.item()

    avg_prep_loss = total_prep_loss / len(train_dataloader_prep)

    # Training the TrOCR model
    prep_model.eval()
    model.train()
    total_trocr_loss = 0.0
    accumulation_steps = 4
    for step, (images, targets) in enumerate(train_dataloader_approx):
        images = images.to(device)
        img_preds = prep_model(images)
        noisy_imgs = add_noise(img_preds.detach().cpu(), AddGaussianNoice(5, is_stochastic=False)).to(device)

        # Compute loss with Tesseract guidance
        labels = process_labels(targets['text'], processor, device)
        tesseract_labels = get_tesseract_text(noisy_imgs)
        tesseract_labels = process_labels(tesseract_labels, processor, device)

        outputs = model(pixel_values=resize_image(noisy_imgs, antialias=antialias), labels=tesseract_labels)
        loss = outputs.loss

        # Backpropagation with gradient accumulation
        loss.backward()
        if (step + 1) % accumulation_steps == 0:
            optimizer_trocr.step()
            optimizer_trocr.zero_grad()

        total_trocr_loss += loss.item()

    avg_trocr_loss = total_trocr_loss / len(train_dataloader_approx)

    # Validation
    (
        val_loss,
        avg_tesseract_cer_fullimage,
        avg_trocr_cer,
        avg_tesseract_cer_patch,
        accuracy,
        avg_word_error_rate,
        avg_f1_score,
        avg_precision_score,
        avg_recall_score
    ) = validate_model(model, prep_model, val_dataloader, processor, 'False', antialias_bool=antialias)

    # Composite score: Combine metrics (adjust weights as needed)
    composite_score = val_loss + avg_tesseract_cer_fullimage + avg_word_error_rate  # Adjust weighting here

    # Save the best model based on the composite score
    if composite_score < best_composite_score:
        best_composite_score = composite_score
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": model.config
        }, f"trained_models/best_model_extended_pos.pth")
        torch.save(prep_model, "trained_models/Prep_model_extended_pos.pth")

    print(f"Extended Epoch {epoch + 1}, Prep Loss: {avg_prep_loss:.4f}, TrOCR Loss: {avg_trocr_loss:.4f}, "
          f"Validation Loss: {val_loss:.4f}, Tesseract CER FullImage: {avg_tesseract_cer_fullimage:.4f}, "
          f"TrOCR CER: {avg_trocr_cer:.4f}, WER: {avg_word_error_rate:.4f}, F1: {avg_f1_score:.4f}, "
          f"Precision: {avg_precision_score:.4f}, Recall: {avg_recall_score:.4f}")

print("Extended training complete.")