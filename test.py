import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.document_dataset import DocumentDataset
from utils import (
    custom_collate_fn, compare_labels, word_error_rate, f1_score_word_level, get_ocr_helper, padder
)
import properties  # Ensure properties.py contains dataset paths

# Command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--ocr', type=str, required=True, help='OCR model to use (e.g., Tesseract, GoogleVisionAPI)')
parser.add_argument('--dataset', type=str, required=True, choices=['pos', 'sroie'], help='Dataset to use')
args = parser.parse_args()

# Select OCR model
ocr = get_ocr_helper(args.ocr)  # Ensure get_ocr_helper handles multiple OCR options

# Dataset paths
dataset_paths = {
    "pos": {
        "train": properties.patch_dataset_train,
        "test": properties.patch_dataset_test,
        "dev": properties.patch_dataset_dev
    },
    "sroie": {
        "train": properties.sroie_dataset_train,
        "dev": properties.sroie_dataset_dev,
        "test": properties.sroie_dataset_test
    }
}

dataset_path = dataset_paths[args.dataset]["dev"]  # Change "dev" to "train" or "test" if needed
print(dataset_path)

# Load Preprocessing Model
prep_model = torch.load(f"trained_models/UNet/Prep_model_{args.dataset.lower()}_best")
prep_model.eval()

# Image Preprocessing Transformation
prep_transform = transforms.Compose([
    transforms.ToTensor()
])

# Load dataset
val_dataset = DocumentDataset(root_dir=dataset_path, transform=prep_transform)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

# Validate model
def validate_model(dataloader):
    total_cer = 0.0
    total_image_count = 0
    total_word_error_rate = 0
    total_f1_score = 0
    total_precision_score = 0
    total_recall_score = 0
    total_cer_patch = 0
    total_correct_count = 0
    total_box_count = 0

    with torch.no_grad():
        for images, targets in dataloader:
            pixel_values = images.to("cuda" if torch.cuda.is_available() else "cpu")
            images = prep_model(pixel_values)
            
            # OCR Predictions
            ocr_text = ocr.get_text(images)
            ground_truth_text = targets['text']
            _, cer = compare_labels(ocr_text, ground_truth_text)
            total_cer += cer

            word_error = word_error_rate(ocr_text[0], ground_truth_text[0])
            f1_score, precision, recall = f1_score_word_level(ocr_text[0], ground_truth_text[0])
            
            total_word_error_rate += word_error
            total_f1_score += f1_score
            total_precision_score += precision
            total_recall_score += recall

            total_image_count += 1
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
                            if args.ocr == "Tesseract":
                                predicted_text = ocr.get_labels(patch)
                            else:
                                predicted_text = ocr.get_text(patch)
                        else:
                            continue

                        # Get the ground truth text for the corresponding box
                        ground_truth_text = texts_for_boxes[j]

                        total_box_count += 1
                        correct_count, cer_patch = compare_labels(predicted_text, [ground_truth_text])
                        total_correct_count += correct_count
                        total_cer_patch += cer_patch


    # Compute Averages


    accuracy = total_correct_count / total_box_count if total_box_count > 0 else 0.0
    avg_cer_patch = total_cer_patch / total_box_count if total_box_count > 0 else 0.0

    avg_cer = total_cer / total_image_count if total_image_count > 0 else float('inf')
    avg_word_error_rate = total_word_error_rate / total_image_count if total_image_count > 0 else float('inf')
    avg_f1_score = total_f1_score / total_image_count if total_image_count > 0 else float('inf')
    avg_precision_score = total_precision_score / total_image_count if total_image_count > 0 else float('inf')
    avg_recall_score = total_recall_score / total_image_count if total_image_count > 0 else float('inf')

    print(f"Word-level CER: {avg_cer_patch:.4f}, Word-level Accuracy: {accuracy:.4f},Document-level CER: {avg_cer:.4f}, Document-level WER: {avg_word_error_rate:.4f}, "
          f"Document-level F1 Score: {avg_f1_score:.4f}, Document-level Precision: {avg_precision_score:.4f}, Document-level Recall: {avg_recall_score:.4f}")

# Run validation
validate_model(val_dataloader)
