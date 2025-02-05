import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import io
import cv2
from skimage.filters import threshold_sauvola, threshold_niblack
from models.model_unet import UNet
from utils import get_ocr_helper, compare_labels, save_img, levenshtein_distance
from google.cloud import vision
ocr = get_ocr_helper("Tesseract")


def get_google_vision_text(image_tensor):
    # Initialize the Vision API client
    client = vision.ImageAnnotatorClient()

    texts = []
    
    for img_tensor in image_tensor:  # Iterate over the batch if there's more than one tensor
        # Convert tensor to NumPy array
        img_array = img_tensor.detach().cpu().numpy()

        # Handle normalization (assuming tensor is in [0, 1])
        if img_array.max() <= 1:
            img_array = (img_array * 255).astype(np.uint8)  # Scale to [0, 255]
        else:
            img_array = img_array.astype(np.uint8)  # Already in [0, 255]

        # Squeeze the channel dimension (from [1, H, W] to [H, W])
        if img_array.shape[0] == 1:
            img_array = np.squeeze(img_array, axis=0)

        # Convert the NumPy array to a PIL image
        pil_image = Image.fromarray(img_array, mode="L")  # 'L' mode is for grayscale
        
        # Save the PIL image to a bytes buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        # Create a Vision API image object
        image = vision.Image(content=image_bytes)

        # Perform text detection
        response = client.text_detection(image=image)
        annotations = response.text_annotations
        
        # Extract text from annotations
        if annotations:
            detected_text = annotations[0].description  # Full detected text
            cleaned_text = detected_text.replace("\n", " ").strip()  # Replace newlines with spaces
            texts.append(cleaned_text)
        else:
            texts.append("")  # Append an empty string if no text detected

    return texts  # Return list of extracted text for the batch

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

# Function to save an image tensor as a file
def save_image(image_tensor, save_path):
    """
    Save a tensor image to a file.
    Args:
        image_tensor (torch.Tensor): The image tensor, shape (channels, height, width).
        save_path (str): File path to save the image.
    """
    # Convert tensor to numpy, assuming a single channel
    image_numpy = image_tensor.squeeze(0).cpu().numpy()  # Remove channel dimension
    image_numpy = (image_numpy * 255).clip(0, 255).astype(np.uint8)  # Scale to [0, 255]
    Image.fromarray(image_numpy, mode='L').save(save_path)

def otsu_threshold_and_save(image_tensor, save_path):
    """
    Apply Otsu thresholding to a tensor image and save the thresholded image.
    Args:
        image_tensor (torch.Tensor): The image tensor, shape (channels, height, width).
        save_path (str): File path to save the thresholded image.
    """
    # Convert tensor to numpy, assuming a single channel
    image_numpy = image_tensor.squeeze(0).cpu().numpy()  # Remove channel dimension
    image_numpy = (image_numpy * 255).clip(0, 255).astype(np.uint8)  # Scale to [0, 255]

    # Apply Otsu thresholding
    _, otsu_thresholded = cv2.threshold(image_numpy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save the thresholded image
    Image.fromarray(otsu_thresholded, mode='L').save(save_path)
    
    return torch.tensor(otsu_thresholded / 255.0).unsqueeze(0)  # Return as tensor

def sauvola_threshold_and_save(image_tensor, save_path):
    """
    Apply Sauvola thresholding to a tensor image and save the thresholded image.
    Args:
        image_tensor (torch.Tensor): The image tensor, shape (channels, height, width).
        save_path (str): File path to save the thresholded image.
    """
    # Convert tensor to numpy, assuming a single channel
    image_numpy = image_tensor.squeeze(0).cpu().numpy()  # Remove channel dimension
    image_numpy = (image_numpy * 255).clip(0, 255).astype(np.uint8)  # Scale to [0, 255]

    # Apply Sauvola thresholding
    sauvola_thresh = threshold_sauvola(image_numpy, window_size=25)
    sauvola_thresholded = (image_numpy > sauvola_thresh).astype(np.uint8) * 255

    # Save the thresholded image
    Image.fromarray(sauvola_thresholded, mode='L').save(save_path)
    
    return torch.tensor(sauvola_thresholded / 255.0).unsqueeze(0)  # Return as tensor

def niblack_threshold_and_save(image_tensor, save_path):
    """
    Apply Niblack thresholding to a tensor image and save the thresholded image.
    Args:
        image_tensor (torch.Tensor): The image tensor, shape (channels, height, width).
        save_path (str): File path to save the thresholded image.
    """
    # Convert tensor to numpy, assuming a single channel
    image_numpy = image_tensor.squeeze(0).cpu().numpy()  # Remove channel dimension
    image_numpy = (image_numpy * 255).clip(0, 255).astype(np.uint8)  # Scale to [0, 255]

    # Apply Niblack thresholding
    niblack_thresh = threshold_niblack(image_numpy, window_size=25)
    niblack_thresholded = (image_numpy > niblack_thresh).astype(np.uint8) * 255

    # Save the thresholded image
    Image.fromarray(niblack_thresholded, mode='L').save(save_path)
    
    return torch.tensor(niblack_thresholded / 255.0).unsqueeze(0)  # Return as tensor

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained UNet model
prep_model = torch.load('trained_models/Prep_model_76_pos_best')
prep_model.eval()  # Set the model to evaluation mode

prep_model2 = torch.load("trained_models/NN-based-prep/prep_tesseract_pos")
prep_model2.eval()  # Set the model to evaluation mode
# Image preprocessing transform
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
])
#cord 9 and 44
# Load an image
image_path = 'data/SROIE2019/SROIE2019/eval/cropped_img/X51006555814.jpg'  # Replace with your image path
input_image = Image.open(image_path).convert('L')  # Convert to grayscale
image_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension

tess_extracted_text = get_tesseract_text(image_tensor)
print("Tesseract Extracted Text from Original Image:", tess_extracted_text)

gv_extracted_text = get_google_vision_text(image_tensor)
print("Google Vision Extracted Text from Original Image:", gv_extracted_text)

# Process the image through the UNet model
with torch.no_grad():
    processed_image_tensor = prep_model(image_tensor.to(device))
    processed_image_tensor2 = prep_model2(image_tensor.to(device))

# Save the processed image
save_image(processed_image_tensor.squeeze(0), 'processed_image.png')  # Remove batch dimension and save
save_image(processed_image_tensor2.squeeze(0), 'ayantha_processed_image.png')  # Remove batch dimension and save

# Apply Otsu thresholding and save the image
otsu_tensor = otsu_threshold_and_save(image_tensor.squeeze(0), 'otsu_thresholded_image.png')

# Get the text from the Otsu thresholded image
otsu_texts = get_tesseract_text(otsu_tensor.unsqueeze(0))
print("Extracted Text from Otsu Thresholded Image:", otsu_texts)

# Apply Sauvola thresholding and save the image
sauvola_tensor = sauvola_threshold_and_save(image_tensor.squeeze(0), 'sauvola_thresholded_image.png')

# Get the text from the Sauvola thresholded image
sauvola_texts = get_tesseract_text(sauvola_tensor.unsqueeze(0))
print("Extracted Text from Sauvola Thresholded Image:", sauvola_texts)

# Apply Niblack thresholding and save the image
niblack_tensor = niblack_threshold_and_save(image_tensor.squeeze(0), 'niblack_thresholded_image.png')

# Get the text from the Niblack thresholded image
niblack_texts = get_tesseract_text(niblack_tensor.unsqueeze(0))
print("Extracted Text from Niblack Thresholded Image:", niblack_texts)

# Get the text from the processed image
ocr_texts = get_tesseract_text(processed_image_tensor)
print("Extracted Text from Processed Image:", ocr_texts)

# Get the text from the processed image
ocr_texts = get_tesseract_text(processed_image_tensor2)
print("Extracted Text from ayantha Processed Image:", ocr_texts)

# Ensure the processed images are saved correctly
print("Processed image saved at 'processed_image.png'")
print("Otsu thresholded image saved at 'otsu_thresholded_image.png'")
print("Sauvola thresholded image saved at 'sauvola_thresholded_image.png'")
print("Niblack thresholded image saved at 'niblack_thresholded_image.png'")