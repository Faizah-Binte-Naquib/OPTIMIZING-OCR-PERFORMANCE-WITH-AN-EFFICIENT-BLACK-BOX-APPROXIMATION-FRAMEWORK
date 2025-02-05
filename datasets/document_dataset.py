import os
import json
import torch
import io
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps
import torch.nn.functional as F
import matplotlib.pyplot as plt


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
        if self.transform:
            image = self.transform(image)

        # Create a list of dictionaries associating each label with its corresponding box
        label_box_pairs = []
        if boxes:
            label_box_pairs = [{'text': label, 'box': torch.tensor(box, dtype=torch.float)} 
                               for label, box in zip(labels, boxes)]
        else:
            label_box_pairs = [{'text': '', 'box': torch.empty((0, 4))}]  # Handle empty case

        connected_text =  self.sort_labels(labels, boxes, 5)

        # Return both the connected string and the label-box pairs
        target = {
            'connected_text': connected_text,
            'label_box_pairs': label_box_pairs
        }

        return image, target
