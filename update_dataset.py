import cv2
import os

# Paths for input and output folders and bounding box files
input_folder = 'data/SROIE2019/SROIE2019/train/img/'
bbox_folder = 'data/SROIE2019/SROIE2019/train/box/'
cropped_folder = 'data/SROIE2019/SROIE2019/train/cropped_img/'
output_folder = 'data/SROIE_processed/dev'

# Area thresholds as a fraction of the image area
min_area_ratio = 0.000005
max_area_ratio = 5

# Size thresholds
max_height = 1500
width_threshold = 900
target_area = max_height * width_threshold
max_resolution = 1350000 # Maximum allowed image resolution (pixels)

# Create output folders if they don't exist
os.makedirs(cropped_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Loop through each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Skip if image is not loaded
        if image is None:
            print(f"Could not load image {filename}")
            continue

        # Get image dimensions and compute the area
        height, width = image.shape[:2]
        image_area = height * width

        # Resize image if resolution exceeds max_resolution
        if image_area > max_resolution:
            scaling_factor = (max_resolution / image_area) ** 0.5
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            height, width = image.shape[:2]  # Update dimensions after resizing
            image_area = height * width
            print(f"Resized {filename} to {new_width}x{new_height} (Resolution: {image_area} pixels)")

        # Calculate min and max area thresholds
        min_area = min_area_ratio * image_area
        max_area = max_area_ratio * image_area

        # Convert image to grayscale and apply binary thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Perform connected components analysis
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # Collect bounding boxes for text regions
        text_boxes = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if min_area < area < max_area:
                text_boxes.append((x, y, w, h))

        # Calculate the overall bounding box if text regions are detected
        if text_boxes:
            x_coords = [x for (x, y, w, h) in text_boxes]
            y_coords = [y for (x, y, w, h) in text_boxes]
            widths = [x + w for (x, y, w, h) in text_boxes]
            heights = [y + h for (x, y, w, h) in text_boxes]

            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(widths), max(heights)

            # Crop the image using the overall bounding box
            cropped_image = image[y_min:y_max, x_min:x_max]

            # Load corresponding bbox file
            bbox_file_path = os.path.join(bbox_folder, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
            updated_bboxes = []

            if os.path.exists(bbox_file_path):
                with open(bbox_file_path, 'r') as f:
                    for line in f:
                        coords = line.strip().split(',')
                        x1, y1 = int(coords[0]), int(coords[1])
                        x2, y2 = int(coords[2]), int(coords[3])
                        x3, y3 = int(coords[4]), int(coords[5])
                        x4, y4 = int(coords[6]), int(coords[7])
                        label = ','.join(coords[8:])

                        # Adjust coordinates relative to (x_min, y_min) of the crop
                        x1_adj, y1_adj = x1 - x_min, y1 - y_min
                        x2_adj, y2_adj = x2 - x_min, y2 - y_min
                        x3_adj, y3_adj = x3 - x_min, y3 - y_min
                        x4_adj, y4_adj = x4 - x_min, y4 - y_min

                        updated_bboxes.append({
                            "bbox": [x1_adj, y1_adj, x2_adj, y2_adj, x3_adj, y3_adj, x4_adj, y4_adj],
                            "label": label
                        })

            # Save the cropped image
            cropped_image_path = os.path.join(cropped_folder, filename)
            cv2.imwrite(cropped_image_path, cropped_image)
            print(f"Saved cropped image {filename} with updated bbox info.")

            # Save updated bbox info to individual text files
            updated_bbox_file_path = os.path.join(cropped_folder, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
            with open(updated_bbox_file_path, 'w') as bbox_file:
                for bbox in updated_bboxes:
                    bbox_file.write(','.join(map(str, bbox["bbox"])) + ',' + bbox["label"] + '\n')
