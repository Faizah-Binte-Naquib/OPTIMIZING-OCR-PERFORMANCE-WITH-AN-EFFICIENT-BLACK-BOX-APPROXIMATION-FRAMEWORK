import cv2
import os

def process_sroie_folder(base_folder):
    # List dataset splits (e.g., train, test, eval)
    dataset_splits = ['train', 'test', 'eval']
    for split in dataset_splits:
        input_folder = os.path.join(base_folder, split, 'img')
        bbox_folder = os.path.join(base_folder, split, 'box')
        cropped_folder = os.path.join(base_folder, split, 'cropped_img')
        
        # Create cropped image output directory if not exists
        os.makedirs(cropped_folder, exist_ok=True)

        # Process each image
        for filename in os.listdir(input_folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(input_folder, filename)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Could not load image {filename} in {split}")
                    continue

                # Get image dimensions and compute the area
                height, width = image.shape[:2]
                image_area = height * width

                # Resize image if resolution exceeds max_resolution
                max_resolution = 1350000  # Max allowed image resolution
                if image_area > max_resolution:
                    scaling_factor = (max_resolution / image_area) ** 0.5
                    new_width = int(width * scaling_factor)
                    new_height = int(height * scaling_factor)
                    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                    height, width = image.shape[:2]
                    image_area = height * width
                    print(f"Resized {filename} in {split} to {new_width}x{new_height}")

                # Thresholds
                min_area_ratio = 0.000005
                max_area_ratio = 5
                min_area = min_area_ratio * image_area
                max_area = max_area_ratio * image_area

                # Convert image to grayscale and apply thresholding
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Connected components analysis
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

                text_boxes = [(x, y, w, h) for i, (x, y, w, h, area) in enumerate(stats[1:], start=1) if min_area < area < max_area]

                if text_boxes:
                    x_min = min(x for x, _, _, _ in text_boxes)
                    y_min = min(y for _, y, _, _ in text_boxes)
                    x_max = max(x + w for x, _, w, _ in text_boxes)
                    y_max = max(y + h for _, y, _, h in text_boxes)

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

                                updated_bboxes.append({
                                    "bbox": [x1 - x_min, y1 - y_min, x2 - x_min, y2 - y_min, x3 - x_min, y3 - y_min, x4 - x_min, y4 - y_min],
                                    "label": label
                                })

                    # Save cropped image
                    cropped_image_path = os.path.join(cropped_folder, filename)
                    cv2.imwrite(cropped_image_path, cropped_image)
                    print(f"Saved cropped image {filename} in {split}")

                    # Save updated bbox info
                    updated_bbox_file_path = os.path.join(cropped_folder, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
                    with open(updated_bbox_file_path, 'w') as bbox_file:
                        for bbox in updated_bboxes:
                            bbox_file.write(','.join(map(str, bbox["bbox"])) + ',' + bbox["label"] + '\n')

# Run the function for SROIE2019 dataset
process_sroie_folder('data/SROIE2019')
