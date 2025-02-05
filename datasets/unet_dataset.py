import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import tesserocr
import numpy as np
import random
from utils import get_files
import properties as properties
from transform_helper import PadWhite
import properties

import random
import matplotlib.pyplot as plt




# class UnetDataset(Dataset):
    
#     def __init__(self, data_dir, transform=None, include_name=False):
#         self.transform = transform
#         self.include_name = include_name
#         self.files = []
#         unprocessed = get_files(data_dir, ['png', 'jpg'])
#         for img in unprocessed:
#                 self.files.append(img)

#         self.image_transform = transforms.Compose([
#                                 transforms.ToTensor(),
#                             ])

#         # if(temp=="train"):
#         #     self.files = self.files[:100]
#         # else:
#         #     self.files = self.files[:10]

#     def __len__(self):
#         return len(self.files)

#     def downsample_image(self, image, max_width=900, max_height=2000, downsample_factor=0.6):
#         width, height = image.size
        
#         # Loop until the image dimensions are within the desired limits
#         while width > max_width or height > max_height:
#             # Calculate new dimensions
#             new_width = int(width * downsample_factor)
#             new_height = int(height * downsample_factor)
            
#             # Resize the image
#             image = image.resize((new_width, new_height), Image.ANTIALIAS)
            
#             # Update dimensions
#             width, height = image.size
        
#         return image
    

#     def plot_image(self, image, transformed_image):

#         image = image.permute(1, 2, 0).numpy()[:, :, 0]  # Convert tensor to numpy array and remove singleton dimension
#         transformed_image = transformed_image.permute(1, 2, 0).numpy()[:, :, 0]  # Convert tensor to numpy array and remove singleton dimension

#         fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#         axes[0].imshow(image, cmap='gray')
#         axes[0].set_title('Original Image')
#         axes[0].axis('off')
#         axes[1].imshow(transformed_image, cmap='gray')
#         axes[1].set_title('Transformed Image')
#         axes[1].axis('off')
#         plt.show()

#     def __getitem__(self, idx):
#         img_name = self.files[idx]
#         image = Image.open(img_name).convert("L")
#         image =  self.downsample_image(image,1000,2000,0.6)

#         #process labels
#         file_name = os.path.basename(img_name)
#         # labels = file_name.split('_')[1]

#         # Downsample image if needed
#         # image = self.downsample_image(image)

#         if self.transform != None:
#             transformed_noisy_image = self.transform(image)
#             transformed_clean_image = self.image_transform(image)
            
#         else:
#             transformed_noisy_image = transforms.ToTensor()(image)
#             transformed_clean_image = self.image_transform(image)


#         if self.include_name:
#             sample = (transformed_noisy_image, transformed_clean_image)
#         else:
#             sample = (transformed_noisy_image, transformed_clean_image)
#         return sample

#     def worker_init(self, pid):
#         return np.random.seed(torch.initial_seed() % (2**32 - 1))


import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class UnetDataset(Dataset):
    
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.files = []
        unprocessed = get_files(data_dir, ['png', 'jpg'])  # Assuming get_files is a function returning file paths
        for img in unprocessed:
            self.files.append(img)

    def __len__(self):
        return len(self.files)

    def downsample_image(self, image, max_width=900, max_height=2000, downsample_factor=0.6):
        width, height = image.size
        
        while width > max_width or height > max_height:
            new_width = int(width * downsample_factor)
            new_height = int(height * downsample_factor)
            image = image.resize((new_width, new_height), Image.ANTIALIAS)
            width, height = image.size
        
        return image

    def __getitem__(self, idx):
        img_name = self.files[idx]
        image = Image.open(img_name).convert("L")  # Load image and convert to grayscale
        image = self.downsample_image(image, 1000, 2000, 0.6)  # Downsample image if needed
        
        if self.transform is not None:
            transformed_noisy_image = self.transform(image)
        else:
            transformed_noisy_image = transforms.ToTensor()(image)
        
        return transformed_noisy_image

    def worker_init(self, pid):
        np.random.seed(torch.initial_seed() % (2**32 - 1))


# import os
# import torch
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset
# from PIL import Image, ImageOps  # Import ImageOps for padding
# import numpy as np

# class UnetDataset(Dataset):
    
#     def __init__(self, data_dir, target_size=(512, 512), transform=None):
#         self.transform = transform
#         self.target_size = target_size  # Set target size for padding
#         self.files = []
#         unprocessed = get_files(data_dir, ['png', 'jpg'])  # Assuming get_files is a function returning file paths
#         for img in unprocessed:
#             self.files.append(img)

#     def __len__(self):
#         return len(self.files)

#     def _add_padding(self, image):
#         # Get the width and height of the image
#         w, h = image.size
#         target_w, target_h = self.target_size

#         # Calculate the padding needed to reach the target size
#         delta_w = target_w - w
#         delta_h = target_h - h
#         pad_left = delta_w // 2
#         pad_top = delta_h // 2
#         pad_right = delta_w - pad_left
#         pad_bottom = delta_h - pad_top

#         # Create a tuple specifying the amount of padding for each side
#         padding = (pad_left, pad_top, pad_right, pad_bottom)

#         # Add padding (fill with 255 for a white background in grayscale)
#         if delta_w > 0 or delta_h > 0:
#             padded_image = ImageOps.expand(image, padding, fill=255)
#         else:
#             padded_image = image  # No padding needed

#         return padded_image

#     def __getitem__(self, idx):
#         img_name = self.files[idx]
#         image = Image.open(img_name).convert("L")  # Load image and convert to grayscale
        
#         # Add padding to the image
#         padded_image = self._add_padding(image)
        
#         if self.transform is not None:
#             transformed_noisy_image = self.transform(padded_image)
#         else:
#             transformed_noisy_image = transforms.ToTensor()(padded_image)
        
#         return transformed_noisy_image

#     def worker_init(self, pid):
#         np.random.seed(torch.initial_seed() % (2**32 - 1))
