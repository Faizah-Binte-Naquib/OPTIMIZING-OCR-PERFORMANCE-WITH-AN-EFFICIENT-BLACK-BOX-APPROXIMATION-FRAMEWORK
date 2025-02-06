# OPTIMIZING OCR PERFORMANCE WITH AN EFFICIENT BLACK-BOX APPROXIMATION FRAMEWORK  

This repository contains the scripts needed to recreate the results mentioned in the manuscript **"OPTIMIZING OCR PERFORMANCE WITH AN EFFICIENT BLACK-BOX APPROXIMATION FRAMEWORK."**  

## 📂 Contents of the Repo  

### **Scripts**  
- `train_trocr.py` - Script to pretrain the TrOCR model.  
- `train_unet.py` - Script to pretrain the UNet model.  
- `train_preprocessor.py` - Script to train the preprocessor.  
- `test.py` - Evaluates the preprocessor with two datasets and two OCR engines.  
- `properties.py` - Contains global properties used by the scripts.  

### **Directories**  
- `trained_models/` - Pretrained preprocessor models and TrOCR models.  
- `datasets/` - Contains data loader scripts.  
- `ocr_helper/` - Contains code to connect with OCR engines.  
- `models/` - Contains the two models (UNet & TrOCR).  

## 🚀 Steps to Run  

### **1. Setup the Environment**  
```bash
# Create and activate a Python virtual environment
python -m venv ocr-test
source ocr-test/bin/activate  # On Windows, use ocr-test\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize the workspace
python init_workspace.py

2. Download the datasets (POS & SROIE)

from https://drive.google.com/drive/folders/1IJsx5VqFVzk_dCbMdYukJUyQNZlynowX?usp=sharing

# Extract the datasets and ensure the folder structure is as follows:

data/
├── patch_dataset/
│   ├── patch_dataset_train/
│   ├── patch_dataset_test/
│   ├── patch_dataset_dev/
│
├── SROIE2019/
│   ├── train/
│   ├── test/
│   ├── eval/

# Update the SROIE dataset
python update_dataset.py

3. Get Pretrained Models

# Download pretrained models from:
# https://drive.google.com/drive/folders/1-MAPdpgTuORs2aXniYoisKyrq5rjqtXU?usp=sharing

# Extract the models into the following directories:
# - Models under "UNet" -> trained_models/UNet/
# - Models under "TrOCR" -> trained_models/TrOCR/

4. Run Evaluation

# Run evaluation with Tesseract OCR on POS dataset
python test.py --ocr Tesseract --dataset pos

# Run evaluation with Tesseract OCR on SROIE dataset
python test.py --ocr Tesseract --dataset sroie


