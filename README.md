# Unknown-box Approximation to Improve Optical Character Recognition Performance
This code repository contains the scripts needs to recreate the results mentioned in the manuscript "OPTIMIZING OCR PERFORMANCE WITH AN EFFICIENT BLACK-BOX
APPROXIMATION FRAMEWORK"
## Contents of the repo
### Scripts
* train_trocr.py - Script to pretrain the TeOCR model
* train_unet.py - Script to pretrain the UNet mdoel
* train_preprocesor.py - Script to train the Preprocessor
* test.py - Evaluate the preprocessor with two datasets and the two OCR engines
* properties.py - Contains global properties used by the scripts
### Directories
* trained_models - Pretrained preprocessor models and trocr models
* datasets - Contains data loader scripts
* ocr_helper - Contains codes to connect with OCR engines
* models - Contains the two models
## Steps to run 
1. We have a requirements.txt file 
2. Create python environment 
3. Run update_dataset.py to update SROIE dataset
4. Run test.py to recreate the results in the paper
