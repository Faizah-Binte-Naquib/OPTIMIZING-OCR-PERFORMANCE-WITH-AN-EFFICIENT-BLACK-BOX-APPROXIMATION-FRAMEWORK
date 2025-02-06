# POS Text areas
pos_text_dataset_train = "./data/textarea_dataset_train"
pos_text_dataset_test = "./data/textarea_dataset_test"
pos_text_dataset_dev = "./data/textarea_dataset_dev"

# VGG
vgg_text_dataset_train = "./data/vgg_train"
vgg_text_dataset_test = "./data/vgg_test"
vgg_text_dataset_dev = "./data/vgg_dev"

# POS Patches
patch_dataset_train = "./data/patch_dataset/patch_dataset_train"
patch_dataset_test = "./data/patch_dataset/patch_dataset_test"
patch_dataset_dev = "./data/patch_dataset/patch_dataset_dev"

#SROIE 
sroie_dataset_train = "data/SROIE2019/train/cropped_img"
sroie_dataset_dev = "data/SROIE2019/eval/cropped_img"
sroie_dataset_test = "data/SROIE2019/test/cropped_img"


trocr_model_path = "./trained_models/TrOCR/"
tror_tensor_board = "./trained_models/trocr_runs/"
prep_model_path = "./trained_models/UNet/"
prep_tensor_board = "./trained_models/UNet/"
img_out_path = "./trained_models/img_out/"
param_path = "./trained_models/params.txt"

trocr_input_size_pos = (384,384)
trocr_input_size_sroie = (512,512)
input_size = (32, 128)
num_workers = 4
char_set = ['`', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
            'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '^', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '~', '€', '}', '\\', '/']

tesseract_path = "/usr/share/tesseract-ocr/4.00/tessdata"
empty_char = ' '
max_char_len = 25
max_length = 512
