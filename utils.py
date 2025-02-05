import os
import torch
import Levenshtein
import editdistance
import numpy as np
import torch.nn.functional as F

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from unidecode import unidecode
import torchvision.utils as utils
import properties

import ocr_helper.tess_helper as tess_helper
import ocr_helper.eocr_helper as eocr_helper
import ocr_helper.googlevision_helper as googlevision_helper



# Custom Collate Function for DataLoader
def custom_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)

    connected_texts = [t['connected_text'] for t in targets]
    label_box_pairs = [t['label_box_pairs'] for t in targets]

    max_boxes = max(len(pairs) for pairs in label_box_pairs)

    all_boxes, texts_for_boxes = [], []
    for pairs in label_box_pairs:
        boxes = [pair['box'] if pair['box'].numel() == 4 else torch.zeros(4) for pair in pairs]
        texts_for_boxes.append([pair['text'] for pair in pairs])
        
        # Pad boxes if necessary
        padded_boxes = torch.cat([torch.stack(boxes), torch.zeros((max_boxes - len(boxes), 4))], dim=0) if len(boxes) < max_boxes else torch.stack(boxes)
        all_boxes.append(padded_boxes)

    return images, {'text': connected_texts, 'texts_for_boxes': texts_for_boxes, 'boxes': torch.stack(all_boxes)}


def get_char_maps(vocabulary=None):
    if vocabulary is None:
        vocab = ['-']+[chr(ord('a')+i) for i in range(26)]+[chr(ord('A')+i)
                                                            for i in range(26)]+[chr(ord('0')+i) for i in range(10)]
    else:
        vocab = vocabulary
    char_to_index = {}
    index_to_char = {}
    cnt = 0
    for c in vocab:
        char_to_index[c] = cnt
        index_to_char[cnt] = c
        cnt += 1
    vocab_size = cnt
    return (char_to_index, index_to_char, vocab_size)


def save_img(images, name, dir, nrow=8):
    img = utils.make_grid(images, nrow=nrow)
    img = transforms.ToPILImage()(img)
    img.save(os.path.join(dir, name + '.png'), 'PNG')


def show_img(images, title="Figure", nrow=8):
    img = utils.make_grid(images, nrow=nrow)
    npimg = img.numpy()
    plt.figure(num=title)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_ununicode(text):
    text = text.replace('_', '-')
    text = text.replace('`', "'")
    text = text.replace('©', "c")
    text = text.replace('°', "'")
    text = text.replace('£', "E")
    text = text.replace('§', "S")

    index = text.find('€')
    if index >= 0:
        text = text.replace('€', '<eur>')
    un_unicode = unidecode(text)
    if index >= 0:
        un_unicode = un_unicode.replace('<eur>', '€')
    return un_unicode


def pred_to_string(scores, labels, index_to_char, show_text=False):
    preds = []
    # (seq_len, batch, vocab_size) -> (batch, seq_len, vocab_size)
    scores = scores.cpu().permute(1, 0, 2)
    for i in range(scores.shape[0]):
        interim = []
        for symbol in scores[i, :]:
            index = torch.argmax(symbol).item()
            interim.append(index)
        out = ""
        for j in range(len(interim)):
            if len(out) == 0 and interim[j] != 0:
                out += index_to_char[interim[j]]
            elif interim[j] != 0 and interim[j - 1] != interim[j]:
                out += index_to_char[interim[j]]
        preds.append(out)
        if show_text:
            print(labels[i], " -> ", out)
    return preds


def compare_labels(preds, labels):
    correct_count = 0
    total_cer = 0
    if not isinstance(labels, (list, tuple)):
        labels = [labels]

    lens = len(labels)
    for i in range(lens):
        if preds[i] == labels[i]:
            correct_count += 1
        distance = Levenshtein.distance(labels[i], preds[i])
        if len(labels[i]) > 0:
            total_cer += distance/len(labels[i])
        else:
            continue
    return correct_count, total_cer


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    # Initialize distance matrix
    distances = range(len(s2) + 1)
    for i2, c2 in enumerate(s2):
        new_distances = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                new_distances.append(distances[i1])
            else:
                new_distances.append(1 + min((distances[i1], distances[i1 + 1], new_distances[-1])))
        distances = new_distances
    return distances[-1]



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


def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


def padder(crop, h, w):
    _, c_h, c_w = crop.shape
    pad_left = (w - c_w)//2
    pad_right = w - pad_left - c_w
    pad_top = (h - c_h)//2
    pad_bottom = h - pad_top - c_h
    pad = torch.nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 1)
    return pad(crop)


def get_text_stack(image, labels, input_size):
    text_crops = []
    labels_out = []
    for lbl in labels:
        label = lbl['label']
        x_min = lbl['x_min']
        y_min = lbl['y_min']
        x_max = lbl['x_max']
        y_max = lbl['y_max']
        text_crop = image[:, y_min:y_max, x_min:x_max]
        text_crop = padder(text_crop, *input_size)
        labels_out.append(label)
        text_crops.append(text_crop)
    return torch.stack(text_crops), labels_out


def get_dir_list(test_dir):
    dir_list = []
    for root, dirs, _ in os.walk(test_dir):
        if not dirs:
            dir_list.append(root)
    return dir_list


def get_file_list(in_dir, filter):
    files = os.listdir(in_dir)
    processed_list = []
    for fil in files:
        if fil[-3:] in filter:
            processed_list.append(os.path.join(in_dir, fil))
    return processed_list


def get_files(in_dir, filter):
    processed_list = []
    for root, _, filenames in os.walk(in_dir):
        for f_name in filenames:
            if f_name.endswith(tuple(filter)):
                img_path = os.path.join(root, f_name)
                processed_list.append(img_path)
    return processed_list


def get_noisy_image(image, std=0.05, mean=0):
    noise = torch.normal(mean, std, image.shape)
    out_img = image + noise
    out_img.data.clamp_(0, 1)
    return out_img



def process_labels_trocr(texts, processor, device, max_length=properties.max_length):
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


def resize_and_expand_channels(image, optimal_height, optimal_width):
    filtered_image = image.repeat(1, 3, 1, 1)
    resized_image = F.interpolate(filtered_image, size=(optimal_height, optimal_width), mode='bicubic', align_corners=False, antialias=True)
    return resized_image

def get_ocr_helper(ocr, is_eval=False):

    if ocr == "Tesseract":
        return tess_helper.TessHelper(is_eval=is_eval)
    elif ocr == "EasyOCR":
        return eocr_helper.EocrHelper(is_eval=is_eval)
    elif ocr == "GoogleVisionAPI":
        return googlevision_helper.GoogleVisionHelper()
    else:
        return None
