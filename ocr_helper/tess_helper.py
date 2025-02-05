import torch
import tesserocr
import numpy as np
from torchvision.transforms import ToPILImage
import easyocr

import properties as properties
import utils


class TessHelper():
    def __init__(self, empty_char=properties.empty_char, is_eval=False):
        self.empty_char = empty_char
        self.is_eval = is_eval
        self.api_single_line = tesserocr.PyTessBaseAPI(
            lang='eng', psm=tesserocr.PSM.SINGLE_LINE, path=properties.tesseract_path, oem=tesserocr.OEM.LSTM_ONLY)
        self.api_single_block = tesserocr.PyTessBaseAPI(
            lang='eng', psm=tesserocr.PSM.SINGLE_BLOCK, path=properties.tesseract_path)

    #FOR PROCESSING GRASCALE IMAGES

    def get_labels(self, imgs):
        labels = []
        for i in range(imgs.shape[0]):
            img = ToPILImage()(imgs[i])
            self.api_single_line.SetImage(img)
            label = self.api_single_line.GetUTF8Text().strip()
            if label == "":
                label = self.empty_char
            if self.is_eval:
                labels.append(label)
                continue

            label = utils.get_ununicode(label)
            if len(label) > properties.max_char_len:
                label = self.empty_char
            labels.append(label)
        return labels

    def get_string(self, imgs):
        strings_batch = []
        for i in range(imgs.shape[0]):
            img = ToPILImage()(imgs[i])
            self.api_single_block.SetImage(img)
            string = self.api_single_block.GetUTF8Text().strip()
            string = utils.get_ununicode(string)
            strings_batch.append(string.split())
        return strings_batch
    
    def get_bounding_boxes(self, imgs):
        bboxes_batch = []
        for i in range(imgs.shape[0]):
            img = ToPILImage()(imgs[i])
            self.api_single_block.SetImage(img)
            self.api_single_block.Recognize()  # Ensure the image is processed
            iterator = self.api_single_block.GetIterator()

            bboxes = []
            while iterator and not iterator.Empty(tesserocr.RIL.WORD):
                text = iterator.GetUTF8Text(tesserocr.RIL.WORD).strip()
                conf = iterator.Confidence(tesserocr.RIL.WORD)
                box = iterator.BoundingBox(tesserocr.RIL.WORD)
                bboxes.append({'text': text, 'box': {'x': box[0], 'y': box[1], 'w': box[2] - box[0], 'h': box[3] - box[1]}, 'confidence': conf})
                iterator.Next(tesserocr.RIL.WORD)
            bboxes_batch.append(bboxes)
        return bboxes_batch



    def get_text(self, image_tensor):
        texts = []
        # Iterate over each image in the batch (assuming shape is [batch_size, channels, height, width])
        for img in image_tensor:   
            # Perform OCR on the image using Tesseract (or any OCR engine)
            tesseract_result = self.get_string(img)  # This returns a list of lists of text
            
            # Flatten and join the result, assuming it is a list of lists
            flattened_text = ' '.join([' '.join(sublist) for sublist in tesseract_result])
            
            texts.append(flattened_text)
        
        return texts  # Return a list of texts for the batch