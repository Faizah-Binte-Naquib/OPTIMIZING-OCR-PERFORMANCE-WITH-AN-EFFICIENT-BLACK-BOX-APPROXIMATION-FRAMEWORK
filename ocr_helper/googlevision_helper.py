from google.cloud import vision
from PIL import Image
import io
import numpy as np

default_client = vision.ImageAnnotatorClient()

class GoogleVisionHelper:
    def __init__(self, client=None):
        """
        Initializes the Google Vision Helper class.
        :param client: Optional custom Vision API client instance.
        """
        self.client = client if client else default_client
    
    def get_text(self, image_tensor):
        """
        Extracts text from a batch of image tensors using Google Cloud Vision API.
        :param image_tensor: List of image tensors.
        :return: List of extracted text for each image.
        """
        texts = []
        
        for img_tensor in image_tensor:
            img_array = img_tensor.detach().cpu().numpy()
            
            if img_array.max() <= 1:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)

            if img_array.shape[0] == 1:
                img_array = np.squeeze(img_array, axis=0)

            pil_image = Image.fromarray(img_array, mode="L")
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

            image = vision.Image(content=image_bytes)
            response = self.client.text_detection(image=image)
            annotations = response.text_annotations
            
            if annotations:
                detected_text = annotations[0].description
                cleaned_text = detected_text.replace("\n", " ").strip()
                texts.append(cleaned_text)
            else:
                texts.append("")

        return texts
