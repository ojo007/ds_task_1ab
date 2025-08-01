import io
import pytesseract
import easyocr
import cv2
import numpy as np
from PIL import Image


class OCRService:
    def __init__(self):
        # Initialize EasyOCR (for handwriting)
        self.easyocr_reader = easyocr.Reader(['en'])

    def _preprocess_image(self, image_data):
        """Preprocess image data (bytes) for better OCR results."""
        try:
            # Convert bytes to OpenCV format
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Enhance contrast
            img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
            # Binarization
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return img
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

    def extract_text_from_image(self, image_data, is_handwritten=False):
        """
        Extract text from image data (bytes) dynamically.

        Args:
            image_data (bytes): Raw image bytes (e.g., from file upload).
            is_handwritten (bool): Whether to use EasyOCR (handwriting) or Tesseract (printed).

        Returns:
            str: Extracted text.
        """
        try:
            # Preprocess image
            processed_img = self._preprocess_image(image_data)
            if processed_img is None:
                return ""

            if is_handwritten:
                # Use EasyOCR for handwriting
                results = self.easyocr_reader.readtext(processed_img, detail=0)
                return " ".join(results)
            else:
                # Use Tesseract for printed text
                text = pytesseract.image_to_string(processed_img, config='--psm 6')
                return text.strip()
        except Exception as e:
            print(f"OCR error: {e}")
            return ""

    # Add an alias method to maintain compatibility
    def extract_text(self, image_data, is_handwritten=False):
        """Alias for extract_text_from_image for backward compatibility."""
        return self.extract_text_from_image(image_data, is_handwritten)