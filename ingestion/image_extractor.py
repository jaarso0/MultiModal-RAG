from ingestion.base import BaseExtractor   #treating ingestion folder as a package
from PIL import Image  #pil is the Pillow library, used for image processing (Python Imaging Library)
import pytesseract 
from config import TESSERACT_PATH

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


class ImageExtractor(BaseExtractor):
    async def extract(self, file_path: str):
        image= Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return [{
            "text":text,
            "source":file_path,
            "modality": "image"
        }]
