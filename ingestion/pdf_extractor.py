from ingestion.base import BaseExtractor   #treating ingestion folder as a package
from ingestion.image_extractor import ImageExtractor
from PIL import Image
import PyPDF2
from pdf2image import convert_from_path
from config import POPPLER_PATH
import os




class PDFExtractor(BaseExtractor):
    def __init__(self):
        self.image_extractor = ImageExtractor()

    async def extract(self, file_path: str):
        chunks = []
        reader = PyPDF2.PdfReader(file_path)              #open the pdf, whatever type it is

        for i, page in enumerate(reader.pages):      #go through every page,
            text = page.extract_text()               # try to get text

            if not text or len(text.strip())<50 : #checking page by page
                #scanned page, convert to image

                images= convert_from_path(file_path, first_page= i+1, last_page=i+1, poppler_path=POPPLER_PATH) #convert the page to image, using poppler

                temp_path= f"temp_page_{i}.png"
                images[0].save(temp_path)
                ocr_chunks = await self.image_extractor.extract(temp_path)
                text  = ocr_chunks[0]['text']
                os.remove(temp_path)

            chunks.append({
                "text": text,
                "source": file_path,
                "modality": "pdf",
                "page": i+1
            })
    
        return chunks



