from ingestion.base import BaseExtractor   
from config import FFMPEG_PATH, WHISPER_MODEL
import whisper
import os

os.environ["PATH"] += os.pathsep + FFMPEG_PATH


class AudioExtractor(BaseExtractor):
    def __init__ (self, model_size = "base"):              #Whisper has different sizes - tiny, base, small, medium, large. 
        self.model = whisper.load_model(model_size)

    async def extract (self, file_path: str):
        result = self.model.transcribe(file_path)  #transcribe takes the audio file and returns a dictionary with the transcribed text and other info
        chunks =[]
        for seg in result ["segments"]:
            chunks.append({
                "text": seg["text"],
                "source": file_path,
                "modality": "audio",
                "start": seg["start"],
                "end": seg["end"]
            })

        return chunks