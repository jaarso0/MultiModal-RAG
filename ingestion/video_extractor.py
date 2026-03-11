from ingestion.base import BaseExtractor   
from ingestion.audio_extractor import AudioExtractor
import cv2
import ollama
import os
import base64
from google import genai
from google.genai import types
import PIL.Image
import time
from config import GEMINI_API_KEY_VISION

FRAME_INTERVAL = 10
client = genai.Client(api_key=GEMINI_API_KEY_VISION)



#base64 is for encoding images into text so they can be sent to ai 

class VideoExtractor(BaseExtractor):
    def __init__(self):
        self.audio_extractor = AudioExtractor()
    
    def _extract_audio(self, video_path: str) -> str:
        base = os.path.splitext(video_path)[0]
        audio_path = f"{base}_temp_audio.wav"  # test_temp_audio.wav instead of test.wav
        os.system(f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y")
        return audio_path
    
    def _frame_to_base64(self, frame):
        _, buffer = cv2.imencode(".jpg", frame)
        return base64.b64encode(buffer).decode("utf-8")
    

    def _describe_frame(self, frame, timestamp: float) -> str:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(rgb_frame)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                f"Describe what is happening in this video frame at timestamp {timestamp:.1f} seconds. Be concise and specific.",
                pil_image
            ]
        )
        return response.text

    async def extract(self, file_path: str):
        chunks = []

        audio_path = self._extract_audio(file_path)
        if os.path.exists(audio_path):
            audio_chunks = await self.audio_extractor.extract(audio_path)
            for chunk in audio_chunks:
                chunk["modality"] = "video_audio"
                chunk["video_source"] = file_path
            
            chunks.extend(audio_chunks)
            os.remove(audio_path)

        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_step = int(fps * FRAME_INTERVAL)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_step ==0:
                timestamp = frame_idx / fps
                print(f"Describing frame at {timestamp:.1f}")
                description = self._describe_frame(frame, timestamp)
                chunks.append({
                    "text": description,
                    "source": file_path,
                    "modality": "video_frame",
                    "timestamp": timestamp
                })
                time.sleep(10)
            frame_idx +=1

        cap.release()
        return chunks