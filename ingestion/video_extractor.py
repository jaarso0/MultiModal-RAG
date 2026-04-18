from ingestion.base import BaseExtractor
from ingestion.audio_extractor import AudioExtractor
import cv2
import os
import time
import torch
import PIL.Image
from transformers import AutoProcessor, AutoModelForCausalLM

# Florence-2 setup
MODEL_ID = "microsoft/Florence-2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    trust_remote_code=True,
    attn_implementation="eager",
).to(DEVICE)
model.eval()
print("Florence-2 loaded")

FRAME_INTERVAL = 10

class VideoExtractor(BaseExtractor):
    def __init__(self):
        self.audio_extractor = AudioExtractor()

    def _extract_audio(self, video_path: str) -> str:
        base = os.path.splitext(video_path)[0]
        audio_path = f"{base}_temp_audio.wav"
        os.system(f'ffmpeg -i "{video_path}" -q:a 0 -map a "{audio_path}" -y')
        return audio_path

    def _describe_frame(self, frame, timestamp: float) -> str:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(rgb_frame)

        prompt = "<DETAILED_CAPTION>"
        inputs = processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt"
        ).to(DEVICE, DTYPE)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
            )

        result = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            result,
            task=prompt,
            image_size=(pil_image.width, pil_image.height)
        )
        return parsed.get("<DETAILED_CAPTION>", "No description available")

    async def extract(self, file_path: str):
        chunks = []

        # Extract and transcribe audio
        audio_path = self._extract_audio(file_path)
        if os.path.exists(audio_path):
            audio_chunks = await self.audio_extractor.extract(audio_path)
            for chunk in audio_chunks:
                chunk["modality"] = "video_audio"
                chunk["video_source"] = file_path
            chunks.extend(audio_chunks)
            os.remove(audio_path)

        # Extract and describe frames
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_step = int(fps * FRAME_INTERVAL)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_step == 0:
                timestamp = frame_idx / fps
                print(f"Describing frame at {timestamp:.1f}s...")
                description = self._describe_frame(frame, timestamp)
                chunks.append({
                    "text": description,
                    "source": file_path,
                    "modality": "video_frame",
                    "timestamp": timestamp
                })
            frame_idx += 1

        cap.release()
        return chunks
