import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
from io import BytesIO
import base64
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

class InferlessPythonModel:

    def initialize(self):
        model_id = "stabilityai/stable-video-diffusion-img2vid"
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
        # self.pipe.enable_model_cpu_offload()
        self.pipe.to("cuda")
        self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        self.pipe.vae = torch.compile(self.pipe.vae, mode="reduce-overhead", fullgraph=True)
    
    def infer(self,inputs):
        image_url = inputs['image_url']
        image = load_image(image_url)
        image = image.resize((1024, 576))
        generator = torch.manual_seed(42)
        frames = self.pipe(image, decode_chunk_size=8, generator=generator).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)

        with open("generated.mp4", "rb") as video_file:
            video_binary_data = video_file.read()
            video_bytes_io = BytesIO(video_binary_data)
            base64_encoded_data = base64.b64encode(video_bytes_io.read())
            base64_string = base64_encoded_data.decode("utf-8")

        return {"generated_video": base64_string}

    def finalize(self):
        pass
