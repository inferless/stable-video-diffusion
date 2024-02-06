from io import BytesIO
import torch
import requests
import os
from glob import glob
from pathlib import Path
from typing import Optional

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image

import uuid
import random
from huggingface_hub import hf_hub_download
import base64


class InferlessPythonModel:
    
    def initialize(self):
        self.pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        self.pipe.to("cuda")
        self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        self.pipe.vae = torch.compile(self.pipe.vae, mode="reduce-overhead", fullgraph=True)
        self.seed = 42
        self.randomize_seed = True,
        self.motion_bucket_id = 127,
        self.fps_id = 6,
        self.version = "svd_xt",
        self.cond_aug = 0.02,
        self.decoding_t = 3,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
        self.device = "cuda",
        self.output_folder = "outputs",

    def infer(self,inputs):
        image_url = inputs['image_url']
        max_64_bit_int = 2**63 - 1
        image = self.resize_image(image_url)

        if image.mode == "RGBA":
            image = image.convert("RGB")

        if(self.randomize_seed):
            seed = random.randint(0, max_64_bit_int)
        generator = torch.manual_seed(seed)

        os.makedirs(self.output_folder, exist_ok=True)
        base_count = len(glob(os.path.join(self.output_folder, "*.mp4")))
        video_path = os.path.join(self.output_folder, f"{base_count:06d}.mp4")

        frames = self.pipe(image, decode_chunk_size=self.decoding_t, generator=generator, motion_bucket_id=self.motion_bucket_id, noise_aug_strength=0.1).frames[0]
        export_to_video(frames, video_path, fps=self.fps_id)
        torch.manual_seed(seed)

        #Video convert to base64
        with open(video_path, "rb") as video_file:
            video_binary_data = video_file.read()
            video_bytes_io = BytesIO(video_binary_data)
            base64_encoded_data = base64.b64encode(video_bytes_io.read())
            base64_string = base64_encoded_data.decode("utf-8")

        return {"generated_video": base64_string}

    def resize_image(self,image_url, output_size=(1024, 576)):
        # Calculate aspect ratio
        response = requests.get(image_url)
        image_data = BytesIO(response.content)

        # Open the image using PIL
        image = Image.open(image_data)
        target_aspect = output_size[0] / output_size[1]  # Aspect ratio of the desired size
        image_aspect = image.width / image.height  # Aspect ratio of the original image

        # Resize then crop if the original image is larger
        if image_aspect > target_aspect:
            # Resize the image to match the target height, maintaining aspect ratio
            new_height = output_size[1]
            new_width = int(new_height * image_aspect)
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            # Calculate coordinates for cropping
            left = (new_width - output_size[0]) / 2
            top = 0
            right = (new_width + output_size[0]) / 2
            bottom = output_size[1]
        else:
            # Resize the image to match the target width, maintaining aspect ratio
            new_width = output_size[0]
            new_height = int(new_width / image_aspect)
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            # Calculate coordinates for cropping
            left = 0
            top = (new_height - output_size[1]) / 2
            right = output_size[0]
            bottom = (new_height + output_size[1]) / 2

        # Crop the image
        cropped_image = resized_image.crop((left, top, right, bottom))
        return cropped_image

    def finalize(self,*args):
        pass