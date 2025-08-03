#批量推理图片
import os
from typing import Optional, Dict
import torch
from PIL import Image
import torchvision
import numpy as np
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDIMScheduler
from datasetnew import StorySalonDataset

from utils.util import get_time_string
from model.unet_2d_condition import UNet2DConditionModel
from model.pipeline import StableDiffusionPipeline
from PIL import Image

def save_original_image(image_tensor, output_path):
    # 将图像转换为范围 [0, 255]
    image = image_tensor.clone().detach()
    if image.min() < 0:  # 如果图像在 [-1, 1] 范围
        image = (image + 1) / 2  # 转换到 [0, 1] 范围
    image = (image * 255).clamp(0, 255).byte()  # 转换到 [0, 255] 范围

    # 将 PyTorch 张量转换为 NumPy 数组
    image_np = image.permute(1, 2, 0).cpu().numpy()  # 调整维度顺序为 (H, W, C)

    # 使用 PIL 保存图像
    pil_image = Image.fromarray(image_np)
    pil_image.save(output_path)

def post_process_generated_image(generated_image):
    image = np.asarray(generated_image).astype(np.float32) / 255.0  # 转换到 [0, 1]
    image = (image * 2.0 - 1.0)  # 转换到 [-1, 1]
    image = (image + 1.0) / 2.0  # 再次转换到 [0, 1]
    image = (image * 255.0).clip(0, 255).astype(np.uint8)  # 裁剪并量化到 [0, 255]
    return Image.fromarray(image)

def validate(
        pretrained_model_path: str,
        logdir: str,
        validation_steps: int = 1000,
        validation_sample_logger: Optional[Dict] = None,
        val_batch_size: int = 1,
):
    accelerator = Accelerator()
    if accelerator.is_main_process:
        os.makedirs(logdir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")

    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    )
    pipeline.set_progress_bar_config(disable=True)

    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"Could not enable memory efficient attention. Make sure xformers is installed: {e}")

    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    unet.to(accelerator.device)

    test_dataset = StorySalonDataset(root="/data/LLM_DATA/StorySalon/StorySalon", dataset_name='test')
    
    val_dataloader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=1)

    val_dataloader = accelerator.prepare(val_dataloader)

    progress_bar = tqdm(range(len(val_dataloader)), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Testing Steps")

    for step, batch in enumerate(val_dataloader):
        vae.eval()
        text_encoder.eval()
        unet.eval()

        image = batch["image"].to(dtype=torch.float32)  # 原始图像
        prompt = batch["prompt"]
        ref_prompt = batch["ref_prompt"]
        ref_images=batch["ref_image"]
       
        processed_ref_images = []

        for ref_image in ref_images:
            
            ref_image = ref_image.squeeze(0)  # 去除额外的 batch 维度 这一步需要
            processed_ref_images.append(ref_image)

        ref_images = torch.stack(processed_ref_images).float()
        
        ref_images = ref_images.unsqueeze(0)  
        
        prompt_ids = tokenizer(prompt, truncation=True, padding="max_length",
                               max_length=tokenizer.model_max_length, return_tensors="pt").input_ids

        with torch.no_grad():
            # 推理过程
            generated_images = pipeline(
                stage = 'multi-image-condition',
                prompt=prompt,
                image_prompt = ref_images,
                prev_prompt = ref_prompt,
                num_inference_steps=40,
                guidance_scale=7,
                height = 512,
                width = 512,
                image_guidance_scale = 3.5
            ).images
            flattened_generated_images = [img for sublist in generated_images for img in sublist] if isinstance(generated_images[0], list) else generated_images
          
            for i, img in enumerate(flattened_generated_images):
                output_folder = os.path.join(logdir, f"{step}_{i}")
                os.makedirs(output_folder, exist_ok=True)

                img.save(os.path.join(output_folder, f"generated_image_{step}_{i}.png"))

                with open(os.path.join(output_folder, f"ref_text_{step}_{i}.txt"), "w") as text_file:
                    text_file.write(prompt[i])

                gt_image_path = os.path.join(output_folder, f"gt_image_{step}_{i}.png")

                save_original_image(image[i], gt_image_path)

        if validation_sample_logger is not None and accelerator.is_main_process and step % validation_steps == 0:
            with autocast(): 
                validation_sample_logger.log_sample_images(batch=batch, pipeline=pipeline, device=accelerator.device,
                                                       step=step)
        progress_bar.update(1)

    accelerator.end_training()


if __name__ == "__main__":
    pretrained_model_path ="./checkpoint"
    logdir = "./inference_results"  
    validate(pretrained_model_path=pretrained_model_path, logdir=logdir)

#CUDA_VISIBLE_DEVICES=0 accelerate launch test_image.py