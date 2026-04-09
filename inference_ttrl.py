import copy
import os
import sys
from typing import List, Optional, Union

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIFFUSERS_SRC = os.path.join(PROJECT_ROOT, "src")
if LOCAL_DIFFUSERS_SRC not in sys.path:
    sys.path.insert(0, LOCAL_DIFFUSERS_SRC)

import clip
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor, CLIPTextModel

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from model.pipeline import StableDiffusionPipeline
from model.unet_2d_condition import UNet2DConditionModel
from utils.util import get_time_string


class TransformersCLIPAdapter:
    def __init__(self, model_name: str, device: torch.device):
        model_id_map = {
            "ViT-B/32": "openai/clip-vit-base-patch32",
            "ViT-B/16": "openai/clip-vit-base-patch16",
            "ViT-L/14": "openai/clip-vit-large-patch14",
        }
        model_id = model_id_map.get(model_name, "openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.model.eval()
        self.device = device

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor.tokenizer(texts, truncation=True, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.model.get_text_features(**inputs)

    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        pixel_values = self.processor.image_processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.no_grad():
            return self.model.get_image_features(pixel_values=pixel_values)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        return self.processor.image_processor(images=image, return_tensors="pt")["pixel_values"][0]


def load_clip_backbone(model_name: str, device: torch.device):
    if hasattr(clip, "load"):
        clip_model, preprocess = clip.load(model_name, device=device)
        clip_model.eval()
        return "openai-clip", clip_model, preprocess
    adapter = TransformersCLIPAdapter(model_name, device)
    return "transformers-clip", adapter, adapter.preprocess_image


def encode_clip_text(clip_backend_type: str, clip_model, texts: List[str], device: torch.device) -> torch.Tensor:
    if clip_backend_type == "openai-clip":
        tokens = clip.tokenize(texts, truncate=True).to(device)
        with torch.no_grad():
            return clip_model.encode_text(tokens)
    return clip_model.encode_text(texts)


def encode_clip_images(clip_backend_type: str, clip_model, images: List[Image.Image], preprocess, device: torch.device) -> torch.Tensor:
    if clip_backend_type == "openai-clip":
        image_inputs = torch.stack([preprocess(img) for img in images]).to(device)
        with torch.no_grad():
            return clip_model.encode_image(image_inputs)
    return clip_model.encode_image(images)


class TTRLOptimizer:
    def __init__(self, model, learning_rate=1e-6, beta=5000.0, epsilon_low=0.2, epsilon_high=0.2, delta=1.5):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.beta = beta
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.delta = delta

    def update(self, rewards, noisy_latents, timesteps, encoder_hidden_states, image_hidden_states=None):
        original_dtype = next(self.model.parameters()).dtype
        self.model = self.model.float()
        self.model.train()
        self.optimizer.zero_grad()

        noisy_latents = noisy_latents.float()
        timesteps = timesteps.float()
        encoder_hidden_states = encoder_hidden_states.float()
        if image_hidden_states is not None:
            if isinstance(image_hidden_states, dict):
                image_hidden_states = {key: value.float() for key, value in image_hidden_states.items()}
            else:
                image_hidden_states = image_hidden_states.float()

        with torch.no_grad():
            ref_model = copy.deepcopy(self.model)
            ref_model.eval()
            ref_model = ref_model.float()

        target_noise = torch.randn_like(noisy_latents).float()
        model_preds = self.model(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            image_hidden_states=image_hidden_states,
            return_dict=False,
        )[0]
        model_losses = F.mse_loss(model_preds, target_noise, reduction="none").mean(dim=[1, 2, 3])

        with torch.no_grad():
            ref_preds = ref_model(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                image_hidden_states=image_hidden_states,
                return_dict=False,
            )[0]
            ref_losses = F.mse_loss(ref_preds, target_noise, reduction="none").mean(dim=[1, 2, 3])

        log_prob_ratio = -(model_losses - ref_losses)
        prob_ratio = torch.exp(log_prob_ratio)
        clipped_ratios = torch.clamp(prob_ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.delta is not None:
            prob_ratio = torch.clamp(prob_ratio, max=self.delta)
            clipped_ratios = torch.clamp(clipped_ratios, max=self.delta)

        rewards_tensor = torch.tensor(rewards, device=noisy_latents.device, dtype=torch.float32)
        advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        unclipped_objective = prob_ratio * advantages
        clipped_objective = clipped_ratios * advantages
        grpo_loss = -torch.min(unclipped_objective, clipped_objective).mean()
        current_loss = model_losses.mean()
        ref_loss = ref_losses.mean()
        total_loss = grpo_loss + 0.1 * current_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.model = self.model.to(original_dtype)
        del ref_model
        torch.cuda.empty_cache()
        return total_loss.item(), current_loss.item(), ref_loss.item()


def majority_voting_reward_fn(images, prompt, ref_images, clip_backend_type, clip_model, preprocess, device, lambda_val=0.6):
    images = [img[0] if isinstance(img, list) and len(img) > 0 else img for img in images]
    text_features = encode_clip_text(clip_backend_type, clip_model, [prompt], device)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    ref_features = encode_clip_images(clip_backend_type, clip_model, [ref_images], preprocess, device)
    ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)
    image_features = encode_clip_images(clip_backend_type, clip_model, images, preprocess, device)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    clip_t_scores = torch.mm(image_features, text_features.t()).squeeze(1).detach().cpu().tolist()
    clip_i_scores = torch.mm(image_features, ref_features.t()).squeeze(1).detach().cpu().tolist()
    composite_rewards = [
        max(0.2, min(0.8, lambda_val * clip_t_scores[idx] + (1 - lambda_val) * clip_i_scores[idx]))
        for idx in range(len(images))
    ]
    best_idx = int(np.argmax(composite_rewards))
    return composite_rewards, best_idx, composite_rewards


def ttrl_inference(
    pretrained_model_path: str,
    logdir: str,
    prompt: str,
    ref_prompt: Union[str, List[str]],
    ref_image: Union[str, List[str]],
    num_inference_steps: int = 40,
    guidance_scale: float = 7.0,
    image_guidance_scale: float = 3.5,
    num_sample_per_prompt: int = 5,
    stage: str = "multi-image-condition",
    mixed_precision: Optional[str] = "fp16",
    ttrl_iterations: int = 5,
    learning_rate: float = 1e-6,
    lambda_val: float = 0.6,
    beta: float = 5000.0,
):
    time_string = get_time_string()
    logdir = f"{logdir}_{time_string}"
    os.makedirs(logdir, exist_ok=True)

    accelerator = Accelerator(mixed_precision=mixed_precision)
    device = accelerator.device
    clip_backend_type, clip_model, preprocess = load_clip_backbone("ViT-B/32", device)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    ttrl_optimizer = TTRLOptimizer(unet, learning_rate=learning_rate, beta=beta)

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
        except Exception as exc:
            print(f"Could not enable xformers: {exc}")

    unet, pipeline = accelerator.prepare(unet, pipeline)
    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    ref_img_path = ref_image[0]
    ref_images_pil = Image.open(ref_img_path).convert("RGB").resize((512, 512))
    image_tensor = transforms.ToTensor()(ref_images_pil) * 2.0 - 1.0
    ref_images_tensor = image_tensor.unsqueeze(0).to(device, dtype=weight_dtype)

    if isinstance(ref_prompt, str):
        ref_prompt = [ref_prompt]
    ref_prompt_ids = [
        tokenizer(
            prompt_item,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(device)
        for prompt_item in ref_prompt
    ]

    text_input = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        encoder_hidden_states = text_encoder(text_input)[0]

    with torch.no_grad():
        ref_latents = vae.encode(ref_images_tensor).latent_dist.sample() * 0.18215
        ref_noise = torch.randn_like(ref_latents)
        ref_timesteps = torch.randint(0, 1000, (ref_latents.shape[0],), device=device) // 10
        noisy_ref = scheduler.add_noise(ref_latents, ref_noise, ref_timesteps.unsqueeze(1))
        ref_text_encodings = torch.cat([text_encoder(ids)[0] for ids in ref_prompt_ids], dim=0)
        unet_output = unet(
            noisy_ref,
            ref_timesteps,
            encoder_hidden_states=ref_text_encodings,
            return_dict=False,
        )
        image_hidden_states = unet_output[1] if len(unet_output) >= 2 else unet_output[0]

    best_reward = -float("inf")
    best_image = None
    global_best_reward = -float("inf")
    global_best_image = None
    training_log = []

    for iteration in range(ttrl_iterations):
        print(f"TTRL iteration {iteration + 1}/{ttrl_iterations}")
        sample_seeds = sorted(torch.randint(0, 100000, (num_sample_per_prompt,)).cpu().tolist())
        generators = []
        for seed in sample_seeds:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            generators.append(generator)

        with torch.no_grad():
            outputs = pipeline(
                stage=stage,
                prompt=prompt,
                image_prompt=ref_images_tensor.unsqueeze(1),
                prev_prompt=ref_prompt,
                height=512,
                width=512,
                generator=generators,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                num_images_per_prompt=num_sample_per_prompt,
            ).images

        generated_images = [output[0] if isinstance(output, list) and len(output) > 0 else output for output in outputs]
        rewards_normalized, majority_idx, first_rewards = majority_voting_reward_fn(
            generated_images,
            prompt,
            ref_images_pil,
            clip_backend_type,
            clip_model,
            preprocess,
            device,
            lambda_val,
        )
        rewards = np.array(rewards_normalized)
        first_rewards = np.array(first_rewards)

        latents_list = []
        timesteps_list = []
        for image in generated_images:
            img_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device, dtype=weight_dtype) * 2.0 - 1.0
            with torch.no_grad():
                latent = vae.encode(img_tensor).latent_dist.sample() * 0.18215
            timestep = torch.randint(0, 1000, (1,), device=device)
            noise = torch.randn_like(latent)
            noisy_latent = scheduler.add_noise(latent, noise, timestep)
            latents_list.append(noisy_latent.squeeze(0))
            timesteps_list.append(timestep.squeeze(0))

        expanded_encoder_hs = encoder_hidden_states.repeat(len(generated_images), 1, 1)
        if isinstance(image_hidden_states, dict):
            expanded_image_hs = {key: value.repeat(len(generated_images), 1, 1) for key, value in image_hidden_states.items()}
        else:
            expanded_image_hs = image_hidden_states.repeat(len(generated_images), 1, 1)

        noisy_latents_tensor = torch.stack(latents_list)
        timesteps_tensor = torch.stack(timesteps_list)
        grpo_loss, current_loss, ref_loss = ttrl_optimizer.update(
            rewards,
            noisy_latents_tensor,
            timesteps_tensor,
            expanded_encoder_hs,
            expanded_image_hs,
        )

        current_best_idx = int(np.argmax(first_rewards))
        current_best_reward = float(first_rewards[current_best_idx])
        current_best_image = generated_images[current_best_idx]

        if current_best_reward > best_reward:
            best_reward = current_best_reward
            best_image = current_best_image
        if current_best_reward > global_best_reward:
            global_best_reward = current_best_reward
            global_best_image = current_best_image

        training_log.append(
            {
                "iteration": iteration,
                "avg_reward": float(np.mean(rewards)),
                "avg_first_reward": float(np.mean(first_rewards)),
                "best_reward": current_best_reward,
                "best_normalized_reward": float(rewards[current_best_idx]),
                "global_best_reward": global_best_reward,
                "grpo_loss": grpo_loss,
                "current_loss": current_loss,
                "ref_loss": ref_loss,
            }
        )

        print(f"Iteration {iteration}: avg reward = {np.mean(rewards):.3f}, GRPO loss = {grpo_loss:.6f}")

        iter_dir = os.path.join(logdir, f"iter_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)
        for index, (image, reward) in enumerate(zip(generated_images, rewards)):
            image.save(os.path.join(iter_dir, f"image_{index}_reward_{reward:.3f}.png"))
        if best_image is not None:
            best_image.save(os.path.join(iter_dir, "best_image.png"))

    if global_best_image is not None:
        global_best_image.save(os.path.join(logdir, "final_best_image.png"))
    with open(os.path.join(logdir, "log.txt"), "w") as handle:
        for item in training_log:
            handle.write(f"Iteration {item['iteration']}: {item}\n")
    return global_best_image, training_log


if __name__ == "__main__":
    pretrained_model_path = "./checkpoint_sc"
    logdir = "./ttrl_results"
    prompt = "A boy named Tom found a stray cat and held it in his arms."
    ref_prompt = ["In this cartoon, the left is a sitting boy and the right is a stray cat."]
    ref_image = ["./boy.png"]

    ttrl_inference(
        pretrained_model_path,
        logdir,
        prompt,
        ref_prompt,
        ref_image,
        num_sample_per_prompt=6,
        ttrl_iterations=5,
        learning_rate=1e-6,
        lambda_val=0.6,
    )
