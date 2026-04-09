import copy
import gc
import os
import sys
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIFFUSERS_SRC = os.path.join(PROJECT_ROOT, "src")
if LOCAL_DIFFUSERS_SRC not in sys.path:
    sys.path.insert(0, LOCAL_DIFFUSERS_SRC)

import clip
import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor, CLIPTextModel

from dataset_ttrl import StorySalonDataset, test_collate_fn
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from model.pipeline_ttrl import GRPOOptimizer, StableDiffusionGRPOPipeline
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


def final_evaluation_reward_fn(images, current_prompt, ref_image, prev_frame_image, clip_backend_type, clip_model, preprocess, device):
    images = [img[0] if isinstance(img, list) and len(img) > 0 else img for img in images]
    text_features = encode_clip_text(clip_backend_type, clip_model, [current_prompt], device)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    ref_features = encode_clip_images(clip_backend_type, clip_model, [ref_image], preprocess, device)
    ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)
    prev_features = None
    if prev_frame_image is not None:
        prev_features = encode_clip_images(clip_backend_type, clip_model, [prev_frame_image], preprocess, device)
        prev_features = prev_features / prev_features.norm(dim=-1, keepdim=True)

    final_rewards = []
    for image in images:
        with torch.no_grad(), autocast():
            image_features = encode_clip_images(clip_backend_type, clip_model, [image], preprocess, device)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_sim = torch.cosine_similarity(image_features, text_features).item()
            ref_sim = torch.cosine_similarity(image_features, ref_features).item()
            prev_sim = torch.cosine_similarity(image_features, prev_features).item() if prev_features is not None else 0.5
            final_reward = 0.5 * text_sim + 0.3 * ref_sim + 0.2 * prev_sim
            final_rewards.append(max(0.1, min(0.9, final_reward)))
    return final_rewards


def calculate_delta(total_grpo_iterations: int, max_gamma: float = 0.3) -> float:
    if total_grpo_iterations <= 1:
        return 0.0
    return round(max_gamma / (total_grpo_iterations - 1), 4)


def majority_voting_reward_fn(
    images,
    prompt,
    ref_images,
    clip_backend_type,
    clip_model,
    preprocess,
    device,
    lambda_val=0.65,
    gamma=0.0,
    global_best_feat=None,
    iteration=0,
    min_ref_sim_threshold=0.4,
    grpo_iterations=5,
):
    images = [img[0] if isinstance(img, list) and len(img) > 0 else img for img in images]
    text_features = encode_clip_text(clip_backend_type, clip_model, [prompt], device)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    ref_features = encode_clip_images(clip_backend_type, clip_model, [ref_images], preprocess, device)
    ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)
    if global_best_feat is not None:
        global_best_feat = global_best_feat.to(device) / global_best_feat.norm(dim=-1, keepdim=True)
    else:
        global_best_feat = ref_features

    image_features_list = []
    clip_t_scores = []
    current_ref_sims = []
    for image in images:
        with torch.no_grad(), autocast():
            image_features = encode_clip_images(clip_backend_type, clip_model, [image], preprocess, device)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features_list.append(image_features)
            clip_t_scores.append(torch.cosine_similarity(image_features, text_features).item())
            current_ref_sims.append(torch.cosine_similarity(image_features, ref_features).item())

    avg_ref_sim = np.mean(current_ref_sims)
    current_gamma = calculate_delta(grpo_iterations) * iteration
    composite_rewards = []
    for idx, feature in enumerate(image_features_list):
        ref_sim = current_ref_sims[idx]
        global_sim = torch.cosine_similarity(feature, global_best_feat).item()
        combined_i_score = (1 - current_gamma) * ref_sim + current_gamma * global_sim
        final_reward = lambda_val * clip_t_scores[idx] + (1 - lambda_val) * combined_i_score
        composite_rewards.append(max(0.2, min(0.8, final_reward)))

    best_idx = int(np.argmax(composite_rewards))
    best_feat = image_features_list[best_idx].clone()
    return composite_rewards, best_idx, best_feat, avg_ref_sim, current_gamma


def process_image_tensor(image_tensor):
    image = image_tensor.clone().detach()
    if image.min() < 0:
        image = (image + 1) / 2
    image = (image * 255).clamp(0, 255).byte()
    image_np = image.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(image_np).resize((512, 512))


def validate_grpo(
    pretrained_model_path: str,
    logdir: str,
    val_batch_size: int = 1,
    grpo_learning_rate: float = 1e-6,
    grpo_iterations: int = 5,
    num_samples_per_prompt: int = 3,
    lambda_val: float = 0.6,
    min_ref_sim_threshold: float = 0.4,
    clip_model_name: str = "ViT-B/32",
    window_size: int = 3,
):
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    time_string = get_time_string()
    logdir = f"{logdir}_{time_string}"
    os.makedirs(logdir, exist_ok=True)
    print(f"Log directory: {logdir}")

    clip_backend_type, clip_model, preprocess = load_clip_backbone(clip_model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")

    initial_unet_state = copy.deepcopy(unet.state_dict())
    grpo_optimizer = GRPOOptimizer(unet, learning_rate=grpo_learning_rate, history_size=2, epsilon_low=0.1, epsilon_high=0.1, delta=1.2)
    initial_grpo_optimizer_state = copy.deepcopy(grpo_optimizer.state_dict())

    pipeline = StableDiffusionGRPOPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        grpo_optimizer=grpo_optimizer,
    )
    pipeline.set_progress_bar_config(disable=True)

    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory-efficient attention")
        except Exception as exc:
            print(f"Could not enable xformers: {exc}")

    weight_dtype = torch.float16
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    unet, pipeline = accelerator.prepare(unet, pipeline)

    test_dataset = StorySalonDataset(root="/data/LLM_DATA/StorySalon/StorySalon", dataset_name="test")
    val_dataloader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4, collate_fn=test_collate_fn)
    val_dataloader = accelerator.prepare(val_dataloader)

    progress_bar = tqdm(total=len(test_dataset), desc="GRPO evaluation")

    def reset_to_initial_state():
        nonlocal unet, grpo_optimizer, pipeline
        print("Resetting model to the initial state M0")
        unet.load_state_dict(copy.deepcopy(initial_unet_state))
        grpo_optimizer.load_state_dict(copy.deepcopy(initial_grpo_optimizer_state))
        for param in unet.parameters():
            param.grad = None
        pipeline.unet = unet
        pipeline.grpo_optimizer = grpo_optimizer
        torch.cuda.empty_cache()
        gc.collect()

    for story_idx, batch in enumerate(val_dataloader):
        print(f"\nProcessing story {story_idx + 1}/{len(test_dataset)}")
        if story_idx > 0:
            torch.cuda.empty_cache()

        reset_to_initial_state()

        ref_images_tensor = batch["ref_image"].to(device, dtype=weight_dtype)
        ref_prompt = batch["ref_prompt"][0] if isinstance(batch["ref_prompt"], list) else batch["ref_prompt"]
        target_frames = batch["target_frames"][0] if isinstance(batch["target_frames"], list) else batch["target_frames"]
        story_id = batch["story_id"][0] if isinstance(batch["story_id"], list) else batch["story_id"]

        print(f"\n==================== Start story {story_idx + 1} ====================")
        print(f"Story ID: {story_id}")
        print(f"Number of target frames: {len(target_frames)}")

        if ref_images_tensor.dim() == 5:
            ref_images_tensor = ref_images_tensor.squeeze(0)

        ref_images_pil = [process_image_tensor(ref_images_tensor[i]) for i in range(ref_images_tensor.shape[0])]
        primary_ref_image = ref_images_pil[0] if ref_images_pil else None
        all_ref_prompts = batch["all_ref_prompts"][0] if isinstance(batch["all_ref_prompts"], list) else batch["all_ref_prompts"]

        with torch.no_grad():
            ref_prompt_ids = tokenizer(
                ref_prompt,
                truncation=True,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(device)
            ref_text_encodings = text_encoder(ref_prompt_ids)[0]

        with torch.no_grad():
            batch_size = 2
            image_hidden_states_list = []
            for start in range(0, ref_images_tensor.shape[0], batch_size):
                batch_images = ref_images_tensor[start : start + batch_size]
                ref_latents = vae.encode(batch_images).latent_dist.sample() * 0.18215
                ref_noise = torch.randn_like(ref_latents)
                ref_timesteps = torch.randint(0, 1000, (ref_latents.shape[0],), device=device) // 10
                noisy_ref = scheduler.add_noise(ref_latents, ref_noise, ref_timesteps)
                batch_text_encodings = ref_text_encodings.repeat(ref_latents.shape[0], 1, 1)
                unet_output = unet(noisy_ref, ref_timesteps, encoder_hidden_states=batch_text_encodings, return_dict=False)
                batch_image_hidden_states = unet_output[1] if len(unet_output) >= 2 else unet_output[0]
                image_hidden_states_list.append(batch_image_hidden_states)

            if isinstance(image_hidden_states_list[0], dict):
                image_hidden_states = {}
                for key in image_hidden_states_list[0]:
                    image_hidden_states[key] = torch.cat([item[key] for item in image_hidden_states_list], dim=0)
            else:
                image_hidden_states = torch.cat(image_hidden_states_list, dim=0)

        frame_best_features = [None] * len(target_frames)
        frame_best_images = [None] * len(target_frames)
        frame_best_rewards = [-float("inf")] * len(target_frames)
        current_best_story = [None] * len(target_frames)
        all_iterations_best_candidates = [[] for _ in range(len(target_frames))]

        story_output_folder = os.path.join(logdir, f"story_{story_idx}")
        os.makedirs(story_output_folder, exist_ok=True)

        for iteration in range(grpo_iterations):
            print(f"\nStory {story_idx + 1}, GRPO iteration {iteration + 1}/{grpo_iterations}")
            iter_output_folder = os.path.join(story_output_folder, f"iteration_{iteration}")
            os.makedirs(iter_output_folder, exist_ok=True)
            all_frame_rewards = [[] for _ in range(len(target_frames))]
            all_frame_best_indices = [None] * len(target_frames)

            max_prev_frames = min(window_size - 1, len(target_frames))
            for frame_idx in range(max_prev_frames):
                target_frame = target_frames[frame_idx]
                prompt = target_frame["prompt"]
                print(f"Generating initial frame {frame_idx + 1}/{max_prev_frames}")

                with torch.no_grad():
                    text_input = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                    ).input_ids.to(device)
                    encoder_hidden_states = text_encoder(text_input)[0]

                generators = [
                    torch.Generator(device=device).manual_seed(
                        torch.randint(0, 1000000, (1,)).item() + iteration * 1000 + frame_idx
                    )
                    for _ in range(num_samples_per_prompt)
                ]

                with torch.no_grad():
                    generated_images = pipeline(
                        stage="multi-image-condition",
                        prompt=prompt,
                        image_prompt=ref_images_tensor.unsqueeze(0),
                        prev_prompt=all_ref_prompts,
                        num_inference_steps=30,
                        guidance_scale=7.0,
                        height=512,
                        width=512,
                        image_guidance_scale=3.5,
                        generator=generators,
                        num_images_per_prompt=num_samples_per_prompt,
                    ).images

                frame_candidates = [img[0] if isinstance(img, list) and len(img) > 0 else img for img in generated_images]
                rewards, best_idx, best_feat, _, _ = majority_voting_reward_fn(
                    frame_candidates,
                    prompt,
                    primary_ref_image,
                    clip_backend_type,
                    clip_model,
                    preprocess,
                    device,
                    lambda_val=lambda_val,
                    gamma=calculate_delta(grpo_iterations) * iteration,
                    global_best_feat=frame_best_features[frame_idx],
                    iteration=iteration,
                    min_ref_sim_threshold=min_ref_sim_threshold,
                    grpo_iterations=grpo_iterations,
                )

                all_frame_rewards[frame_idx] = rewards
                all_frame_best_indices[frame_idx] = best_idx

                if rewards[best_idx] > frame_best_rewards[frame_idx]:
                    frame_best_features[frame_idx] = best_feat
                    frame_best_images[frame_idx] = frame_candidates[best_idx]
                    frame_best_rewards[frame_idx] = rewards[best_idx]
                    current_best_story[frame_idx] = frame_candidates[best_idx]

                all_iterations_best_candidates[frame_idx].append(frame_candidates[best_idx])
                frame_output_folder = os.path.join(iter_output_folder, f"frame_{frame_idx}")
                os.makedirs(frame_output_folder, exist_ok=True)
                frame_candidates[best_idx].save(os.path.join(frame_output_folder, f"best_reward_{rewards[best_idx]:.6f}.png"))
                with open(os.path.join(frame_output_folder, "candidates_rewards.log"), "w") as handle:
                    handle.write(f"Frame {frame_idx} candidate rewards\n")
                    handle.write(f"Prompt: {prompt[:100]}...\n")
                    handle.write(f"Best candidate index: {best_idx}\n")
                    handle.write(f"Best candidate reward: {rewards[best_idx]:.10f}\n")
                    handle.write("\nAll candidate rewards:\n")
                    for cand_idx, reward in enumerate(rewards):
                        handle.write(f"Candidate_{cand_idx}: {reward:.10f}\n")

            for window_start in range(max(len(target_frames) - window_size + 1, 0)):
                window_end = window_start + window_size
                target_frame_idx = window_end - 1
                if target_frame_idx >= len(target_frames) or window_start >= len(ref_images_pil):
                    continue

                prompt = target_frames[target_frame_idx]["prompt"]
                print(f"Window [{window_start}:{window_end - 1}] -> generating frame {target_frame_idx}")

                head_gt_image = ref_images_pil[window_start]
                head_tensor = transforms.ToTensor()(head_gt_image).unsqueeze(0)
                head_tensor = head_tensor * 2.0 - 1.0
                head_tensor = head_tensor.to(device, dtype=weight_dtype)
                generators = [
                    torch.Generator(device=device).manual_seed(
                        torch.randint(0, 1000000, (1,)).item() + iteration * 1000 + target_frame_idx
                    )
                    for _ in range(num_samples_per_prompt)
                ]

                with torch.no_grad():
                    generated_images = pipeline(
                        stage="multi-image-condition",
                        prompt=prompt,
                        image_prompt=head_tensor.unsqueeze(0),
                        prev_prompt=[all_ref_prompts[window_start]],
                        num_inference_steps=50,
                        guidance_scale=7.0,
                        height=512,
                        width=512,
                        image_guidance_scale=3.5,
                        generator=generators,
                        num_images_per_prompt=num_samples_per_prompt,
                    ).images

                frame_candidates = [img[0] if isinstance(img, list) and len(img) > 0 else img for img in generated_images]
                rewards, best_idx, best_feat, _, _ = majority_voting_reward_fn(
                    frame_candidates,
                    prompt,
                    head_gt_image,
                    clip_backend_type,
                    clip_model,
                    preprocess,
                    device,
                    lambda_val=lambda_val,
                    gamma=calculate_delta(grpo_iterations) * iteration,
                    global_best_feat=frame_best_features[target_frame_idx],
                    iteration=iteration,
                    min_ref_sim_threshold=min_ref_sim_threshold,
                    grpo_iterations=grpo_iterations,
                )

                all_frame_rewards[target_frame_idx] = rewards
                all_frame_best_indices[target_frame_idx] = best_idx

                if rewards[best_idx] > frame_best_rewards[target_frame_idx]:
                    frame_best_features[target_frame_idx] = best_feat
                    frame_best_images[target_frame_idx] = frame_candidates[best_idx]
                    frame_best_rewards[target_frame_idx] = rewards[best_idx]
                    current_best_story[target_frame_idx] = frame_candidates[best_idx]

                all_iterations_best_candidates[target_frame_idx].append(frame_candidates[best_idx])
                frame_output_folder = os.path.join(iter_output_folder, f"frame_{target_frame_idx}")
                os.makedirs(frame_output_folder, exist_ok=True)
                frame_candidates[best_idx].save(os.path.join(frame_output_folder, f"best_reward_{rewards[best_idx]:.6f}.png"))
                with open(os.path.join(frame_output_folder, "candidates_rewards.log"), "w") as handle:
                    handle.write(f"Frame {target_frame_idx} candidate rewards\n")
                    handle.write(f"Prompt: {prompt[:100]}...\n")
                    handle.write(f"Window: [{window_start}:{window_end - 1}]\n")
                    handle.write(f"Best candidate index: {best_idx}\n")
                    handle.write(f"Best candidate reward: {rewards[best_idx]:.10f}\n")
                    handle.write("\nAll candidate rewards:\n")
                    for cand_idx, reward in enumerate(rewards):
                        handle.write(f"Candidate_{cand_idx}: {reward:.10f}\n")

            story_rewards = []
            for frame_idx in range(len(target_frames)):
                if all_frame_rewards[frame_idx] and all_frame_best_indices[frame_idx] is not None:
                    story_rewards.append(all_frame_rewards[frame_idx][all_frame_best_indices[frame_idx]])
            avg_story_reward = np.mean(story_rewards) if story_rewards else 0.0
            print(f"Iteration {iteration}: average story reward = {avg_story_reward:.6f}")

            for frame_idx, best_img in enumerate(current_best_story):
                if best_img is not None:
                    best_img.save(os.path.join(iter_output_folder, f"frame_{frame_idx}_best.png"))

            with open(os.path.join(iter_output_folder, "iteration_summary.log"), "w") as handle:
                handle.write(f"Iteration {iteration} summary\n")
                handle.write(f"Story ID: {story_id}\n")
                handle.write(f"Number of target frames: {len(target_frames)}\n")
                handle.write(f"Average iteration reward: {avg_story_reward:.10f}\n")
                handle.write("\nBest reward per frame\n")
                for frame_idx, reward in enumerate(story_rewards):
                    handle.write(f"Frame_{frame_idx}: {reward:.10f}\n")

        print(f"\nStarting final evaluation over {grpo_iterations} iteration-level candidates...")
        final_best_images = [None] * len(target_frames)
        final_best_rewards = [-float("inf")] * len(target_frames)
        final_best_iterations = [None] * len(target_frames)

        for frame_idx in range(len(target_frames)):
            print(f"Final evaluation for frame {frame_idx}...")
            frame_candidates = all_iterations_best_candidates[frame_idx]
            if not frame_candidates:
                continue
            if frame_idx < window_size - 1:
                ref_image = primary_ref_image
                print("  Using the initial reference image")
            else:
                window_start = frame_idx - (window_size - 1)
                if window_start < len(ref_images_pil):
                    ref_image = ref_images_pil[window_start]
                    print(f"  Using window start frame {window_start} as reference")
                else:
                    ref_image = primary_ref_image
                    print("  Window start frame is out of range, falling back to the initial reference image")

            prev_frame_image = final_best_images[frame_idx - 1] if frame_idx > 0 and final_best_images[frame_idx - 1] is not None else None
            if prev_frame_image is not None:
                print("  Using the best image from the previous frame as the temporal reference")
            else:
                print("  No previous-frame reference is available; using a neutral temporal score")

            current_prompt = target_frames[frame_idx]["prompt"]
            final_rewards = final_evaluation_reward_fn(
                frame_candidates,
                current_prompt,
                ref_image,
                prev_frame_image,
                clip_backend_type,
                clip_model,
                preprocess,
                device,
            )
            best_final_idx = int(np.argmax(final_rewards))
            final_best_images[frame_idx] = frame_candidates[best_final_idx]
            final_best_rewards[frame_idx] = final_rewards[best_final_idx]
            final_best_iterations[frame_idx] = best_final_idx
            print(f"Frame {frame_idx}: final best reward = {final_rewards[best_final_idx]:.6f} (from iteration {best_final_idx})")

        comparison_folder = os.path.join(story_output_folder, "final_comparison")
        os.makedirs(comparison_folder, exist_ok=True)
        best_sequence_folder = os.path.join(story_output_folder, "final_best_sequence")
        os.makedirs(best_sequence_folder, exist_ok=True)

        for frame_idx in range(len(target_frames)):
            if frame_idx < len(final_best_images) and final_best_images[frame_idx] is not None:
                final_best_images[frame_idx].save(os.path.join(comparison_folder, f"frame_{frame_idx}_final_best_gen.png"))
            if frame_idx < len(ref_images_pil):
                ref_images_pil[frame_idx].save(os.path.join(comparison_folder, f"frame_{frame_idx}_gt.png"))

        for frame_idx, best_img in enumerate(final_best_images):
            if best_img is not None:
                iteration_source = final_best_iterations[frame_idx] if frame_idx < len(final_best_iterations) else "unknown"
                best_img.save(
                    os.path.join(
                        best_sequence_folder,
                        f"frame_{frame_idx}_final_best_reward_{final_best_rewards[frame_idx]:.6f}_iter{iteration_source}.png",
                    )
                )

        with open(os.path.join(story_output_folder, "story_final_log.txt"), "w") as handle:
            handle.write("Final story summary after final evaluation\n")
            handle.write(f"Story ID: {story_id}\n")
            handle.write(f"Number of target frames: {len(target_frames)}\n")
            handle.write(f"GRPO iterations: {grpo_iterations}\n")
            handle.write(f"Window size: {window_size}\n")
            handle.write("\nFinal best reward per frame:\n")
            for frame_idx, reward in enumerate(final_best_rewards):
                iteration_source = final_best_iterations[frame_idx] if frame_idx < len(final_best_iterations) else "unknown"
                handle.write(f"Frame_{frame_idx}: {reward:.10f} (from iteration {iteration_source})\n")
            valid_rewards = [reward for reward in final_best_rewards if reward > -float("inf")]
            handle.write(f"\nAverage final reward: {np.mean(valid_rewards):.10f}\n")
            handle.write("\nFinal evaluation weights:\n")
            handle.write("- Text alignment: 0.5\n")
            handle.write("- Reference consistency: 0.3\n")
            handle.write("- Temporal coherence: 0.2\n")

        valid_rewards = [reward for reward in final_best_rewards if reward > -float("inf")]
        print(f"\nFinal evaluation complete. Average reward: {np.mean(valid_rewards):.6f}")
        progress_bar.update(1)
        torch.cuda.empty_cache()
        gc.collect()

    accelerator.end_training()


if __name__ == "__main__":
    validate_grpo(
        pretrained_model_path="./checkpoint_sc",
        logdir="./results",
        grpo_learning_rate=1e-6,
        grpo_iterations=5,
        num_samples_per_prompt=3,
        lambda_val=0.65,
        min_ref_sim_threshold=0.45,
        window_size=3,
    )
