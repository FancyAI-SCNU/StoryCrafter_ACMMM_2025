import copy
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import is_accelerate_available, logging
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer

from model.unet_2d_condition import UNet2DConditionModel

logger = logging.get_logger(__name__)


class GRPOOptimizer:
    def __init__(
        self,
        model,
        learning_rate=2e-5,
        optimizer_class=torch.optim.AdamW,
        epsilon_low=0.2,
        epsilon_high=0.2,
        delta=1.5,
        history_size=3,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self.optimizer = optimizer_class(model.parameters(), lr=learning_rate)
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.delta = delta
        self.clip_stats = {"low_clip_ratio": [], "high_clip_ratio": [], "total_clip_ratio": []}
        self.old_policy_performance = []
        self.history_size = history_size

    def state_dict(self):
        return {
            "optimizer_state": self.optimizer.state_dict(),
            "clip_stats": copy.deepcopy(self.clip_stats),
            "old_policy_performance": copy.deepcopy(self.old_policy_performance),
            "model_state": copy.deepcopy(self.model.state_dict()),
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer_state"])
        self.clip_stats = copy.deepcopy(state_dict["clip_stats"])
        self.old_policy_performance = copy.deepcopy(state_dict["old_policy_performance"])
        self.model.load_state_dict(state_dict["model_state"])

    def reset(self):
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.learning_rate)
        self.optimizer.zero_grad(set_to_none=True)
        self.clip_stats = {"low_clip_ratio": [], "high_clip_ratio": [], "total_clip_ratio": []}
        self.old_policy_performance = []
        for param in self.model.parameters():
            param.grad = None

    def compute_diffusion_sequence_quality(self, pred_noise, target_noise):
        pred_noise = pred_noise.float()
        target_noise = target_noise.float()
        mse_loss = F.mse_loss(pred_noise, target_noise, reduction="none").mean(dim=[1, 2, 3])
        sequence_quality = torch.exp(-mse_loss)
        return sequence_quality, mse_loss

    def compute_group_advantages(self, rewards):
        device = next(self.model.parameters()).device
        rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
        return (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

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
                image_hidden_states_float = {key: value.float() for key, value in image_hidden_states.items()}
            else:
                image_hidden_states_float = image_hidden_states.float()
        else:
            image_hidden_states_float = None

        with torch.no_grad():
            ref_model = copy.deepcopy(self.model)
            ref_model.eval()
            ref_model = ref_model.float()

        target_noise = torch.randn_like(noisy_latents).float()

        model_preds = self.model(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            image_hidden_states=image_hidden_states_float,
            return_dict=False,
        )[0]

        with torch.no_grad():
            ref_preds = ref_model(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                image_hidden_states=image_hidden_states_float,
                return_dict=False,
            )[0]

        model_losses = F.mse_loss(model_preds, target_noise, reduction="none").mean(dim=[1, 2, 3])
        ref_losses = F.mse_loss(ref_preds, target_noise, reduction="none").mean(dim=[1, 2, 3])

        log_prob_ratio = -(model_losses - ref_losses)
        prob_ratio = torch.exp(log_prob_ratio)

        clip_low = 1 - self.epsilon_low
        clip_high = 1 + self.epsilon_high
        clipped_ratios = torch.clamp(prob_ratio, clip_low, clip_high)

        if self.delta is not None:
            prob_ratio = torch.clamp(prob_ratio, max=self.delta)
            clipped_ratios = torch.clamp(clipped_ratios, max=self.delta)

        advantages = self.compute_group_advantages(rewards)

        is_low_clipped = (prob_ratio < clip_low) & (advantages < 0)
        is_high_clipped = (prob_ratio > clip_high) & (advantages > 0)
        is_clipped = is_low_clipped | is_high_clipped

        self.clip_stats["low_clip_ratio"].append(is_low_clipped.float().mean().item())
        self.clip_stats["high_clip_ratio"].append(is_high_clipped.float().mean().item())
        self.clip_stats["total_clip_ratio"].append(is_clipped.float().mean().item())

        unclipped_objective = prob_ratio * advantages
        clipped_objective = clipped_ratios * advantages
        grpo_objective = torch.min(unclipped_objective, clipped_objective)

        base_loss = model_losses.mean()
        grpo_loss = -torch.mean(grpo_objective)
        total_loss = grpo_loss + 0.1 * base_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        sequence_quality, _ = self.compute_diffusion_sequence_quality(model_preds, target_noise)
        for quality in sequence_quality.detach().cpu().numpy():
            if len(self.old_policy_performance) >= self.history_size:
                self.old_policy_performance.pop(0)
            self.old_policy_performance.append(quality)

        self.model = self.model.to(original_dtype)
        del ref_model
        torch.cuda.empty_cache()

        return total_loss.item(), grpo_loss.item(), base_loss.item()


class StableDiffusionGRPOPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        grpo_optimizer: Optional[GRPOOptimizer] = None,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.grpo_optimizer = grpo_optimizer
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def _clean_up_cached_inputs(self):
        if hasattr(self.text_encoder, "_cache"):
            self.text_encoder._cache = {}
        if hasattr(self.vae, "_cache"):
            self.vae._cache = {}
        for attr in ["text_embeddings", "prev_text_embeddings", "image_prompts", "zero_image_prompts"]:
            if hasattr(self, attr):
                delattr(self, attr)
        torch.cuda.empty_cache()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")
        for cpu_offloaded_model in [self.unet, self.text_encoder]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)[0]
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device), attention_mask=attention_mask)[0]
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents):
        batch = latents.shape[0]
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = rearrange(image, "b c h w -> b h w c", b=batch)
        return image.cpu().float().numpy()

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        if callback_steps is None or not isinstance(callback_steps, int) or callback_steps <= 0:
            raise ValueError(f"`callback_steps` has to be a positive integer but is {callback_steps}.")

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device
            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected inputs shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_training_data(self, generated_images: List, prompts: List[str], device: torch.device, weight_dtype: torch.dtype):
        latents_list = []
        timesteps_list = []
        encoder_hs_list = []

        for frame_images, frame_prompt in zip(generated_images, prompts):
            for cand_img in frame_images:
                if hasattr(cand_img, "convert"):
                    cand_img = cand_img.convert("RGB")
                    img_array = np.array(cand_img)
                    img_tensor = torch.from_numpy(img_array).float() / 255.0
                    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                else:
                    img_tensor = cand_img

                img_tensor = img_tensor * 2.0 - 1.0
                img_tensor = img_tensor.to(device, dtype=weight_dtype)
                with torch.no_grad():
                    latent = self.vae.encode(img_tensor).latent_dist.sample() * 0.18215

                timestep = torch.randint(0, 1000, (1,), device=device)
                noise = torch.randn_like(latent)
                noisy_latent = self.scheduler.add_noise(latent, noise, timestep)

                text_input = self.tokenizer(frame_prompt, return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    encoder_hidden_states = self.text_encoder(text_input)[0]

                latents_list.append(noisy_latent)
                timesteps_list.append(timestep)
                encoder_hs_list.append(encoder_hidden_states)

        noisy_latents_tensor = torch.cat(latents_list, dim=0)
        timesteps_tensor = torch.cat(timesteps_list, dim=0)
        encoder_hs_tensor = torch.cat(encoder_hs_list, dim=0)
        return noisy_latents_tensor, timesteps_tensor, encoder_hs_tensor

    def update_model_with_grpo_rewards(
        self,
        grpo_rewards: List[float],
        generated_images: List[List],
        prompts: List[str],
        device: torch.device,
        weight_dtype: torch.dtype,
        image_hidden_states: Optional[Any] = None,
    ):
        if self.grpo_optimizer is None:
            return None, None, None

        noisy_latents, timesteps, encoder_hidden_states = self.prepare_training_data(
            generated_images,
            prompts,
            device,
            weight_dtype,
        )

        total_loss, grpo_loss, base_loss = self.grpo_optimizer.update(
            rewards=grpo_rewards,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            image_hidden_states=image_hidden_states,
        )
        return total_loss, grpo_loss, base_loss

    @torch.no_grad()
    def __call__(
        self,
        stage: str,
        prompt: Union[str, List[str]],
        image_prompt: Optional[torch.FloatTensor] = None,
        prev_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 3.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        self.check_inputs(prompt, height, width, callback_steps)

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        text_embeddings = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )

        ref_image_num = 0
        if image_prompt is not None:
            image_prompt = image_prompt.to(device=device, dtype=text_embeddings.dtype)
            if image_prompt.dim() == 3:
                image_prompt = image_prompt.unsqueeze(0)
            t_image_prompts = torch.transpose(image_prompt, 0, 1)
            ref_image_num = t_image_prompts.shape[0]

        if isinstance(prev_prompt, str):
            prev_prompt = [prev_prompt]
        prev_text_embeddings = []
        if prev_prompt is not None and ref_image_num > 0:
            if len(prev_prompt) < ref_image_num:
                prev_prompt = prev_prompt + [prev_prompt[-1]] * (ref_image_num - len(prev_prompt))
            for prompt_item in prev_prompt:
                prev_text_embeddings.append(
                    self._encode_prompt(
                        prompt_item,
                        device,
                        num_images_per_prompt,
                        do_classifier_free_guidance,
                        negative_prompt,
                    )
                )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        zero_image_prompts = []
        image_prompts = []
        if image_prompt is not None:
            zero_image_prompt = t_image_prompts[0] * 0
            if zero_image_prompt.dim() == 3:
                zero_image_prompt = zero_image_prompt.unsqueeze(0)
            zero_image_prompt = self.vae.encode(zero_image_prompt).latent_dist.sample()
            zero_image_prompt = zero_image_prompt * 0.18215
            zero_image_prompt = zero_image_prompt.repeat(num_images_per_prompt, 1, 1, 1)
            for _ in range(ref_image_num):
                zero_image_prompts.append(zero_image_prompt)

            for t_image_prompt in t_image_prompts:
                if t_image_prompt.dim() == 3:
                    t_image_prompt = t_image_prompt.unsqueeze(0)
                new_image_prompt = self.vae.encode(t_image_prompt).latent_dist.sample()
                new_image_prompt = new_image_prompt * 0.18215
                new_image_prompt = new_image_prompt.repeat(num_images_per_prompt, 1, 1, 1)
                image_prompts.append(new_image_prompt)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        noise = torch.randn_like(latents)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_index, timestep in enumerate(timesteps):
                if image_prompt is not None and ref_image_num > 0:
                    ref_timestep = (timestep / 10).long()
                    img_conditions = []

                    for image_index in range(ref_image_num):
                        if stage == "auto-regressive":
                            noisy_image_prompt = self.scheduler.add_noise(
                                image_prompts[image_index],
                                noise,
                                ref_timestep * (ref_image_num - image_index),
                            )
                            noisy_zero_image_prompt = self.scheduler.add_noise(
                                zero_image_prompts[image_index],
                                noise,
                                ref_timestep * (ref_image_num - image_index),
                            )
                        elif stage == "multi-image-condition":
                            noisy_image_prompt = self.scheduler.add_noise(
                                image_prompts[image_index],
                                noise,
                                ref_timestep,
                            )
                            noisy_zero_image_prompt = self.scheduler.add_noise(
                                zero_image_prompts[image_index],
                                noise,
                                ref_timestep,
                            )
                        else:
                            noisy_image_prompt = image_prompts[image_index]
                            noisy_zero_image_prompt = zero_image_prompts[image_index]

                        noisy_image_prompt = (
                            torch.cat([noisy_zero_image_prompt, noisy_image_prompt, noisy_image_prompt])
                            if do_classifier_free_guidance
                            else noisy_image_prompt
                        )

                        if do_classifier_free_guidance and len(prev_text_embeddings) > image_index:
                            p_text_embeddings = torch.cat(
                                [
                                    prev_text_embeddings[image_index],
                                    prev_text_embeddings[image_index][num_images_per_prompt:],
                                ]
                            )
                        else:
                            p_text_embeddings = text_embeddings

                        if stage == "multi-image-condition":
                            img_dif_condition = self.unet(
                                noisy_image_prompt,
                                ref_timestep,
                                encoder_hidden_states=p_text_embeddings,
                                return_dict=False,
                            )[1]
                        elif stage == "auto-regressive":
                            img_dif_condition = self.unet(
                                noisy_image_prompt,
                                ref_timestep * (ref_image_num - image_index),
                                encoder_hidden_states=p_text_embeddings,
                                return_dict=False,
                            )[1]
                        else:
                            img_dif_condition = None
                        img_conditions.append(img_dif_condition)

                    if stage in {"multi-image-condition", "auto-regressive"}:
                        img_dif_conditions = {}
                        for key in img_conditions[0].keys():
                            img_dif_conditions[key] = torch.cat(
                                [img_condition[key] for img_condition in img_conditions],
                                dim=1,
                            )
                    else:
                        img_dif_conditions = None
                else:
                    img_dif_conditions = None

                t_embeddings = (
                    torch.cat([text_embeddings[:num_images_per_prompt], text_embeddings])
                    if do_classifier_free_guidance
                    else text_embeddings
                )

                latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

                noise_pred = self.unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=t_embeddings,
                    image_hidden_states=img_dif_conditions,
                    return_dict=False,
                )[0].to(dtype=latents.dtype)

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_image, noise_pred_all = noise_pred.chunk(3)
                    noise_pred = (
                        noise_pred_uncond
                        + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                        + guidance_scale * (noise_pred_all - noise_pred_image)
                    )

                latents = self.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs).prev_sample

                if step_index == len(timesteps) - 1 or (
                    (step_index + 1) > num_warmup_steps and (step_index + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and step_index % callback_steps == 0:
                        callback(step_index, timestep, latents)

                torch.cuda.empty_cache()

        del zero_image_prompts, image_prompts, noise, t_embeddings, latent_model_input
        torch.cuda.empty_cache()

        image = self.decode_latents(latents)
        has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        del text_embeddings, prev_text_embeddings, latents, timesteps
        if "image_prompt" in locals():
            del image_prompt
        self._clean_up_cached_inputs()
        torch.cuda.empty_cache()

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    @staticmethod
    def numpy_to_pil(images):
        pil_images = []
        for sequence in images:
            pil_images.append(DiffusionPipeline.numpy_to_pil(sequence))
        return pil_images
