import os
import cv2
import random
from typing import Optional, Dict

from omegaconf import OmegaConf

import torch
import torch.utils.data
import torch.nn.functional as F
import torch.utils.checkpoint

from torch.cuda.amp import autocast

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel

from utils.util import get_time_string, get_function_args
from model.unet_2d_condition import UNet2DConditionModel
from model.pipeline import StableDiffusionPipeline
from dataset import StorySalonDataset
import sys
from inference_sc import test, test2
import numpy as np
from PIL import Image
from torchvision import transforms
import statistics
import clip
import time

logger = get_logger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class SampleLogger:
    def __init__(
            self,
            logdir: str,
            subdir: str = "sample",
            stage: str = 'multi-image-condition',
            num_samples_per_prompt: int = 1,
            num_inference_steps: int = 40,
            guidance_scale: float = 7.0,
            image_guidance_scale: float = 3.5,
    ) -> None:
        self.stage = stage  
        self.guidance_scale = guidance_scale
        self.image_guidance_scale = image_guidance_scale
        self.num_inference_steps = num_inference_steps
        self.num_sample_per_prompt = num_samples_per_prompt
        self.logdir = os.path.join(logdir, subdir)
        os.makedirs(self.logdir)

    def log_sample_images(
            self, batch, pipeline: StableDiffusionPipeline, device: torch.device, step: int
    ):
        sample_seeds = torch.randint(0, 100000, (self.num_sample_per_prompt,))
        sample_seeds = sorted(sample_seeds.numpy().tolist())
        self.sample_seeds = sample_seeds
        self.prompts = batch["prompt"]
        self.prev_prompts = batch["ref_prompt"]

        for idx, prompt in enumerate(tqdm(self.prompts, desc="Generating sample images")):
            image = batch["image"][idx, :, :, :].unsqueeze(0)
            ref_images = batch["ref_image"][idx, :, :, :, :].unsqueeze(0)
            image = image.to(device=device)
            ref_images = ref_images.to(device=device)
            generator = []
            for seed in self.sample_seeds:
                generator_temp = torch.Generator(device=device)
                generator_temp.manual_seed(seed)
                generator.append(generator_temp)
            sequence = pipeline(  
                stage=self.stage,
                prompt=prompt,
                image_prompt=ref_images,
                prev_prompt=self.prev_prompts,
                height=image.shape[2],
                width=image.shape[3],
                generator=generator,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                image_guidance_scale=self.image_guidance_scale,
                num_images_per_prompt=self.num_sample_per_prompt,
            ).images

            image = (image + 1.) / 2.  # for visualization
            image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            cv2.imwrite(os.path.join(self.logdir, f"{step}_{idx}_{seed}.png"), image[:, :, ::-1] * 255)
            v_refs = []
            ref_images = ref_images.squeeze(0)
            for ref_image in ref_images:
                # v_ref = (ref_image + 1.) / 2. # for visualization
                v_ref = ref_image.permute(1, 2, 0).detach().cpu().numpy()
                v_refs.append(v_ref)
            for i in range(len(v_refs)):
                cv2.imwrite(os.path.join(self.logdir, f"{step}_{idx}_{seed}_ref_{i}.png"), v_refs[i][:, :, ::-1] * 255)

            with open(os.path.join(self.logdir, f"{step}_{idx}_{seed}" + '.txt'), 'a') as f:
                f.write(batch['prompt'][idx])
                f.write('\n')
                f.write('\n')
                for prev_prompt in self.prev_prompts:
                    f.write(prev_prompt[0])
                    f.write('\n')
            for i, img in enumerate(sequence):
                img[0].save(os.path.join(self.logdir, f"{step}_{idx}_{sample_seeds[i]}_output.png"))


def custom_collate_fn(batch):
    images = []
    ref_images = []
    masks = []
    ref_masks = []
    ref_prompts = []
    prompts = []

    # 找到 batch 中最大参考图像数量
    max_refs = max(len(item['ref_image']) for item in batch)
    # 如果 max_refs 大于 8，限制为 8
    max_refs = min(max_refs, 8)

    # 找到 batch 中最大掩码数量
    max_masks = max(len(item['mask']) for item in batch)
    max_masks = min(max_masks, 8)

    for item in batch:
        # 主图像
        images.append(item['image'])

        # 参考图像
        ref_imgs = item['ref_image'][:max_refs]  # 只保留前 max_refs 张
        if len(ref_imgs) < max_refs:
            # 对参考图像进行零填充，确保填充值的维度与 ref_imgs 一致
            zeros = torch.zeros((max_refs - len(ref_imgs),) + ref_imgs.shape[1:], dtype=ref_imgs.dtype,
                                device=ref_imgs.device)
            ref_imgs = torch.cat([ref_imgs, zeros], dim=0)
        ref_images.append(ref_imgs)

        # 掩码
        item_masks = item['mask'][:max_masks]  # Truncate first
        if item_masks.shape[0] < max_masks:
            zeros = torch.zeros((max_masks - item_masks.shape[0],) + item_masks.shape[1:], dtype=item_masks.dtype, device=item_masks.device)
            item_masks = torch.cat([item_masks, zeros], dim=0)
        masks.append(item_masks)

        # 参考提示词和当前提示词
        ref_prompts.append(item['ref_prompt'])
        prompts.append(item['prompt'])

    # 将主图像、参考图像和掩码堆叠成张量
    images = torch.stack(images, dim=0)
    ref_images = torch.stack(ref_images, dim=0)
    masks = torch.stack(masks, dim=0)
    ref_image_ids = [item['ref_image_ids'] for item in batch]
    image_ids = [item['image_id'] for item in batch]

    # 返回包含所有处理后的数据
    return {
        'image': images,  # [batch_size, 3, 512, 512]
        'ref_image': ref_images,  # [batch_size, max_refs, 3, 512, 512]
        'mask': masks,  # [batch_size, max_masks, 3, 512, 512]
        'ref_prompt': ref_prompts,  # [batch_size]
        'prompt': prompts,  # [batch_size]
        'max_refs': max_refs,
        'max_masks': max_masks,
        'ref_image_ids':ref_image_ids,
        'image_ids':image_ids,
    }

def attention_loss(cross_attention, target_mask):
    """
    计算目标图的交叉注意力图与目标掩码之间的损失。
    """
    return F.mse_loss(cross_attention, target_mask)


class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            self.features = output[0]
        else:
            self.features = output
        
    def close(self):
        self.hook.remove()

def train(
        pretrained_model_path: str,
        logdir: str,
        train_steps: int = 300,
        validation_steps: int = 1000,
        validation_sample_logger: Optional[Dict] = None,
        gradient_accumulation_steps: int = 30,  # important hyper-parameter
        seed: Optional[int] = None,
        mixed_precision: Optional[str] = "fp16",
        train_batch_size: int = 4,
        val_batch_size: int = 1,
        learning_rate: float = 3e-5,
        scale_lr: bool = False,
        lr_scheduler: str = "constant",
        # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
        lr_warmup_steps: int = 0,
        use_8bit_adam: bool = True,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        max_grad_norm: float = 1.0,
        checkpointing_steps: int = 2000,
):
    args = get_function_args()
    time_string = get_time_string()
    logdir += f"_{time_string}"

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    if accelerator.is_main_process:
        os.makedirs(logdir, exist_ok=True)
        OmegaConf.save(args, os.path.join(logdir, "config.yml"))

    if seed is not None:
        set_seed(seed)

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device,download_root='./clip' )
    clip_model.eval()  

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")

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
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

        ############################################################################################
    # 同理ref_model也加载参数并全部设置grad=false
    ref_unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    ref_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    ref_noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    
    ref_pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=ref_unet,
        scheduler=ref_scheduler,
    )  
    
    ref_pipeline.set_progress_bar_config(disable=True)

    if is_xformers_available():
        try:
            ref_pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )
            
    ref_unet.requires_grad_(False)

    ############################################################################################################################################################################

    trainable_modules = ("attn3")
    for name, module in unet.named_modules():
        if name.endswith(trainable_modules):
            for params in module.parameters():
                params.requires_grad = True
        # for params in module.parameters():
        #     params.requires_grad = True

    if scale_lr:
        learning_rate = (
                learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = unet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    train_dataset = StorySalonDataset(root="/data/LLM_DATA/StorySalon/StorySalon", dataset_name='train')
    val_dataset = StorySalonDataset(root="/data/LLM_DATA/StorySalon/StorySalon", dataset_name='test')

    print(train_dataset.__len__())
    print(val_dataset.__len__())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                                                   num_workers=8, collate_fn=custom_collate_fn)  # 每次从数据集中加载一batch_size数据 num_workers是线程并行数
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=1)

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=train_steps * gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Add hook to the target layer
    target_layer_hook = FeatureHook(accelerator.unwrap_model(unet).up_blocks[3])

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    ref_unet.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("StoryGen")
    step = 0

    if validation_sample_logger is not None and accelerator.is_main_process:
        validation_sample_logger = SampleLogger(**validation_sample_logger, logdir=logdir)

    progress_bar = tqdm(range(step, train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            accelerator.wait_for_everyone()

    train_data_yielder = make_data_yielder(train_dataloader)
    val_data_yielder = make_data_yielder(val_dataloader)

    while step < train_steps:
        batch = next(train_data_yielder)  # 这里处理getitem
        ref_image_ids = batch['ref_image_ids']
        image_ids = batch['image_ids']

        vae.eval()
        text_encoder.eval()
        unet.train()

        image = batch["image"].to(dtype=weight_dtype)  # [b,3,512,512] 
        prompt = batch["prompt"]  # b 生成图的提示
        prompt_ids = tokenizer(prompt, truncation=True, padding="max_length", max_length=tokenizer.model_max_length,
                               return_tensors="pt").input_ids  # [b,77] 文本进行编码 padding都填充到最大长度保持一致 以张量返回
        b, c, h, w = image.shape  # [b,3,512,512]

        latents = vae.encode(image).latent_dist.sample()  # [b,4,64,64] 过vae
        latents = latents * 0.18215

        prev_prompts = batch["ref_prompt"]  #
        prev_prompt_ids = []
        for prev_prompt in prev_prompts:
            prev_prompt_ids.append(
                tokenizer(prev_prompt, truncation=True, padding="max_length", max_length=tokenizer.model_max_length,
                          return_tensors="pt").input_ids.squeeze(0))
        t_prev_prompt_ids = torch.stack(prev_prompt_ids)  # (b, 77) 沿着新维度3进行拼接
        ref_images = batch["ref_image"].to(dtype=weight_dtype)  # (b, max_ref, 3, 512, 512)
        t_ref_images = torch.transpose(ref_images, 0, 1)  # (max_ref, b, 3, 512, 512) 将第 0 维和第 1 维进行互换。

        ref_image_list = []  # [max_ref x (b, 4, 64, 64)]
        for t_ref_image in t_ref_images:
            new_ref_image = vae.encode(t_ref_image).latent_dist.sample()  # 过vae
            new_ref_image = new_ref_image * 0.18215
            ref_image_list.append(new_ref_image)

        # Now compute the attention loss between ref_images and ref_masks
        ref_mask_list = batch["mask"].to(dtype=weight_dtype)  # [b, max_refs, 3, 512, 512]
        ref_mask_list = ref_mask_list.permute(1, 0, 2, 3, 4) #ref_mask_list[max_refs,b,3,512,512]
        ref_mask_latent_list = []
        for t_ref_mask in ref_mask_list:  # t_ref_mask: [b, 1, 512, 512]
            
            new_ref_mask = vae.encode(t_ref_mask).latent_dist.sample()  # VAE编码并采样
            new_ref_mask = new_ref_mask * 0.18215  # 和图像保持一致的缩放
            ref_mask_latent_list.append(new_ref_mask)  # [b, 4, 64, 64]
        # Sample noise that we'll add
        noise = torch.randn_like(latents)  # [-1, 1] [b,4,64,64] 生成与 latents 形状相同的随机噪声张量
        ref_noise = torch.randn_like(latents)  # [b,4,64,64] use a different noise here 参考图像噪声
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,),
                                  device=latents.device)  # 每个b都有一个随机时间步 扩散过程中对某个时间点对噪声进行扰动
        ref_timesteps = timesteps / 10 
        timesteps = timesteps.long()
        ref_timesteps = ref_timesteps.long()

        # Add noise according to the noise magnitude at each timestep (this is the forward diffusion process)
        noisy_latent = noise_scheduler.add_noise(latents, noise, timesteps)  # 通过将 noise 添加到 latents

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(prompt_ids.to(accelerator.device))[0]  # B * 77 * 768 文本encoder

        # Compute image diffusion features
        ref_img_features = []
        ref_mask_features = []
        p = random.uniform(0, 1)  # 正态分布

        max_refs = batch['max_refs']
        max_masks = batch['max_masks']
        t_prev_prompt_ids = t_prev_prompt_ids.repeat(max_refs, 1, 1)
        # if (max_refs==1):
        for i in range(max_refs):
            noisy_ref_image = noise_scheduler.add_noise(ref_image_list[i], ref_noise, ref_timesteps)* (max_refs - i)
            prev_encoder_hidden_states = text_encoder(t_prev_prompt_ids[i].to(accelerator.device))[0]
            img_dif_conditions = unet(noisy_ref_image, ref_timesteps * (max_refs - i), 
                                encoder_hidden_states=prev_encoder_hidden_states,return_dict=False)[1] # 去噪处理
            ref_img_features.append(img_dif_conditions)
        for i in range(max_masks):
            # Clear hook features before mask forward pass
            target_layer_hook.features = None
            
            noisy_ref_mask = noise_scheduler.add_noise(ref_mask_latent_list[i], ref_noise, ref_timesteps) * (max_masks - i)
            _ = unet(noisy_ref_mask, ref_timesteps * (max_masks - i),
                               encoder_hidden_states=encoder_hidden_states, return_dict=False)
            
            mask_feature = target_layer_hook.features
            if mask_feature is not None:
                ref_mask_features.append(mask_feature)

        img_dif_conditions = {}
        #检查
        if len(ref_img_features) == 0 or not isinstance(ref_img_features[0], dict):
            print(f"Warning: ref_img_features at step {step} is empty or invalid")
            continue  # 跳过这个 batch，或者采取其他处理方法

        for k, v in ref_img_features[0].items():
            img_dif_conditions[k] = torch.cat([ref_img_feature[k] for ref_img_feature in ref_img_features],dim=1)  # 每一个参考图像帧的特征字典中的每个键 k沿着第二维拼接
        
        def compute_cross_attention(target_img_features, mask_features):
            """
            计算 target_img_features 和 mask_features 之间的交叉注意力图。
            """
            dtype = target_img_features.dtype  
            mask_features = mask_features.to(dtype=dtype)  
            

            assert target_img_features.shape == mask_features.shape, "target_img_features 和 mask_features 的形状必须相同"
            b, seq_len, d = target_img_features.shape

            attention_scores = torch.bmm(target_img_features, mask_features.transpose(1, 2))  # [b, seq_len, seq_len]

            attention_weights = F.softmax(attention_scores/ (d ** 0.5), dim=-1)  # [b, seq_len, seq_len]

            cross_attention_map = torch.bmm(attention_weights, mask_features)  # [b, seq_len, d]

            return cross_attention_map
        
        # Clear hook features before the main forward pass
        target_layer_hook.features = None
        
        # Predict the noise residual
        model_pred = unet(noisy_latent, timesteps, encoder_hidden_states=encoder_hidden_states,
                          image_hidden_states=img_dif_conditions, return_dict=False)[0]

        # Get the feature map from the hook
        target_feature_map = target_layer_hook.features

        attention_loss = 0
        total_attention_loss = 0
        num_masks = len(ref_mask_features)
        # We need to process the feature maps to match dimensions for attention
        if target_feature_map is not None and ref_mask_features:
            b, c, h, w = target_feature_map.shape
            target_feature_map_reshaped = target_feature_map.view(b, c, h * w).permute(0, 2, 1) # [b, h*w, c]

            for mask_feature_map in ref_mask_features:
                if mask_feature_map.shape[1:] != target_feature_map.shape[1:]:
                    mask_feature_map = F.interpolate(mask_feature_map, size=target_feature_map.shape[2:], mode='bilinear', align_corners=False)

                mask_feature_map_reshaped = mask_feature_map.view(b, c, h * w).permute(0, 2, 1)
                
                # L2 normalize
                norm_target = F.normalize(target_feature_map_reshaped, dim=-1)
                norm_mask   = F.normalize(mask_feature_map_reshaped, dim=-1)

                # Cross attention
                cross_attention_map = compute_cross_attention(norm_target, norm_mask)
                # Cosine loss
                total_attention_loss += 1 - F.cosine_similarity(cross_attention_map, norm_mask, dim=-1).mean()
    
        attention_loss = total_attention_loss / num_masks
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        loss += attention_loss  # Add the attention loss to the main loss

        ###############################################################
        #RL
        num_inference_steps = 40
        guidance_scale = 7
        image_guidance_scale = 3.5
        num_sample_per_prompt = 4   # RL生成的图的数目
        mixed_precision = "fp16"
        stage = 'multi-image-condition'
        beta_dpo = 5000 # 设置参数
        
        # RL生成的四张图
        with torch.no_grad():
            RL_image_ids = []
            for i in range(len(ref_image_ids)):
                prev_prompt_tmp = [prompt[i] for q in range(len(ref_image_ids[i]))]     # prev_prompts就是Prompts
                RL_images = test2(tokenizer,
                                text_encoder,
                                vae,
                                ref_unet,
                                scheduler,
                                ref_pipeline,
                                accelerator.device,
                                "./inference_RL/",
                                i,
                                prompt[i],
                                prev_prompt_tmp,
                                ref_image_ids[i],
                                num_inference_steps,
                                guidance_scale,
                                image_guidance_scale,
                                num_sample_per_prompt,
                                stage, 
                                mixed_precision)
                RL_image_ids.append(["./inference_RL/" + f"{i}/" + f"output_{j}.png" for j in range(num_sample_per_prompt)])
        # 这里获得clip评估
        def compute_clip_reward(image_path, text, ref_image_path=None, alpha=0.6, beta=0.4):
            """
            计算强化学习奖励：
            r(x) = α * CLIP-T(image, text) + β * CLIP-I(image, ref_image)
            
            :param image_path: 当前图像路径
            :param text: 文字描述
            :param ref_image_path: 参考图像路径
            :param alpha: CLIP-T 权重
            :param beta: CLIP-I 权重
            :return: 最终奖励值
            """
            # 处理图像
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            
            text_tokens = clip.tokenize([text], truncate = True).to(device)

            # 计算 CLIP-T (图像-文本相似度)
            with torch.no_grad():
                image_features = clip_model.encode_image(image)
                text_features = clip_model.encode_text(text_tokens)
                clip_t = torch.cosine_similarity(image_features, text_features).item()

            # 计算 CLIP-I (图像-图像相似度)
            if ref_image_path:
                ref_image = preprocess(Image.open(ref_image_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    ref_image_features = clip_model.encode_image(ref_image)
                    clip_i = torch.cosine_similarity(image_features, ref_image_features).item()
            else:
                clip_i = 0  # 没有前帧时，只计算 CLIP-T

            # 计算最终奖励
            reward = alpha * clip_t + beta * clip_i
            return reward, clip_t, clip_i
        
        reward_RL = []
        for index, img_ids in enumerate(RL_image_ids):
            reward_tmp = []
            text = batch["prompt"][index]
            ref_image_path = image_ids[index]
            for image_path in img_ids:
                with torch.no_grad():
                    reward, clip_t, clip_i = compute_clip_reward(image_path, text, ref_image_path)
                reward_tmp.append(reward)
            reward_RL.append(reward_tmp)

        # 处理RL_images，参照dataset的处理
        # inside_term_RL就是主要的loss了
        inside_term_RL = torch.zeros(len(ref_image_ids))
        for index, i in enumerate(RL_image_ids):
            image_RL = []
            for j in i:     # 直接在这里按照每一组来处理了
                image = np.ascontiguousarray(Image.open(j).convert('RGB'))
                image = Image.fromarray(image).convert('RGB')
                image = image.resize((512, 512))
                image = transforms.ToTensor()(image)
                image = torch.from_numpy(np.ascontiguousarray(image)).float()
                image = image * 2. - 1.
                image_RL.append(image.to(dtype=weight_dtype))
            image_RL = torch.stack(image_RL, dim=0).to(accelerator.device, dtype=weight_dtype)
            prompt_RL = [batch["prompt"][index] for k in range(num_sample_per_prompt)]
            with torch.no_grad():
                prompt_ids_RL = tokenizer(prompt_RL, truncation=True, padding="max_length", max_length=tokenizer.model_max_length,
                                return_tensors="pt").input_ids  # [b,77] 文本进行编码 padding都填充到最大长度保持一致 以张量返回    

            b, c, h, w = image_RL.shape

            with torch.no_grad():
                latents_RL = vae.encode(image_RL).latent_dist.sample()
                latents_RL = latents_RL * 0.18215
            
            prev_prompt_RL = prev_prompts[index]
            with torch.no_grad():
                tmp_prompt_ids = tokenizer(prev_prompt, truncation=True, padding="max_length", max_length=tokenizer.model_max_length,
                            return_tensors="pt").input_ids.squeeze(0)
            prev_prompt_ids_RL = [tmp_prompt_ids for k in range(num_sample_per_prompt)]
            t_prev_prompt_ids_RL = torch.stack(prev_prompt_ids_RL)
            

            # 使用torch来重复image
            ref_images_RL = ref_images[index].unsqueeze(0).expand(num_sample_per_prompt, -1, -1, -1, -1) 
            t_ref_images_RL = torch.transpose(ref_images_RL, 0, 1)  # (max_ref, b, 3, 512, 512) 将第 0 维和第 1 维进行互换。
            
            ref_image_list_RL = []  # [max_ref x (b, 4, 64, 64)]
            with torch.no_grad():
                for t_ref_image in t_ref_images_RL:
                    new_ref_image = vae.encode(t_ref_image).latent_dist.sample()  # 过vae
                    new_ref_image = new_ref_image * 0.18215
                    ref_image_list_RL.append(new_ref_image)
                
            
            # Sample noise that we'll add
            noise_RL = torch.randn_like(latents_RL)  # [-1, 1] [b,4,64,64] 生成与 latents 形状相同的随机噪声张量
            ref_noise_RL = torch.randn_like(latents_RL)  # [b,4,64,64] use a different noise here 参考图像噪声
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,),
                                    device=latents_RL.device)  # 每个b都有一个随机时间步 扩散过程中对某个时间点对噪声进行扰动
            ref_timesteps = timesteps / 10  
            timesteps = timesteps.long()
            ref_timesteps = ref_timesteps.long()

            # Add noise according to the noise magnitude at each timestep (this is the forward diffusion process)，这个生成器看下后面ref的模型是否也需要
            noisy_latent_RL = noise_scheduler.add_noise(latents_RL, noise_RL, timesteps)  # 通过将 noise 添加到 latents

            # Get the text embedding for conditioning
            encoder_hidden_states_RL = text_encoder(prompt_ids_RL.to(accelerator.device))[0]  # B * 77 * 768 文本encoder

            # Compute image diffusion features
            ref_img_features_RL = []
            p = random.uniform(0, 1)  # 正态分布

            # max_refs = batch['max_refs']
            t_prev_prompt_ids_RL = t_prev_prompt_ids_RL.repeat(max_refs, 1, 1)
            
            ###############################################################################训练模型的预测
            # if (max_refs==1):
            for i in range(max_refs):
                noisy_ref_image_RL = noise_scheduler.add_noise(ref_image_list_RL[i], ref_noise_RL, ref_timesteps)* (max_refs - i)
                prev_encoder_hidden_states_RL = text_encoder(t_prev_prompt_ids_RL[i].to(accelerator.device))[0]
                img_dif_conditions_RL = unet(noisy_ref_image_RL, ref_timesteps * (max_refs - i), encoder_hidden_states=prev_encoder_hidden_states_RL,return_dict=False)[1] # 去噪处理
                ref_img_features_RL.append(img_dif_conditions_RL)
                
            img_dif_conditions_RL = {} #提取的参考图像特征
            #检查
            if len(ref_img_features_RL) == 0 or not isinstance(ref_img_features_RL[0], dict):
                print(f"Warning: ref_img_features_RL at step {step} is empty or invalid")
                continue  # 跳过这个 batch，或者采取其他处理方法

            for k, v in ref_img_features_RL[0].items():
                img_dif_conditions_RL[k] = torch.cat([ref_img_feature_RL[k] for ref_img_feature_RL in ref_img_features_RL],dim=1)  # 每一个参考图像帧的特征字典中的每个键 k沿着第二维拼接
            
            # Predict the noise residual
            model_pred_RL = unet(noisy_latent_RL, timesteps, encoder_hidden_states=encoder_hidden_states_RL,
                            image_hidden_states=img_dif_conditions_RL, return_dict=False)[0]
            
            # 计算losses，用GRPO加diffusion-dpo的思想
            model_losses = (model_pred_RL - noise_RL).pow(2).mean(dim=[1,2,3])
            model_losses = list(model_losses.chunk(num_sample_per_prompt))
           
            ############################################################################### ref模型的预测
            with torch.no_grad():
                ref_img_features_RL_ref = []
                for i in range(max_refs):
                    noisy_ref_image_RL = noise_scheduler.add_noise(ref_image_list_RL[i], ref_noise_RL, ref_timesteps)* (max_refs - i)
                    prev_encoder_hidden_states_RL = text_encoder(t_prev_prompt_ids_RL[i].to(accelerator.device))[0]
                    img_dif_conditions_RL_ref = ref_unet(noisy_ref_image_RL, ref_timesteps * (max_refs - i), encoder_hidden_states=prev_encoder_hidden_states_RL,return_dict=False)[1] # 去噪处理
                    ref_img_features_RL_ref.append(img_dif_conditions_RL_ref)
                
                img_dif_conditions_ref = {} #提取的参考图像特征
                #检查
                if len(ref_img_features_RL_ref) == 0 or not isinstance(ref_img_features_RL_ref[0], dict):
                    print(f"Warning: ref_img_features_RL_ref at step {step} is empty or invalid")
                    continue  # 跳过这个 batch，或者采取其他处理方法

                for k, v in ref_img_features_RL_ref[0].items():
                    img_dif_conditions_RL_ref[k] = torch.cat([ref_img_feature_RL_ref[k] for ref_img_feature_RL_ref in ref_img_features_RL_ref],dim=1)  # 每一个参考图像帧的特征字典中的每个键 k沿着第二维拼接
                
                # Predict the noise residual
                model_pred_RL_ref = ref_unet(noisy_latent_RL, timesteps, encoder_hidden_states=encoder_hidden_states_RL,
                                image_hidden_states=img_dif_conditions_RL_ref, return_dict=False)[0]
                # 计算losses
                model_losses_ref = (model_pred_RL_ref - noise_RL).pow(2).mean(dim=[1,2,3])
                model_losses_ref = list(model_losses_ref.chunk(num_sample_per_prompt))
            
            #用GRPO加diffusion_dpo的思想算loss
            reward_RL_std = statistics.stdev(reward_RL[index])
            reward_RL_mean = sum(reward_RL[index]) / len(reward_RL[index])
            inside_term_tmp = 0
            for ind in range(num_sample_per_prompt):
                inside_term_tmp += ((reward_RL[index][ind] - reward_RL_mean) / reward_RL_std) * (model_losses[ind] - model_losses_ref[ind])
            
            scale_term = -1 / num_sample_per_prompt * beta_dpo
            inside_term_tmp = scale_term * inside_term_tmp
            inside_term_RL[index] = inside_term_tmp
        loss_RL =  -0.01 * F.logsigmoid(inside_term_RL).mean()
        loss += loss_RL

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            step += 1
            if accelerator.is_main_process:
                if validation_sample_logger is not None and step % validation_steps == 0:
                    unet.eval()
                    val_batch = next(val_data_yielder)
                    with autocast():
                        validation_sample_logger.log_sample_images(
                            batch=val_batch,
                            pipeline=pipeline,
                            device=accelerator.device,
                            step=step,
                        )
                if step % checkpointing_steps == 0:
                    pipeline_save = StableDiffusionPipeline(
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        unet=accelerator.unwrap_model(unet),
                        scheduler=scheduler,
                    )
                    checkpoint_save_path = os.path.join(logdir, f"checkpoint_{step}")
                    pipeline_save.save_pretrained(checkpoint_save_path)

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=step)
    
    # Clean up the hook
    target_layer_hook.close()
    
    accelerator.end_training()



if __name__ == "__main__":
    config = "./config/stage2_config.yml"
    train(**OmegaConf.load(config))

# CUDA_VISIBLE_DEVICES=0 accelerate launch train_SC_stage2.py