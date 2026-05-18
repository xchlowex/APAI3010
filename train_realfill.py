import random
import argparse
import copy
import itertools
import logging
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms_v2
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from torchvision import transforms
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from peft import PeftModel, LoraConfig, get_peft_model
from models import DINOv2GeometricAdapter

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.1")

logger = get_logger(__name__)

def frequency_weighted_loss(pred_noise, target_noise, alpha_hf=2.0):
    
    """
    Calculates MSE loss with a higher weight on high-frequency components.
    Works on 4D Latent Tensors [B, 4, 64, 64].
    """

    # dtype = pred_noise.dtype
    pred_noise = pred_noise.float()
    target_noise = target_noise.float()
    
    # 1. Use a 3x3 kernel for Latent Space (64x64)
    # Ensure kernel is odd and sigma is appropriate
    kernel_size = [3, 3]
    sigma = [0.5, 0.5] 

    # 2. Extract Low Frequency (Blurred version)
    # torchvision.transforms.functional.gaussian_blur handles [B, C, H, W]
    low_freq_target = gaussian_blur(target_noise, kernel_size=kernel_size, sigma=sigma)
    low_freq_pred = gaussian_blur(pred_noise, kernel_size=kernel_size, sigma=sigma)
    
    # 3. Extract High Frequency (Residual)
    high_freq_target = target_noise - low_freq_target
    high_freq_pred = pred_noise - low_freq_pred
    
    # 4. Calculate weighted loss
    loss_lf = F.mse_loss(low_freq_pred, low_freq_target)
    loss_hf = F.mse_loss(high_freq_pred, high_freq_target)
    
    return loss_lf + (alpha_hf * loss_hf)

    

def make_mask(images, resolution, times=30):
    mask, times = torch.ones_like(images[0:1, :, :]), np.random.randint(1, times)
    min_size, max_size, margin = np.array([0.03, 0.25, 0.01]) * resolution
    max_size = min(max_size, resolution - margin * 2)

    for _ in range(times):
        width = np.random.randint(int(min_size), int(max_size))
        height = np.random.randint(int(min_size), int(max_size))

        x_start = np.random.randint(int(margin), resolution - int(margin) - width + 1)
        y_start = np.random.randint(int(margin), resolution - int(margin) - height + 1)
        mask[:, y_start:y_start + height, x_start:x_start + width] = 0

    mask = 1 - mask if random.random() < 0.5 else mask
    return mask

def save_model_card(
    repo_id: str,
    images=None,
    base_model=str,
    repo_folder=None,
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
prompt: "a photo of sks"
tags:
- stable-diffusion-inpainting
- stable-diffusion-inpainting-diffusers
- text-to-image
- diffusers
- realfill
inference: true
---
    """
    model_card = f"""
# RealFill - {repo_id}

This is a realfill model derived from {base_model}. The weights were trained using [RealFill](https://realfill.github.io/).
You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


@torch.no_grad()
def log_validation(
    vae,
    text_encoder,
    tokenizer,
    unet,
    args,
    accelerator,
    weight_dtype,
    epoch,   # Added epoch so the tracker knows which step this is
    adapter, # Added adapter so we can use the geometric branch
):
    logger.info(f"Running validation... Generating {args.num_validation_images} images")


    # 1. Initialize standard pipeline
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae, keep_fp32_wrapper=True),
        tokenizer=tokenizer,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    
    # 2. Unwrap and Inject GeoFill Components

    pipeline.unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
    pipeline.text_encoder = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True)
    pipeline.to(accelerator.device)

    pipeline.vae.to(accelerator.device, dtype=weight_dtype)
    pipeline.unet.to(accelerator.device, dtype=weight_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    orig_vae_encode = pipeline.vae.encode
    orig_vae_decode = pipeline.vae.decode


    def vae_encode_dtype_safe(x, *args, **kwargs):
        vae_param = next(pipeline.vae.parameters())
        x = x.to(device=vae_param.device, dtype=vae_param.dtype)
        return orig_vae_encode(x, *args, **kwargs)
    
    def vae_decode_dtype_safe(z, *args, **kwargs):
        vae_param = next(pipeline.vae.parameters())
        # print("VAE decode input before:", z.dtype, z.device)
        z = z.to(device=vae_param.device, dtype=vae_param.dtype)
        # print("VAE decode input after:", z.dtype, z.device)

        return orig_vae_decode(z, *args, **kwargs)


    pipeline.vae.encode = vae_encode_dtype_safe
    pipeline.vae.decode = vae_decode_dtype_safe


    # Ensure Adapter in eval mode on correct device
    unwrapped_adapter = accelerator.unwrap_model(adapter).eval()

    # 3. Load target data for inpainting
    target_dir = Path(args.train_data_dir) / "target"
    image = Image.open(target_dir / "target.png").convert("RGB")
    mask = Image.open(target_dir / "mask.png").convert("L")


    # This matches the 'ref_transforms' in your RealFillDataset class

    ref_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load the first image from your reference folder
    ref_dir = Path(args.train_data_dir) / "ref"
    ref_image_path = sorted(list(ref_dir.glob("*.png")) + list(ref_dir.glob("*.jpg")))[0]
    ref_pil = Image.open(ref_image_path).convert("RGB")
    ref_tensor = ref_transform(ref_pil).unsqueeze(0).to(accelerator.device)
    

    # Initialize processor
    mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
    image_processor = VaeImageProcessor(vae_scale_factor=8)

    # Process images into tensors on the correct device and dtype
    # init_image = image_processor.preprocess(image).to(accelerator.device, dtype=weight_dtype)
    # mask = mask_processor.preprocess(mask_image).to(accelerator.device, dtype=weight_dtype)
    
    init_image = image.resize((args.resolution, args.resolution))
    mask = mask.resize((args.resolution, args.resolution))


    # 4. PREPARE THE 333-TOKEN CONTEXT
    # Project via Adapter (This runs the internal DINO + Projection layers)
    projected_geometry = unwrapped_adapter(ref_tensor).to(
        device=accelerator.device,
        dtype=weight_dtype
    ) # [1, 256, 1024]

    # C. Get Text Embeddings
    prompt = "a photo of sks"
    text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    prompt_embeds = pipeline.text_encoder(text_inputs.input_ids.to(accelerator.device))[0] # [1, 77, 1024]

    # D. Concatenate into 333 tokens
    fused_embeds = torch.cat([prompt_embeds, projected_geometry], dim=1).to(device=accelerator.device, dtype=weight_dtype)

# 1. Create matching Negative Prompt Embeds (333 tokens)
    # A. Get the standard 77-token null embedding
    uncond_tokens = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(accelerator.device)
    uncond_embeds = pipeline.text_encoder(uncond_tokens)[0].to(dtype=weight_dtype) # [1, 77, 1024]

    # B. Create zero-padding for the geometric part (256 tokens)
    # This ensures the negative prompt has no "geometric" influence
    neg_padding = torch.zeros((1, 256, 1024), device=accelerator.device, dtype=weight_dtype)

    # C. Concatenate to reach 333
    negative_fused_embeds = torch.cat([uncond_embeds, neg_padding], dim=1) # [1, 333, 1024]

    # 5. Run Inference
    images = []
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

# Debugging
    # print("==== VALIDATION DEBUG ====")
    # print("vae dtype:", next(pipeline.vae.parameters()).dtype)
    # print("unet dtype:", next(pipeline.unet.parameters()).dtype)
    # print("text encoder dtype:", next(pipeline.text_encoder.parameters()).dtype)
    # print("fused_embeds dtype:", fused_embeds.dtype)
    # print("negative_fused_embeds dtype:", negative_fused_embeds.dtype)
    # print("image type:", type(init_image))
    # print("mask type:", type(mask))
    # print("fused shape:", fused_embeds.shape)
    # print("negative shape:", negative_fused_embeds.shape)
    # print("==========================")

    for _ in range(args.num_validation_images):
        output = pipeline(
            prompt_embeds=fused_embeds, 
            negative_prompt_embeds=negative_fused_embeds, # Add this line!
            image=init_image,
            mask_image=mask,
            num_inference_steps=50, 
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        images.append(output)

    # # 5. Run Inference with fused_embeds
    # images = []
    # generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    # for _ in range(args.num_validation_images):
    #     # We use prompt_embeds instead of prompt string to bypass internal CLIP encoding
    #     output = pipeline(
    #         prompt_embeds=fused_embeds, 
    #         image=image, 
    #         mask_image=mask_image,
    #         num_inference_steps=50, # Reduced for speed in validation
    #         guidance_scale=args.guidance_scale if hasattr(args, 'guidance_scale') else 7.5,
    #         generator=generator
    #     ).images[0]
    #     images.append(output)

    # 6. Logging to Trackers
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log({"validation": [wandb.Image(img) for img in images]})

    # Restore the original VAE encode/decode methods so training VAE is not left wrapped.
    pipeline.vae.encode = orig_vae_encode
    pipeline.vae.decode = orig_vae_decode

    del pipeline
    torch.cuda.empty_cache()
    return images

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of images.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_conditioning`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run realfill validation every X steps. RealFill validation consists of running the conditioning"
            " `args.validation_conditioning` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="realfill-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--unet_learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate to use for unet.",
    )
    parser.add_argument(
        "--text_encoder_learning_rate",
        type=float,
        default=4e-5,
        help="Learning rate to use for text encoder.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--wandb_key",
        type=str,
        default=None,
        help=("If report to option is set to wandb, api-key for wandb used for login to wandb "),
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default=None,
        help=("If report to option is set to wandb, project name in wandb for log tracking  "),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=27,
        help=("The alpha constant of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="The dropout rate of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        help="The bias type of the Lora update matrices. Must be 'none', 'all' or 'lora_only'.",
    )
    parser.add_argument(
        "--freq_loss_alpha",
        type=float,
        default=2.0,
        help="Alpha weight for high-frequency loss component in frequency-weighted loss.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

class RealFillDataset(Dataset):
    """
    A dataset to prepare the training and conditioning images and
    the masks with the dummy prompt for fine-tuning the model.
    It pre-processes the images, masks and tokenizes the prompts.
    """

    def __init__(
        self,
        train_data_root,
        tokenizer,
        size=512,
    ):
        self.size = size
        self.tokenizer = tokenizer

        self.ref_data_root = Path(train_data_root) / "ref"
        self.target_image = Path(train_data_root) / "target" / "target.png"
        self.target_mask = Path(train_data_root) / "target" / "mask.png"
        if not (self.ref_data_root.exists() and self.target_image.exists() and self.target_mask.exists()):
            raise ValueError("Train images root doesn't exist.")

        self.train_images_dir = os.path.join(train_data_root, "ref")
        valid_extensions = (".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG")
        self.train_images_path = [
            os.path.join(self.train_images_dir, f) 
            for f in sorted(os.listdir(self.train_images_dir)) 
            if f.lower().endswith(valid_extensions)
        ]
        self.target_image_path = os.path.join(train_data_root, "target", "target.png") # Adjust filename if needed
        self.train_images_path.append(self.target_image_path)
        
        self.num_train_images = len(self.train_images_path)
        self.train_prompt = "a photo of sks"

        self.transform = transforms_v2.Compose(
            [
                transforms_v2.RandomResize(size, int(1.125 * size)),
                transforms_v2.RandomCrop(size),
                # transforms_v2.ToImageTensor(),
                transforms_v2.ToImage(),
                transforms_v2.ConvertImageDtype(),
                transforms_v2.Normalize([0.5], [0.5]),
            ]
        )
        # Add a transform specifically for DINOv2 (needs 224x224)
        self.dinov2_transform = transforms_v2.Compose(
            [
                transforms_v2.Resize((224, 224), interpolation=transforms_v2.InterpolationMode.BICUBIC),
                # transforms_v2.ToImageTensor(),
                transforms_v2.ToImage(),
                transforms_v2.ToDtype(torch.float32, scale=True), # Use ToDtype instead of ConvertImageDtype                transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ADD THIS
           ])


    def __len__(self):
        return self.num_train_images

    def __getitem__(self, index):
        example = {}

        image = Image.open(self.train_images_path[index])
        image = exif_transpose(image)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        if index < len(self) - 1:
            weighting = Image.new("L", image.size)
        else:
            weighting = Image.open(self.target_mask)
            weighting = exif_transpose(weighting)

        image, weighting = self.transform(image, weighting) # The range of weighting becomes [-1, 1] after self.transform
        example["images"], example["weightings"] = image, weighting[0:1] < 0

        if index == len(self) - 1:
            example["masks"] = 1 - (example["weightings"]).float()
        elif random.random() < 0.1:
            example["masks"] = torch.ones_like(example["images"][0:1])
        else:
            example["masks"] = make_mask(example["images"], self.size)

        example["conditioning_images"] = example["images"] * (example["masks"] < 0.5)

        train_prompt = "" if random.random() < 0.1 else self.train_prompt
        example["prompt_ids"] = self.tokenizer(
            train_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

    # 6. FIXED: Select and transform the DINO reference image
        # This must happen AFTER ref_img is loaded from disk
        num_refs = len(self.train_images_path) - 1
        ref_idx = random.randint(0, num_refs - 1)
        ref_img_pil = Image.open(self.train_images_path[ref_idx]).convert("RGB")
        example["ref_pixel_values"] = self.dinov2_transform(ref_img_pil)
        
        return example

def collate_fn(examples):
    input_ids = [example["prompt_ids"] for example in examples]
    images = [example["images"] for example in examples]
    masks = [example["masks"] for example in examples]
    weightings = [example["weightings"] for example in examples]
    conditioning_images = [example["conditioning_images"] for example in examples]
    
    # ADDED THIS LINE:
    ref_pixel_values = [example["ref_pixel_values"] for example in examples]

    images = torch.stack(images).to(memory_format=torch.contiguous_format).float()
    masks = torch.stack(masks).to(memory_format=torch.contiguous_format).float()
    weightings = torch.stack(weightings).to(memory_format=torch.contiguous_format).float()
    conditioning_images = torch.stack(conditioning_images).to(memory_format=torch.contiguous_format).float()
    
    # ADDED THIS LINE:
    ref_pixel_values = torch.stack(ref_pixel_values).to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "images": images,
        "masks": masks,
        "weightings": weightings,
        "conditioning_images": conditioning_images,
        "ref_pixel_values": ref_pixel_values, # ADDED THIS LINE
    }
    return batch

def main(args):
    
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32

    # Load DINOv2 ONCE here
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2_model.to(accelerator.device) 
    dinov2_model.eval()

    # KEEP this in float32 as well.
    # Note: adapter is created after unet is loaded (below) since it needs unet.config.cross_attention_dim
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

        wandb.login(key=args.wandb_key)
        wandb.init(project=args.wandb_project_name)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # Create the adapter and pass the model into it
    # Use unet's cross_attention_dim (typically 1024 for SD 2.1)
    adapter = DINOv2GeometricAdapter(dinov2_model=dinov2_model, sd_dim=unet.config.cross_attention_dim)
    adapter.to(accelerator.device, dtype=torch.float32)


    unet_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
    )
    unet = get_peft_model(unet, unet_config)

    text_encoder_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
    )
    text_encoder = get_peft_model(text_encoder, text_encoder_config)
    
    vae.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                model_to_save = accelerator.unwrap_model(model)
                
                if hasattr(model_to_save, "base_model"):
                    # Handle LoRA models
                    is_unet = isinstance(model_to_save.base_model.model, type(accelerator.unwrap_model(unet).base_model.model))
                    sub_dir = "unet" if is_unet else "text_encoder"
                    model_to_save.save_pretrained(os.path.join(output_dir, sub_dir))
                elif hasattr(model_to_save, "proj"):
                    # Handle your custom DINO adapter
                    torch.save(model_to_save.state_dict(), os.path.join(output_dir, "adapter.bin"))
    
                if weights:
                    weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            model = models.pop()
            model_to_load = accelerator.unwrap_model(model)

            # 1. Handle PEFT models (UNet, Text Encoder)
            if hasattr(model_to_load, "base_model"):
                # Check if it's UNet or Text Encoder to get the right subfolder
                is_unet = isinstance(model_to_load.base_model.model, type(accelerator.unwrap_model(unet).base_model.model))
                sub_dir = "unet" if is_unet else "text_encoder"
                
                # Use PEFT's specific loading logic
                model.load_adapter(os.path.join(input_dir, sub_dir), "default")
                
            # 2. Handle your Custom Adapter (DINOv2GeometricAdapter)
            elif hasattr(model_to_load, "proj"):
                adapter_path = os.path.join(input_dir, "adapter.bin")
                if os.path.exists(adapter_path):
                    # Map to CPU first to avoid OOM, then move to device
                    state_dict = torch.load(adapter_path, map_location="cpu")
                    model_to_load.load_state_dict(state_dict)
                    logger.info(f"Successfully loaded custom adapter weights from {adapter_path}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.unet_learning_rate = (
            args.unet_learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

        args.text_encoder_learning_rate = (
            args.text_encoder_learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
        
    else:
        optimizer_class = torch.optim.AdamW

    # Move VAE to device with weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)

    # Create one unified optimizer
    optimizer = optimizer_class(
        [
            {"params": unet.parameters(), "lr": args.unet_learning_rate},
            {"params": text_encoder.parameters(), "lr": args.text_encoder_learning_rate},
            {"params": adapter.proj.parameters(), "lr": args.unet_learning_rate}
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Create dataset and dataloader
    train_dataset = RealFillDataset(
        train_data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
    )

    # Calculate training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare all components with accelerator
    unet, text_encoder, adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, adapter, optimizer, train_dataloader, lr_scheduler
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers("realfill", config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        text_encoder.train()
        adapter.train()

        logger.info("***** Running training *****")
        for step, batch in enumerate(train_dataloader):
            ####################################
            # 1. Prepare Inputs
            # Get latents for the target image
            latents = vae.encode(batch["images"].to(dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Get latents for the conditioning (masked) image
            conditionings = vae.encode(batch["conditioning_images"].to(dtype=weight_dtype)).latent_dist.sample()
            conditionings = conditionings * vae.config.scaling_factor

            # Prepare mask (must match latent resolution)
            masks = batch["masks"].to(dtype=weight_dtype)
            masks = F.interpolate(masks, size=latents.shape[2:])

            # Forward Diffusion
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Concatenate for Inpainting UNet (9 channels total)
            inputs = torch.cat([noisy_latents, masks, conditionings], dim=1)

            # 2. Extract Conditioning Features
            # Important: Include 'adapter' in the accumulate call
            with accelerator.accumulate(unet, text_encoder, adapter):
                # Text Embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                # Geometric Embeddings from DINOv2

                # Force unwrap the reference images before passing to the adapter
                ref_images = batch["ref_pixel_values"]
                if isinstance(ref_images, (list, tuple)):
                    ref_images = ref_images[0]

                # print(f"DEBUG: Passing to adapter - Type: {type(ref_images)}, Shape: {ref_images.shape}")
                # Now pass the clean tensor
                geom_hidden_states = adapter(ref_images.to(accelerator.device))                
                # geom_hidden_states = geom_hidden_states.to(dtype=weight_dtype)  # Already in weight_dtype

                # Combined Cross-Attention Source
                combined_hidden_states = torch.cat([encoder_hidden_states, geom_hidden_states], dim=1)
                # print(f"Combined cross-attention shape: {combined_hidden_states.shape}") # Removed () and fixed plural name

                # print("DEBUG 4: Calling UNet")
                # 3. Predict Noise
                model_pred = unet(
                    sample=inputs, 
                    timestep=timesteps, 
                    encoder_hidden_states=combined_hidden_states
                    ).sample

                # print('Successful pass to unet')
                 
                # 4. Frequency-Aware Loss
                # Use the custom function you defined at the top of the script
                loss = frequency_weighted_loss(model_pred, noise, alpha_hf=args.freq_loss_alpha)

                # Backprop
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    # ADDED: adapter.parameters() needs clipping too
                    params_to_clip = itertools.chain(
                        unet.parameters(), 
                        text_encoder.parameters(), 
                        adapter.parameters() 
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
            ####################################

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                if args.report_to == "wandb":
                    accelerator.print(progress_bar)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        unet.eval()
                        text_encoder.eval()
                        
                        log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            adapter
                        )

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # 1. Properly unwrap and merge the UNet
        unwrapped_unet = accelerator.unwrap_model(unet)
        merged_unet = unwrapped_unet.merge_and_unload()

        # 2. Properly unwrap and merge the Text Encoder
        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True)
        merged_text_encoder = unwrapped_text_encoder.merge_and_unload()

        # 3. Load the pipeline with the merged components
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=merged_unet,
            text_encoder=merged_text_encoder,
            revision=args.revision,
            torch_dtype=weight_dtype # Good practice to ensure dtype consistency
        )

        pipeline.save_pretrained(args.output_dir)

        # --- ADD THIS PART ---
        # Save the custom DINOv2 adapter state
        adapter_save_path = os.path.join(args.output_dir, "adapter.bin")
        unwrapped_adapter = accelerator.unwrap_model(adapter)
        torch.save(unwrapped_adapter.state_dict(), adapter_save_path)
        # print(f"Custom adapter saved to {adapter_save_path}")
        # ---------------------


        # Final inference
        images = log_validation(
            vae,
            text_encoder,
            tokenizer,
            unet,
            args,
            accelerator,
            weight_dtype,
            global_step,
            adapter
        )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)
