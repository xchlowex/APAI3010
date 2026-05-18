import argparse
import os
import glob
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from diffusers import (
    StableDiffusionInpaintPipeline, 
    UNet2DConditionModel,
    DDPMScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms as T
from models import DINOv2GeometricAdapter

def parse_args():
    parser = argparse.ArgumentParser(description="GeoFill Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained LoRA/UNet folder")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to adapter.bin")
    parser.add_argument("--validation_image", type=str, required=True)
    parser.add_argument("--validation_mask", type=str, required=True)
    parser.add_argument("--ref_image", type=str, required=True, help="Reference image for DINOv2")
    parser.add_argument("--output_dir", type=str, default="./test-infer/")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512) # SD2 usually 512 or 768
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    weight_dtype = torch.float32 # Keep float32 for stability unless OOM

    # 1. Load Base Pipeline
    local_path = "sd2-community/stable-diffusion-2-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(local_path, torch_dtype=weight_dtype)
    
    # 2. Load Trained Components (UNet and Text Encoder LoRAs)
    pipe.unet = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet").to(device)
    pipe.text_encoder = CLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder").to(device)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    
    # 3. Load DINOv2 & Adapter
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()
    adapter = DINOv2GeometricAdapter(dinov2_model, sd_dim=pipe.unet.config.cross_attention_dim).to(device)
    
    adapter_state = torch.load(args.adapter_path, map_location=device)
    adapter.load_state_dict(adapter_state)
    adapter.eval()
    pipe.to(device)

    # 4. Prepare Images
    image = Image.open(args.validation_image).convert("RGB").resize((args.resolution, args.resolution))
    mask_image = Image.open(args.validation_mask).convert("L").resize((args.resolution, args.resolution))
    ref_image = Image.open(args.ref_image).convert("RGB").resize((224, 224)) # DINOv2 default size

    ref_paths = glob.glob(os.path.join(os.path.dirname(args.ref_image), "*.png"))
    all_ref_tensors = []

    # DINO Preprocessing
    transform = T.Compose([
        T.ToTensor(), 
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for path in ref_paths:
        ref_img = Image.open(path).convert("RGB").resize((224, 224))
        ref_t = transform(ref_img).unsqueeze(0).to(device, dtype=weight_dtype)
        all_ref_tensors.append(ref_t)

    # Stack them into one big batch: [N, 3, 224, 224]
    ref_batch = torch.cat(all_ref_tensors, dim=0)

    # 5. Extract Combined Embeddings (Token Fusion)
    prompt = "a photo of sks"
    with torch.no_grad():
        # Get Text Tokens
        text_inputs = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
        text_embeddings = pipe.text_encoder(text_inputs.input_ids)[0]
        
    # NEW: Get Multi-Reference Geometric Tokens
        # Process the whole batch through DINO and the Adapter
        batch_geom_embeddings = adapter(ref_batch) # Shape: [N, 256, 1024]
        
        # Average across the N images to get a "Global Object Representation"
        geom_embeddings = torch.mean(batch_geom_embeddings, dim=0, keepdim=True) # Shape: [1, 256, 1024]
        
        # Concatenate as before
        combined_embeddings = torch.cat([text_embeddings, geom_embeddings], dim=1)
        
        
        # Handle Unconditioned (Negative Prompt) for Guidance Scale
        uncond_inputs = pipe.tokenizer("", padding="max_length", max_length=combined_embeddings.shape[1], return_tensors="pt").to(device)
        # Note: In practice, you'd usually pad the null text to match the combined length
        uncond_embeddings = pipe.text_encoder(pipe.tokenizer("", padding="max_length", max_length=77, return_tensors="pt").to(device).input_ids)[0]
        # Pad unconditioned to match combined length
        padding = torch.zeros((1, geom_embeddings.shape[1], combined_embeddings.shape[2])).to(device, dtype=weight_dtype)
        uncond_combined = torch.cat([uncond_embeddings, padding], dim=1)

    # 6. Run Inference
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None

    for idx in range(4): # Reduced count for testing
        result = pipe(
            prompt_embeds=combined_embeddings,
            negative_prompt_embeds=uncond_combined,
            image=image, 
            mask_image=mask_image, 
            height=args.resolution, 
            width=args.resolution, 
            num_inference_steps=50, 
            guidance_scale=7.5, 
            generator=generator, 
        ).images[0]
        
        result = Image.composite(result.convert("RGB"), image, mask_image)
        result.save(f"{args.output_dir}/geofill_{idx}.png")
        print(f"Saved: {idx}.png")

    print("Inference Complete.")