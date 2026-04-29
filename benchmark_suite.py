import torch
from dreamsim import dreamsim

# Direct imports for TorchMetrics 1.9.0
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
# from torch.hub import dino_vits16

import open_clip
from PIL import Image
import numpy as np

from torchvision import transforms

        
class RealFillBench:

    def __init__(self, device="cuda"):
        self.device = device
        
        # 1. Pixel Metrics
        self.psnr = PSNR().to(device)
        self.ssim = SSIM().to(device)
        # 2. Perceptual Metric (The paper uses VGG-based LPIPS)
        self.lpips = LPIPS(net_type='vgg').to(device)

        # Standard Transform for Pixel Metrics (PSNR/SSIM)
        self.pixel_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(), # Scales to [0, 1] automatically
        ])

        # 3. dreamsim
        self.dreamsim_model, self.dreamsim_preprocess = dreamsim(pretrained=True, device=self.device)
        self.dreamsim_model.eval() # Set to evaluation mode
        
        # 4. Semantic Metrics (CLIP and DINO)
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )

        # dino initialization
        self.clip_model.to(device)
        self.dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)
        self.dino_model.eval()

        # DINO requires specific normalization
        self.dino_preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        

    def calculate_pixel_metrics(self, gen_path, target_path):
        img_gen = self.pixel_transform(Image.open(gen_path).convert('RGB')).unsqueeze(0).to(self.device)
        img_tgt = self.pixel_transform(Image.open(target_path).convert('RGB')).unsqueeze(0).to(self.device)
        
        return {
            "psnr": self.psnr(img_gen, img_tgt).item(),
            "ssim": self.ssim(img_gen, img_tgt).item(),
            "lpips": self.lpips(img_gen, img_tgt).item()
        }
        

    def calculate_clip_score(self, gen_img_path, ref_img_path):
        """Measures Identity Preservation (Similarity between result and reference)"""
        img1 = self.clip_preprocess(Image.open(gen_img_path)).unsqueeze(0).to(self.device)
        img2 = self.clip_preprocess(Image.open(ref_img_path)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feat1 = self.clip_model.encode_image(img1)
            feat2 = self.clip_model.encode_image(img2)
            feat1 /= feat1.norm(dim=-1, keepdim=True)
            feat2 /= feat2.norm(dim=-1, keepdim=True)
            similarity = (feat1 @ feat2.T).item()
        return similarity

    
    def calculate_dreamsim_score(self, img_path1, img_path2):
        # 1. Load images
        img1_pil = Image.open(img_path1).convert('RGB')
        img2_pil = Image.open(img_path2).convert('RGB')
        
        # 2. Preprocess
        img1 = self.dreamsim_preprocess(img1_pil).to(self.device)
        img2 = self.dreamsim_preprocess(img2_pil).to(self.device)
        
        # 3. DEBUG/FIX: Ensure they are exactly 4D [B, C, H, W]
        if img1.ndim == 3:
            img1 = img1.unsqueeze(0)
        elif img1.ndim == 5:
            img1 = img1.squeeze(0) # Remove accidental extra dimension
    
        if img2.ndim == 3:
            img2 = img2.unsqueeze(0)
        elif img2.ndim == 5:
            img2 = img2.squeeze(0)
    
        # 4. Run model
        with torch.no_grad():
            distance = self.dreamsim_model(img1, img2)
        return distance.item()

    
    def calculate_dino(self, gen_img_path, ref_img_path):
        """Calculates DINO semantic similarity (cosine similarity of CLS tokens)"""
        img1 = self.dino_preprocess(Image.open(gen_img_path).convert('RGB')).unsqueeze(0).to(self.device)
        img2 = self.dino_preprocess(Image.open(ref_img_path).convert('RGB')).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # DINO models return the CLS token by default in eval mode
            feat1 = self.dino_model(img1)
            feat2 = self.dino_model(img2)
            
            # Normalize for cosine similarity
            feat1 = torch.nn.functional.normalize(feat1, dim=-1)
            feat2 = torch.nn.functional.normalize(feat2, dim=-1)
            
            similarity = (feat1 @ feat2.T).item()
        return similarity

    def run_sanity_check(sample_name, base_dir):
        # Initialize
        device = "cuda" if torch.cuda.is_available() else "cpu"
        evaluator = RealFillBench(device=device)
        print(f"Evaluator initialized on {device}")

        # Define Paths
        gen_path = os.path.join(base_dir, "outputs", sample_name, "generated.png")
        target_path = os.path.join(base_dir, "data", sample_name, "target", "target.png")

        if not os.path.exists(gen_path) or not os.path.exists(target_path):
            print(f"Error: Missing files for {sample_name}")
            print(f"Checked: {gen_path}")
            return None


    def calculate_all(self, gen_path, target_path, ref_path=None):
        """Unified entry point for benchmarking"""
        # 1. Pixel and Perceptual (Gen vs GT)
        results = self.calculate_pixel_metrics(gen_path, target_path)
        
        # 2. DreamSim (Gen vs GT)
        results['dreamsim'] = self.calculate_dreamsim_score(gen_path, target_path)
        
        # 3. Identity (Gen vs Reference) - If ref_path is provided
        if ref_path:
            results['clip_id'] = self.calculate_clip_score(gen_path, ref_path)
            results['dino_id'] = self.calculate_dino(gen_path, ref_path)
    
        return results
    
