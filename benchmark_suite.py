# import torch
# import dreamsim
# # For versions >= 1.0.0
# from torchmetrics.image import PeakSignalToNoiseRatio
# from torchmetrics.image import StructuralSimilarityIndexMeasure
# # LPIPS is usually kept in its own image submodule
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
# # from torch.hub import dino_vits16

# import open_clip
# from PIL import Image
# import numpy as np

# # Direct imports for TorchMetrics 1.9.0
# from torchmetrics.image import PeakSignalToNoiseRatio
# from torchmetrics.image import StructuralSimilarityIndexMeasure
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# from torchvision import transforms

try:
    # Modern torchmetrics (1.0.0+)
    from torchmetrics.image import PeakSignalToNoiseRatio as PSNR
    from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
except ImportError:
    try:
        # Older versions or specific sub-paths
        from torchmetrics.regression import PeakSignalToNoiseRatio as PSNR
        from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
    except ImportError:
        # Fallback for even older versions
        from torchmetrics import PeakSignalToNoiseRatio as PSNR
        from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
        from torchmetrics import LearnedPerceptualImagePatchSimilarity as LPIPS
        
class RealFillBench:

    def __init__(self, device="cuda"):
        self.device = device
        # 1. Pixel Metrics
        self.psnr = PeakSignalToNoiseRatio().to(device)
        self.ssim = StructuralSimilarityIndexMeasure().to(device)
        
        # 2. Perceptual Metric (The paper uses VGG-based LPIPS)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
        
        # 3. Semantic Metrics (CLIP and DINO)
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )

        self.dreamsim_model, self.dreamsim_preprocess = dreamsim(
            pretrained=True, device=self.device)

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
        
    def calculate_pixel_metrics(self, pred, target):
        """pred/target: Tensors [1, 3, H, W] normalized [0, 1]"""
        return {
            "PSNR": self.psnr(pred, target).item(),
            "SSIM": self.ssim(pred, target).item(),
            "LPIPS": self.lpips(pred, target).item()
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
        img1 = self.dreamsim_preprocess(Image.open(img_path1).convert('RGB')).to(self.device)
        img2 = self.dreamsim_preprocess(Image.open(img_path2).convert('RGB')).to(self.device)
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


    def calculate_all(self, gen_path, target_path):
        # Existing metrics...
        results = {}
        # ... (PSNR, SSIM, LPIPS, CLIP)
        
        # New DINO Metric
        results['dino'] = self.calculate_dino(gen_path, target_path)
        
        return results

    # def evaluation(gen_path, target_path):
    #     # Calculate
    #     results = evaluator.calculate_all(gen_path, target_path)
        
    #     print(f"--- Results for {sample_name} ---")
    #     for metric, value in results.items():
    #         print(f"{metric}: {value}")
        
    #     return results