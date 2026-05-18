import torch
import torch.nn as nn

"""
extract geomtric features from reference images  
"""

import torch
import torch.nn as nn

class DINOv2GeometricAdapter(nn.Module):
    def __init__(self, dinov2_model, sd_dim=1024):
        super().__init__()
        self.dinov2 = dinov2_model 
        print('Running dino model')

        # Freeze DINOv2
        for param in self.dinov2.parameters():
            param.requires_grad = False 
        # print(f'frozen dinov2 backbone')

        dinov2_out_dim = self.dinov2.embed_dim
    
        # We define the internal layers of the "projection" 
        # so we can control exactly what goes into them.
        # print("DEBUG 2: Inside Adapter, entering Projection")
        
        self.proj = nn.ModuleDict({
            "fc1": nn.Linear(dinov2_out_dim, 768),
            "norm": nn.LayerNorm(768),
            "gelu": nn.GELU(),
            "fc2": nn.Linear(768, sd_dim)
        })
        # print(f'Initialized adapter with {sd_dim} dimensions')

    def to(self, *args, **kwargs):
        """
        Custom to() method that only converts the projection layers.
        The DINOv2 backbone is kept in its original dtype (typically FP32)
        to avoid numerical issues with batch normalization in FP16.
        """
        # Handle both positional and keyword arguments like PyTorch's to() method
        device = None
        dtype = None
        
        # Parse positional arguments first (like PyTorch's standard to() method)
        if len(args) > 0:
            # First positional arg can be device or dtype
            arg = args[0]
            if isinstance(arg, torch.device) or isinstance(arg, str):
                device = arg
            elif isinstance(arg, torch.dtype):
                dtype = arg
        
        # Second positional arg (if any) is the other type
        if len(args) > 1:
            arg = args[1]
            if device is None and (isinstance(arg, torch.device) or isinstance(arg, str)):
                device = arg
            elif dtype is None and isinstance(arg, torch.dtype):
                dtype = arg
        
        # Override with keyword arguments if provided
        if 'device' in kwargs:
            device = kwargs['device']
        if 'dtype' in kwargs:
            dtype = kwargs['dtype']
        
        # Only convert the projection layers, not the DINOv2 backbone
        if device is not None:
            self.proj.to(device=device)
        if dtype is not None:
            self.proj.to(dtype=dtype)
        
        return self

    def forward(self, ref_image, *args, **kwargs):
        # 1. Input Safety: Handle potential tuple wrapping from the dataloader
        if isinstance(ref_image, (list, tuple)):
            ref_image = ref_image[0]

        # 2. Ensure 4D (Add batch dim if missing)
        if ref_image.ndim == 3:
            ref_image = ref_image.unsqueeze(0)

        # 3. Sync Device with the frozen backbone (keep dtype as-is)
        backbone_param = next(self.dinov2.parameters())
        ref_image = ref_image.to(device=backbone_param.device)

        # 4. Feature Extraction
        with torch.no_grad():
            features = self.dinov2.forward_features(ref_image)
            
            # Extraction: DINOv2 'forward_features' usually returns a dict
            if isinstance(features, dict):
                # 'x_norm_patchtokens' is the standard key for DINOv2 patch features
                patch_tokens = features.get("x_norm_patchtokens", features.get("patch_tokens"))
            elif isinstance(features, (list, tuple)):
                patch_tokens = features[0]
            else:
                patch_tokens = features

        # 5. Final Guard: Ensure we have a pure Tensor
        if isinstance(patch_tokens, (list, tuple)):
            patch_tokens = patch_tokens[0]

        if not torch.is_tensor(patch_tokens):
             raise TypeError(f"Critical Failure: patch_tokens is {type(patch_tokens)}, not a Tensor!")

        # Cast to match projection layer dtype
        patch_tokens = patch_tokens.to(dtype=self.proj["fc1"].weight.dtype)

        # 6. Projection
        h = self.proj["fc1"](patch_tokens)
        h = self.proj["norm"](h)
        h = self.proj["gelu"](h)
        geometric_embeddings = self.proj["fc2"](h)
        
        return geometric_embeddings

    def _remap_old_state_dict(self, state_dict):
        if any(key.startswith("proj.0") or key.startswith("proj.1") or key.startswith("proj.3") for key in state_dict.keys()):
            remap = {
                "proj.0.weight": "proj.fc1.weight",
                "proj.0.bias": "proj.fc1.bias",
                "proj.1.weight": "proj.norm.weight",
                "proj.1.bias": "proj.norm.bias",
                "proj.3.weight": "proj.fc2.weight",
                "proj.3.bias": "proj.fc2.bias",
            }
            return {remap.get(key, key): value for key, value in state_dict.items()}
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        state_dict = self._remap_old_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)
