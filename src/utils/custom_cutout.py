"""
Custom Cutout implementation that fills erased regions with uniform random values [0, 1]
instead of standard normal distribution values.
"""

import torch
import torchvision.transforms.v2 as T
import math
from typing import Tuple


class UniformCutout(T.Transform):
    """
    Custom Cutout that fills erased regions with uniform random values in [0, 1].
    
    This class implements RandomErasing behavior but modifies the filling
    to use uniform distribution instead of normal distribution when value='random'.
    
    Args:
        p: Probability that the random erasing operation will be performed.
        scale: Range of proportion of erased area against input image.
        ratio: Range of aspect ratio of erased area.
        value: Erasing value. 
            - If 'random', uses uniform distribution [0, 1] (different from original RandomErasing)
            - If 'zeros', fills with zeros
            - If a number, fills with that constant value
        inplace: Whether to apply the operation in-place.
    """
    
    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: str = 'random',
        inplace: bool = False
    ):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace
        
    def forward(self, img):
        """Apply cutout to the input image."""
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Input should be a torch.Tensor, got {type(img)}")
            
        if torch.rand(1) >= self.p:
            return img
            
        if not self.inplace:
            img = img.clone()
            
        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        area = img_h * img_w
        
        log_ratio = torch.log(torch.tensor(self.ratio))
        
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
            
            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            
            if not (h < img_h and w < img_w):
                continue
                
            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            
            # Determine the fill value
            if self.value == 'random':
                # Use uniform distribution [0, 1] instead of normal distribution
                v = torch.empty([img_c, h, w], dtype=img.dtype, device=img.device).uniform_(0, 1)
            elif self.value == 'zeros':
                v = torch.zeros([img_c, h, w], dtype=img.dtype, device=img.device)
            else:
                # Use constant value
                v = torch.empty([img_c, h, w], dtype=img.dtype, device=img.device).fill_(self.value)
            
            # Apply cutout
            img[..., i:i+h, j:j+w] = v
            return img
            
        return img


if __name__ == "__main__":
    """Test code for custom cutout implementation"""
    
    print("Testing UniformCutout Implementation")
    print("=" * 50)
    
    # Create a test image (3, 64, 64) filled with 0.5
    test_img = torch.full((3, 64, 64), 0.5)
    
    # Test 1: Basic functionality
    print("\nTest 1: Basic functionality")
    cutout = UniformCutout(p=1.0, scale=(0.1, 0.2), ratio=(1.0, 1.0), value='random')
    transformed = cutout(test_img.clone())
    
    diff = (transformed - test_img).abs().sum()
    print(f"  Cutout applied: {diff > 0}")
    
    if diff > 0:
        diff_map = (transformed - test_img).abs().sum(dim=0)
        modified_pixels = (diff_map > 0).sum()
        print(f"  Modified pixels: {modified_pixels.item()}")
        
        mask = diff_map > 0
        modified_values = transformed[:, mask]
        print(f"  Modified region stats:")
        print(f"    Min: {modified_values.min():.4f}")
        print(f"    Max: {modified_values.max():.4f}")
        print(f"    Mean: {modified_values.mean():.4f}")
    
    # Test 2: Probability test
    print("\nTest 2: Probability p=0.5 (100 trials)")
    cutout = UniformCutout(p=0.5, scale=(0.1, 0.2), ratio=(1.0, 1.0), value='random')
    applications = 0
    
    for _ in range(100):
        transformed = cutout(test_img.clone())
        if (transformed - test_img).abs().sum() > 0:
            applications += 1
    
    print(f"  Cutout applied {applications}/100 times (expected ~50)")
    
    # Test 3: Value range verification for uniform distribution
    print("\nTest 3: Uniform distribution [0, 1] verification (100 applications)")
    cutout = UniformCutout(p=1.0, scale=(0.1, 0.2), ratio=(1.0, 1.0), value='random')
    all_min = []
    all_max = []
    
    for _ in range(100):
        transformed = cutout(test_img.clone())
        diff_map = (transformed - test_img).abs().sum(dim=0)
        mask = diff_map > 0
        if mask.any():
            modified_values = transformed[:, mask]
            all_min.append(modified_values.min().item())
            all_max.append(modified_values.max().item())
    
    if all_min:
        print(f"  Global min across all trials: {min(all_min):.4f} (should be ~0)")
        print(f"  Global max across all trials: {max(all_max):.4f} (should be ~1)")
        print(f"  Average min: {sum(all_min)/len(all_min):.4f}")
        print(f"  Average max: {sum(all_max)/len(all_max):.4f}")
    
    # Test 4: Fixed size approximation (16x16 on 64x64 image)
    print("\nTest 4: Fixed size 16x16 approximation")
    # 16x16 = 256 pixels, 64x64 = 4096 pixels, ratio = 256/4096 = 0.0625
    cutout = UniformCutout(p=1.0, scale=(0.0625, 0.0625), ratio=(1.0, 1.0), value='random')
    
    pixel_counts = []
    for _ in range(10):
        transformed = cutout(test_img.clone())
        diff_map = (transformed - test_img).abs().sum(dim=0)
        modified_pixels = (diff_map > 0).sum().item()
        if modified_pixels > 0:
            pixel_counts.append(modified_pixels)
    
    if pixel_counts:
        avg_pixels = sum(pixel_counts) / len(pixel_counts)
        print(f"  Average modified pixels: {avg_pixels:.1f} (target: 256)")
        print(f"  Actual sizes: {pixel_counts}")
    
    # Test 5: Compare with standard RandomErasing
    print("\nTest 5: Comparison with standard RandomErasing")
    
    # Our uniform cutout
    uniform_cutout = UniformCutout(p=1.0, scale=(0.1, 0.2), ratio=(1.0, 1.0), value='random')
    uniform_transformed = uniform_cutout(test_img.clone())
    
    # Standard RandomErasing
    standard_cutout = T.RandomErasing(p=1.0, scale=(0.1, 0.2), ratio=(1.0, 1.0), value='random')
    standard_transformed = standard_cutout(test_img.clone())
    
    # Compare statistics
    for name, img in [("UniformCutout", uniform_transformed), 
                      ("Standard RandomErasing", standard_transformed)]:
        diff_map = (img - test_img).abs().sum(dim=0)
        mask = diff_map > 0
        if mask.any():
            values = img[:, mask]
            print(f"\n  {name}:")
            print(f"    Min: {values.min():.4f}, Max: {values.max():.4f}")
            print(f"    Mean: {values.mean():.4f}, Std: {values.std():.4f}")
    
    print("\n✓ All tests completed successfully!")