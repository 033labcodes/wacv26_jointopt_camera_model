import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def save_classification_samples(save_dir, images, predictions, targets, num_samples=5):
    """Save sample images for classification task."""
    num_samples = min(num_samples, len(images))
    
    # Build grid images
    grid_images = []
    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title(f'Pred: {predictions[i]}\nGT: {targets[i]}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_{i:03d}.png'))
        plt.close()
        
        # Add to grid
        grid_images.append(torch.from_numpy(img).permute(2, 0, 1))
    
    # Save grid
    if grid_images:
        grid = make_grid(grid_images, nrow=min(5, num_samples), padding=2)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        Image.fromarray(grid).save(os.path.join(save_dir, 'samples_grid.png')) 