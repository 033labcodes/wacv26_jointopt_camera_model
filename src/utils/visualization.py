import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def save_classification_samples(save_dir, images, predictions, targets, num_samples=5):
    """分類タスクのサンプル画像を保存する
    
    Args:
        save_dir (str): 保存先ディレクトリ
        images (list): RGB画像のリスト（torch.Tensor）
        predictions (list): 予測ラベルのリスト
        targets (list): 正解ラベルのリスト
        num_samples (int): 保存するサンプル数
    """
    # サンプル数を調整
    num_samples = min(num_samples, len(images))
    
    # グリッド画像の作成
    grid_images = []
    for i in range(num_samples):
        # RGB画像をnumpyに変換
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        
        # 予測と正解ラベルを画像に描画
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title(f'Pred: {predictions[i]}\nGT: {targets[i]}')
        plt.axis('off')
        
        # 画像をnumpy配列に変換
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_{i:03d}.png'))
        plt.close()
        
        # グリッド用に画像を追加
        grid_images.append(torch.from_numpy(img).permute(2, 0, 1))
    
    # グリッド画像の作成と保存
    if grid_images:
        grid = make_grid(grid_images, nrow=min(5, num_samples), padding=2)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        Image.fromarray(grid).save(os.path.join(save_dir, 'samples_grid.png')) 