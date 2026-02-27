import os
import json
import yaml
import torch
import timm
import random
import numpy as np
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from torchvision import models
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from hsi_dataset import HFD100_Dataset
from model_wrapper import ModelWrapper
from models.css_model import CSSModel
from models.isp_model import ColorCorrectionMatrix, DerivativeClippingGamma
from utils.visualization import save_classification_samples


def load_config(config_path):
    """Load config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return validate_config(config)


def validate_config(config):
    """Validate config (data_dir optional, filled from env/default)."""
    if 'data_dir' not in config or config['data_dir'] is None:
        config['data_dir'] = os.environ.get('HFD100_DATA_DIR', './data')
    for field in ['camera_name', 'device']:
        if field not in config or config[field] is None:
            raise ValueError(f'Config missing required field: {field}')
    return config


def load_eval_config(eval_config_path):
    """Load evaluation config (eval_config.yaml)."""
    default_eval_cfg = {
        'run_parameters': {
            'train_run_dir': 'runs/train/your_run_name',
            'checkpoint': 'best_model',
            'output_dir': 'eval_output',
            'batch_size': 32,
            'num_workers': 4,
            'num_visualization_samples': 20,
        },
        'load_weights': {'css': True, 'gamma': True, 'ccm': True, 'base': True},
    }
    if not os.path.exists(eval_config_path):
        print(f"Warning: eval config {eval_config_path} not found. Using defaults.")
        return default_eval_cfg
    with open(eval_config_path, 'r') as f:
        eval_cfg = yaml.safe_load(f)

    # Default run_parameters
    if 'run_parameters' not in eval_cfg:
        eval_cfg['run_parameters'] = {}
    for k, v in default_eval_cfg['run_parameters'].items():
        eval_cfg['run_parameters'].setdefault(k, v)

    # Default load_weights
    if 'load_weights' not in eval_cfg:
        eval_cfg['load_weights'] = {}
    for k, v in default_eval_cfg['load_weights'].items():
        eval_cfg['load_weights'].setdefault(k, v)

    return eval_cfg


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate CSS model')
    parser.add_argument('--config', type=str, required=True, help='Training config path')
    parser.add_argument('--eval_config', type=str, required=True, help='Eval config path (eval_config.yaml)')
    return parser.parse_args()


def setup_dataset(config, batch_size, num_workers):
    """Setup test dataset."""
    if 'dataset_name' not in config:
        raise ValueError("Config must contain 'dataset_name'.")
        
    data_dir = config.get('data_dir', os.environ.get('HFD100_DATA_DIR', './data'))
    test_dataset = HFD100_Dataset(
        dataset_name=config['dataset_name'],
        dataset_type='test',
        camera_name=config['camera_name'],
        data_dir=data_dir
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader, test_dataset.num_classes()


def setup_models(train_config, train_run_dir_path, device, eval_cfg, num_classes):
    """Setup models and load weights."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_subdir = eval_cfg.get('run_parameters', {}).get('checkpoint', 'best_model')
    
    # CSS model
    init_css_weights_path = os.path.join(script_dir, 'camera_parameters/css', f"cmf_{train_config['camera_name']}.pt")
    
    if not os.path.exists(init_css_weights_path):
        raise FileNotFoundError(f"CSS weights not found: {init_css_weights_path}")
    init_css_weights = torch.load(init_css_weights_path).float()
    css_model = CSSModel(init_weights=init_css_weights, in_channels=init_css_weights.shape[1], trainable=False).to(device)

    if eval_cfg.get('load_weights', {}).get('css', True):
        weights_path = os.path.join(train_run_dir_path, checkpoint_subdir, 'css_model.pth')
        print(f"Loading CSS weights: {weights_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"CSS weights not found: {weights_path}")
        css_model.load_state_dict(torch.load(weights_path))
    else:
        print("CSS: Using default CMF (no trained weights).")
    css_model.eval()
    
    # Gamma model
    gamma_model = DerivativeClippingGamma(init_gamma=1/2.2, grad_th=train_config['gradient_clipping'], trainable=train_config['train_gamma']).to(device)
    if eval_cfg.get('load_weights', {}).get('gamma', True):
        weights_path = os.path.join(train_run_dir_path, checkpoint_subdir, 'gamma_model.pth')
        print(f"Loading gamma weights: {weights_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Gamma weights not found: {weights_path}")
        gamma_model.load_state_dict(torch.load(weights_path))
    else:
        print("Gamma: Using default (no trained weights).")
    gamma_model.eval()
    
    # CCM model (same logic as train: XYZ -> ccm_sRGB)
    if train_config['camera_name'] == 'XYZ':
        init_ccm_weights_path = os.path.join(script_dir, 'camera_parameters/ccm', "ccm_sRGB.pt")
    else:
        init_ccm_weights_path = os.path.join(script_dir, 'camera_parameters/ccm', f"ccm_{train_config['camera_name']}.pt")
    
    if os.path.exists(init_ccm_weights_path):
        init_ccm_weights = torch.load(init_ccm_weights_path).float()
    else:
        if train_config['camera_name'] != 'sRGB':
            print(f"Warning: CCM file not found {init_ccm_weights_path}. Using identity for {train_config['camera_name']}.")
        init_ccm_weights = torch.eye(3).float()
    ccm_model = ColorCorrectionMatrix(init_ccm=init_ccm_weights, trainable=False).to(device)
    if eval_cfg.get('load_weights', {}).get('ccm', True):
        weights_path = os.path.join(train_run_dir_path, checkpoint_subdir, 'ccm_model.pth')
        print(f"Loading CCM weights: {weights_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"CCM weights not found: {weights_path}")
        ccm_model.load_state_dict(torch.load(weights_path))
    else:
        print(f"CCM: Using default for {train_config['camera_name']} (no trained weights).")
    ccm_model.eval()
    
    
    if eval_cfg.get('load_weights', {}).get('base', True):
        weights_path = os.path.join(train_run_dir_path, checkpoint_subdir, 'classification_model.pth')
        print(f"Loading classification weights: {weights_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Classification weights not found: {weights_path}")

        model_name = train_config.get('classification_model', 'ResNet')
        print(f"Loading classification model: {model_name}")

        base_model = None
        if model_name == 'ResNet':
            base_model = models.resnet18(weights=None)
            num_ftrs = base_model.fc.in_features
            base_model.fc = nn.Linear(num_ftrs, num_classes)
            loaded_state_dict = torch.load(weights_path, map_location=device)
        
            # Remap fc.1 -> fc (train uses Sequential with Dropout, eval uses Linear)
            key_mappings = {
                'fc.1.weight': 'fc.weight',
                'fc.1.bias': 'fc.bias',
                'heads.head.1.weight': 'heads.head.weight',
                'heads.head.1.bias': 'heads.head.bias',
            }
            
            new_state_dict = loaded_state_dict.copy()
            for saved_key, new_key in key_mappings.items():
                if saved_key in new_state_dict:
                    new_state_dict[new_key] = new_state_dict.pop(saved_key)
            base_model.load_state_dict(new_state_dict, strict=False)

        elif model_name == 'ViT':
            base_model = timm.create_model('vit_small_patch16_224.augreg_in21k', pretrained=True, img_size=64, num_classes=num_classes).to(device)
            pretrained_state_dict = torch.load(weights_path, map_location=device)
            base_model.load_state_dict(pretrained_state_dict)
        elif model_name == 'SE_ResNet':
            base_model = timm.create_model('seresnet50', pretrained=True).to(device)
            num_ftrs = base_model.fc.in_features
            base_model.fc = nn.Linear(num_ftrs, num_classes)
            loaded_state_dict = torch.load(weights_path, map_location=device)
        
            key_mappings = {
                'fc.1.weight': 'fc.weight',
                'fc.1.bias': 'fc.bias',
                'heads.head.1.weight': 'heads.head.weight',
                'heads.head.1.bias': 'heads.head.bias',
            }
            
            new_state_dict = loaded_state_dict.copy()
            for saved_key, new_key in key_mappings.items():
                if saved_key in new_state_dict:
                    new_state_dict[new_key] = new_state_dict.pop(saved_key)
            base_model.load_state_dict(new_state_dict, strict=False)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

        base_model = base_model.to(device)
        base_model.eval()
    
    return css_model, gamma_model, ccm_model, base_model


def custom_evaluate(model_wrapper, test_loader, output_dir, num_visualization_samples=20):
    """Evaluation (same pipeline as ModelWrapper.process_batch)."""
    css_model = model_wrapper.css_model
    gamma_model = model_wrapper.gamma_model
    ccm_model = model_wrapper.ccm_model
    base_model = model_wrapper.classification_model
    device = model_wrapper.device
    camera_name = model_wrapper.camera_name

    css_model.eval()
    gamma_model.eval()
    ccm_model.eval()
    base_model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    num_batches = len(test_loader)
    
    # Confusion matrix
    all_predictions = []
    all_targets = []
    
    # Visualization output dir
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Random indices for visualization
    total_samples = num_batches * test_loader.batch_size
    vis_indices = random.sample(range(total_samples), min(num_visualization_samples, total_samples))
    
    sample_count = 0
    sample_images = []
    sample_predictions = []
    sample_targets = []
    
    with torch.no_grad():
        for i, (inputs_hsi, target) in enumerate(test_loader):
            print(f'\rEvaluation: {i+1}/{num_batches}', end='')
            
            # HSI -> RGB (same order as process_batch)
            inputs_hsi = inputs_hsi.to(device)
            rgb_images = css_model(inputs_hsi)
            rgb_images = ccm_model(rgb_images)
            rgb_images = rgb_images.clamp(min=0)
            rgb_images = gamma_model(rgb_images)
            
            # Classification
            outputs = base_model(rgb_images)
            loss = model_wrapper.criterion(outputs, target.to(device))
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target.to(device)).sum().item()
            
            # Collect for confusion matrix
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.numpy())
            
            # Collect samples for visualization
            for j in range(len(rgb_images)):
                global_idx = i * test_loader.batch_size + j
                if global_idx in vis_indices:
                    sample_images.append(rgb_images[j])
                    sample_predictions.append(predicted[j].item())
                    sample_targets.append(target[j].item())
    
    print("\nEvaluation complete.")
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Classification report
    report = classification_report(all_targets, all_predictions, output_dict=True)
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Save visualization samples
    if sample_images:
        save_classification_samples(
            vis_dir,
            sample_images,
            sample_predictions,
            sample_targets,
            num_samples=len(sample_images)
        )
    
    # AA (Average Accuracy)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    aa = np.nanmean(class_accuracy)

    metrics = {
        'loss': total_loss / num_batches,
        'accuracy': 100. * correct / total,
        'average_accuracy': 100. * aa
    }
    
    return metrics


def save_eval_run_config(args, train_cfg, eval_cfg, output_dir):
    """Save eval run config (args, train config, eval config)."""
    full_eval_config = {
        'command_line_args': vars(args),
        'training_config': train_cfg,
        'evaluation_module_config': eval_cfg 
    }
    
    with open(os.path.join(output_dir, 'eval_run_config.yaml'), 'w') as f:
        yaml.dump(full_eval_config, f, default_flow_style=False, sort_keys=False)


def main():
    args = parse_args()
    train_config = load_config(args.config)
    eval_cfg = load_eval_config(args.eval_config)
    
    device = torch.device(train_config['device'])
    
    # output_dir from eval_cfg, prefixed with runs/eval
    output_dir_resolved = os.path.join('runs/eval', eval_cfg['run_parameters']['output_dir'])
    os.makedirs(output_dir_resolved, exist_ok=True)
    
    save_eval_run_config(args, train_config, eval_cfg, output_dir_resolved)
    
    # batch_size, num_workers from eval_cfg
    test_loader, num_classes = setup_dataset(
        train_config, 
        eval_cfg['run_parameters']['batch_size'], 
        eval_cfg['run_parameters']['num_workers']
    )
    
    css_model, gamma_model, ccm_model, base_model = setup_models(
        train_config, 
        eval_cfg['run_parameters']['train_run_dir'],
        device, 
        eval_cfg,
        num_classes
    )
    
    criterion = nn.CrossEntropyLoss()
    
    model_wrapper = ModelWrapper(
        css_model=css_model,
        gamma_model=gamma_model,
        ccm_model=ccm_model,
        classification_model=base_model,
        criterion=criterion,
        optimizer=None,
        lr_schedule=None,
        device=device,
        logger=None,
        camera_name=train_config['camera_name']
    )
    
    print(f"Starting evaluation: {output_dir_resolved}")
    metrics = custom_evaluate(
        model_wrapper=model_wrapper,
        test_loader=test_loader,
        output_dir=output_dir_resolved,
        num_visualization_samples=eval_cfg['run_parameters']['num_visualization_samples']
    )
    
    # Save results
    with open(os.path.join(output_dir_resolved, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nEvaluation complete. Results saved to {output_dir_resolved}")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Accuracy (OA): {metrics['accuracy']:.2f}%")
    print(f"Average Accuracy (AA): {metrics['average_accuracy']:.2f}%")


if __name__ == '__main__':
    main()
