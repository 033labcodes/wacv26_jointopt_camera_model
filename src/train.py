import os
import yaml
import torch
from huggingface_hub import hf_hub_download
import torch.hub
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
import torch.nn as nn
from torchvision import models
import timm 

from hsi_dataset import HFD100_Dataset
from train_logger import TrainLogger
from model_wrapper import ModelWrapper
from models.css_model import CSSModel
from models.isp_model import ColorCorrectionMatrix, DerivativeClippingGamma


def load_config(config_path):
    """Load config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CSS model')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--data_dir', type=str, default=None, help='Override data_dir (HDF5 directory)')
    parser.add_argument('--srgb_max_dir', type=str, default=None, help='Directory for sRGB max JSON files (default: same as data_dir)')
    parser.add_argument('--gradient_clipping', type=float, default=None, help='Directly set gradient clipping threshold')
    parser.add_argument('--css_smoothness_weight', type=float, default=0.0, help='Weight for CSS smoothness constraint loss (0.0 = no smoothness constraint)')
    parser.add_argument('--camera_name', type=str, default=None, help='Camera name to override config (e.g., Sony Nex5N)')
    parser.add_argument('--train_css', type=lambda x: (str(x).lower() == 'true'), default=None, help='Override train_css (true/false)')
    parser.add_argument('--css_lr', type=float, default=None, help='Override css_lr (float)')
    parser.add_argument('--train_ccm', type=lambda x: (str(x).lower() == 'true'), default=None, help='Override train_ccm (true/false)')
    parser.add_argument('--ccm_lr', type=float, default=None, help='Override ccm_lr (float)')
    parser.add_argument('--train_gamma', type=lambda x: (str(x).lower() == 'true'), default=None, help='Override train_gamma (true/false)')
    parser.add_argument('--gamma_lr', type=float, default=None, help='Override gamma_lr (float)')
    parser.add_argument('--dataset_name', type=str, default=None, help='Dataset name to override config (e.g., HFD100_Flower)')
    parser.add_argument('--train_classification', type=lambda x: (str(x).lower() == 'true'), default=None, help='Override train_classification (true/false)')
    parser.add_argument('--classification_lr', type=float, default=None, help='Override classification_lr (float)')
    parser.add_argument('--classification_model', type=str, default=None, help='Override classification_model (e.g., ResNet, ViT, SE_ResNet)')
    parser.add_argument('--save_name', type=str, default=None, help='Override save_name (e.g., wacv_exp1_resnet_nocss_noisp_leaves_canon)')
    return parser.parse_args()

# Hugging Face repo for pretrained classification weights
HF_PRETRAINED_REPO = "dekkaiinu/wacv26_jointopt_pretrained"
# Mapping: (classification_model, dataset_name) -> HF filename
_HF_PRETRAINED_FILES = {
    ("ResNet", "HFD100_Flower"): "resnet18_flower_classification_model.pth",
    ("ResNet", "HFD100_Leaves"): "resnet18_leaves_classification_model.pth",
    ("ViT", "HFD100_Flower"): "vits16_flower_classification_model.pth",
    ("ViT", "HFD100_Leaves"): "vits16_leaves_classification_model.pth",
    ("SE_ResNet", "HFD100_Flower"): "seresnet50_flower_classification_model.pth",
    ("SE_ResNet", "HFD100_Leaves"): "seresnet50_leaves_classification_model.pth",
}

def _get_classifier_weights_path(config):
    """Download and return path to classifier weights from Hugging Face."""
    key = (config["classification_model"], config["dataset_name"])
    filename = _HF_PRETRAINED_FILES.get(key)
    if not filename:
        return None
    return hf_hub_download(repo_id=HF_PRETRAINED_REPO, filename=filename)

# Config keys overridable by args (arg name same as config key)
_ARG_OVERRIDE_KEYS = [
    'data_dir', 'srgb_max_dir', 'camera_name', 'train_css', 'train_ccm', 'css_lr', 'ccm_lr', 'gamma_lr',
    'dataset_name', 'train_gamma', 'train_classification', 'classification_lr',
    'classification_model', 'save_name', 'gradient_clipping',
]

def apply_arg_overrides(args, config):
    """Apply arg values to config."""
    for key in _ARG_OVERRIDE_KEYS:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

def setup_dataset(config):
    """Setup dataset and dataloaders."""
    transform_list = [
        T.RandomHorizontalFlip(p=0.5),
        T.RandomChoice([
            T.RandomRotation([0, 0]),
            T.RandomRotation([90, 90]),
            T.RandomRotation([180, 180]),
            T.RandomRotation([270, 270]),
        ])
    ]
    train_transform = T.Compose(transform_list)

    data_dir = config.get('data_dir', os.environ.get('HFD100_DATA_DIR', './data'))
    srgb_max_dir = config.get('srgb_max_dir')
    train_dataset = HFD100_Dataset(dataset_name=config['dataset_name'], dataset_type='train', camera_name=config['camera_name'], transform=train_transform, data_dir=data_dir, srgb_max_dir=srgb_max_dir)
    val_dataset = HFD100_Dataset(dataset_name=config['dataset_name'], dataset_type='val', camera_name=config['camera_name'], data_dir=data_dir, srgb_max_dir=srgb_max_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)
    
    return train_loader, val_loader, train_dataset.num_classes()

def setup_models(config, num_classes, device):
    """Setup models."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    css_weights_path = os.path.join(script_dir, 'camera_parameters/css', f"cmf_{config['camera_name']}.pt")
    css_weights = torch.load(css_weights_path).float()

    css_model = CSSModel(init_weights=css_weights, in_channels=css_weights.shape[1], trainable=config['train_css']).to(device)
    
    if config['camera_name'] == 'XYZ':
        ccm_weights_path = os.path.join(script_dir, 'camera_parameters/ccm', "ccm_sRGB.pt")
    else:
        ccm_weights_path = os.path.join(script_dir, 'camera_parameters/ccm', f"ccm_{config['camera_name']}.pt")
    ccm_weights = torch.load(ccm_weights_path).float()
    ccm_model = ColorCorrectionMatrix(init_ccm=ccm_weights, trainable=config['train_ccm']).to(device)
    
    gamma_model = DerivativeClippingGamma(init_gamma=1/2.2, grad_th=config['gradient_clipping'], trainable=config['train_gamma']).to(device)

    
    weights_path = _get_classifier_weights_path(config)
    if weights_path:
        print(f"Loading classifier weights from: {weights_path}")

    if config['classification_model'] == 'ResNet':
        classification_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
        num_ftrs = classification_model.fc.in_features
        classification_model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_ftrs, num_classes)).to(device)

        if weights_path is not None:
            pretrained_state_dict = torch.load(weights_path, map_location=device)
            current_model_state_dict = classification_model.state_dict()
            
            new_state_dict = {}
            for key, param in pretrained_state_dict.items():
                if key == 'fc.weight':
                    new_state_dict['fc.1.weight'] = param
                elif key == 'fc.bias':
                    new_state_dict['fc.1.bias'] = param
                elif key in current_model_state_dict:
                    new_state_dict[key] = param
            
            current_model_state_dict.update(new_state_dict)
            classification_model.load_state_dict(current_model_state_dict)

    elif config['classification_model'] == 'ViT':
        classification_model = timm.create_model('vit_small_patch16_224.augreg_in21k', pretrained=True, img_size=64, num_classes=num_classes).to(device)
        if weights_path is not None:
            pretrained_state_dict = torch.load(weights_path, map_location=device)
            classification_model.load_state_dict(pretrained_state_dict)
    
    elif config['classification_model'] == 'SE_ResNet':
        classification_model = timm.create_model('seresnet50', pretrained=True).to(device)
        num_ftrs = classification_model.fc.in_features
        classification_model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_ftrs, num_classes)).to(device)
        if weights_path is not None:
            pretrained_state_dict = torch.load(weights_path, map_location=device)
            classification_model.load_state_dict(pretrained_state_dict)
    else:
        raise ValueError(f"Unsupported model type: {config['classification_model']}")

    return css_model, gamma_model, ccm_model, classification_model

def setup_optimizer(config, css_model, gamma_model, ccm_model, classification_model):
    """Setup optimizer and LR scheduler."""
    optimizer_param_groups = []
    
    if config['train_css']:
        optimizer_param_groups.append({'params': css_model.parameters(), 'lr': config['css_lr']})    
    if config['train_gamma']:
        optimizer_param_groups.append({'params': gamma_model.parameters(), 'lr': config['gamma_lr']})
    if config['train_ccm']:
        optimizer_param_groups.append({'params': ccm_model.parameters(), 'lr': config['ccm_lr']})    
    if config['train_classification']:
        optimizer_param_groups.append({'params': classification_model.parameters(), 'lr': config['classification_lr']})

    if not optimizer_param_groups:
        raise ValueError('No parameters to optimize')

    optimizer = optim.Adam(optimizer_param_groups)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step_size'], gamma=config['lr_gamma'])
    
    return optimizer, lr_scheduler

def main():
    args = parse_args()
    config = load_config(args.config)
    apply_arg_overrides(args, config)

    save_dir = os.path.join('runs/train', config['save_name'])
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    train_loader, val_loader, num_classes = setup_dataset(config)
    device = torch.device(config['device'])
    css_model, gamma_model, ccm_model, classification_model = setup_models(config, num_classes, device)
    
    logger = TrainLogger(save_dir=save_dir)
    
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer, lr_scheduler = setup_optimizer(config, css_model, gamma_model, ccm_model, classification_model)
    model_wrapper = ModelWrapper(
        css_model=css_model,
        gamma_model=gamma_model,
        ccm_model=ccm_model,
        classification_model=classification_model,
        criterion=criterion,
        optimizer=optimizer,
        lr_schedule=lr_scheduler,
        device=device,
        logger=logger,
        camera_name=config['camera_name'],
        css_smoothness_weight=args.css_smoothness_weight
    )
    
    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        logger.start_epoch()
        logger.update_epoch(epoch, config['epochs'])
        
        train_metrics = model_wrapper.train(train_loader)
        logger.log_train_loss(train_metrics)

        val_metrics = model_wrapper.evaluate(val_loader)
        logger.log_val_loss(val_metrics)
        
        logger.print_log(epoch, train_metrics, val_metrics)
        
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            print(f'Saving best model (val_loss: {val_metrics["val_loss"]:.4f})')
            logger.save_checkpoint(
                epoch=epoch,
                css_model=css_model,
                gamma_model=gamma_model,
                ccm_model=ccm_model,
                classification_model=classification_model,
                is_best=True
            )

        # Save latest every epoch (overwrite)
        logger.save_checkpoint(
            epoch=epoch,
            css_model=css_model,
            gamma_model=gamma_model,
            ccm_model=ccm_model,
            classification_model=classification_model,
            is_best=False
        )
    

if __name__ == '__main__':
    main()