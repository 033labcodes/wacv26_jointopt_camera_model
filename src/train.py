import os
import json
import yaml
import torch
import torch.hub
import wandb
import argparse
from datetime import datetime
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
import torch.nn as nn
from torchvision import models
import sys
import math
import timm 

from hsi_dataset import HFD100_Dataset
from train_logger import TrainLogger
from model_wrapper import ModelWrapper
from models.custom_css import CustomCSS
from models.custom_isp import GradientLimitedGammaV1, GradientLimitedGammaV2, ColorCorrectionMatrix, Gamma, GammaEpsAdd, GammaEpsClip
from utils.custom_cutout import UniformCutout


def load_config(config_path):
    """設定ファイルを読み込む"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return validate_config(config)

def validate_config(config):
    """設定の検証"""
    required_fields = ['data_dir', 'camera_name', 'device']
    for field in required_fields:
        if field not in config or config[field] is None:
            raise ValueError(f'設定ファイルに必須フィールド {field} がありません')

    # gradient_clipping の検証とデフォルト値の設定
    default_grad_clip = 2.0 # GradientLimitedGammaV2のデフォルトthに合わせるか、適切な値を設定
    if 'gradient_clipping' not in config or config['gradient_clipping'] is None:
        print(f"Warning: 'gradient_clipping' is not set in config. Using default value: {default_grad_clip}")
        config['gradient_clipping'] = default_grad_clip
    elif not isinstance(config['gradient_clipping'], (int, float)):
        print(f"Warning: 'gradient_clipping' is not a number (found: {config['gradient_clipping']}). Using default value: {default_grad_clip}")
        config['gradient_clipping'] = default_grad_clip
    elif config['gradient_clipping'] <= 0:
        print(f"Warning: 'gradient_clipping' must be positive (found: {config['gradient_clipping']}). Using default value: {default_grad_clip}")
        config['gradient_clipping'] = default_grad_clip

    config.setdefault('log_scale_input', None) # Default for new param
    config.setdefault('gamma_model_type', 'GradientLimitedGammaV2') # Default gamma model
    config.setdefault('gamma_eps', 1e-7) # Default epsilon for new gamma models
    return config

def parse_args():
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(description='Train CSS model')
    parser.add_argument('--config', type=str, required=True, help='設定ファイルのパス')
    parser.add_argument('--gradient_clipping', type=float, default=None, help='Directly set gradient clipping threshold (overridden by log_scale_input if provided)')
    parser.add_argument('--log_scale_input', type=float, default=None, help='Input value (log scale) to calculate gradient for gamma clipping (used by WandB sweep)')
    parser.add_argument('--css_smoothness_weight', type=float, default=0.0, help='Weight for CSS smoothness constraint loss (0.0 = no smoothness constraint)')
    parser.add_argument('--gamma_model_type', type=str, default=None, help='Type of gamma model to use (e.g., GradientLimitedGammaV2, GammaEpsAdd, GammaEpsClip)')
    parser.add_argument('--gamma_eps', type=float, default=None, help='Epsilon value for GammaEpsAdd and GammaEpsClip models')
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
    parser.add_argument('--classification_dropout_p', type=float, default=None, help='Override classification_dropout_p (float)')
    parser.add_argument('--classification_model', type=str, default=None, help='Override classification_model (e.g., ResNet, ViT, WideResNet, SE_ResNet)')
    parser.add_argument('--classifer_weights', type=str, default=None, help='Override classifer_weights (e.g., /path/to/weights.pth)')
    parser.add_argument('--save_name', type=str, default=None, help='Override save_name (e.g., wacv_exp1_resnet_nocss_noisp_leaves_canon)')
    return parser.parse_args()

def setup_dataset(config):
    """データセットのセットアップ"""
    transform_list = [
        T.RandomHorizontalFlip(p=0.5),
        T.RandomChoice([
            T.RandomRotation([0, 0]),
            T.RandomRotation([90, 90]),
            T.RandomRotation([180, 180]),
            T.RandomRotation([270, 270]),
        ])
    ]
    # 画像サイズは64x64を想定
    image_size = 64
    
    # Calculate scale based on size settings
    cutout_min_size = 8
    cutout_max_size = 32
    cutout_n_holes = 1
    cutout_prob = 1.0

    # Random size between min_size and max_size
    min_area_ratio = (cutout_min_size / image_size) ** 2
    max_area_ratio = (cutout_max_size / image_size) ** 2
    scale = (min_area_ratio, max_area_ratio)
    
    # n_holes分のCutoutを追加
    for _ in range(cutout_n_holes):
        transform_list.append(
            UniformCutout(
                p=cutout_prob,
                scale=scale,
                ratio=(1.0, 1.0),
                value='random',
                inplace=False
            )
        )
    
    train_transform = T.Compose(transform_list)

    data_dir = config.get('data_dir', os.environ.get('HFD100_DATA_DIR', './data'))
    train_dataset = HFD100_Dataset(dataset_name=config['dataset_name'], dataset_type='train', camera_name=config['camera_name'], transform=train_transform, data_dir=data_dir)
    val_dataset = HFD100_Dataset(dataset_name=config['dataset_name'], dataset_type='val', camera_name=config['camera_name'], data_dir=data_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)
    
    return train_loader, val_loader, train_dataset.num_classes()

def setup_models(config, num_classes, device):
    """モデルのセットアップ"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if config['camera_name'] == 'sRGB' or config['camera_name'] == 'gray':
        # css_weights_path = '/mnt/hdd1/youta/ws/optimize_css_hfd100/src/camera_parameters/cmf/cie1931_xyz_cmf.pt'
        css_weights_path = os.path.join(script_dir, 'camera_parameters/cmf/cie1931_xyz_cmf.pt')
    else:
        # css_weights_path = os.path.join('camera_parameters/css', f"cmf_{config['camera_name']}.pt")
        css_weights_path = os.path.join(script_dir, 'camera_parameters/css', f"cmf_{config['camera_name']}.pt")
    css_weights = torch.load(css_weights_path).float()
    # css_weights = torch.full_like(css_weights, 0.5).float()

    css_model = CustomCSS(init_weights=css_weights, in_channels=css_weights.shape[1], trainable=config['train_css']).to(device)
    
    if config['camera_name'] == 'sRGB' or config['camera_name'] == 'gray':
        # ccm_weights = torch.load(os.path.join('camera_parameters/ccm', "ccm_sRGB.pt")).float()
        ccm_weights_path = os.path.join(script_dir, 'camera_parameters/ccm', "ccm_sRGB.pt")
    else:
        # ccm_weights_path = os.path.join('camera_parameters/ccm', f"ccm_{config['camera_name']}.pt")
        ccm_weights_path = os.path.join(script_dir, 'camera_parameters/ccm', f"ccm_{config['camera_name']}.pt")
    ccm_weights = torch.load(ccm_weights_path).float()
    # ccm_weights = torch.randn(3, 3).float()
    ccm_model = ColorCorrectionMatrix(init_ccm=ccm_weights, trainable=config['train_ccm']).to(device)
    
    print('gradient_clipping: ', config['gradient_clipping'])
    if config['gamma_model_type'] == 'GradientLimitedGammaV1':
        gamma_model = GradientLimitedGammaV1(init_gamma=1/2.2, grad_th=config['gradient_clipping'], trainable=config['train_gamma']).to(device)
    elif config['gamma_model_type'] == 'GradientLimitedGammaV2':
        gamma_model = GradientLimitedGammaV2(init_gamma=1/2.2, grad_th=config['gradient_clipping'], trainable=config['train_gamma']).to(device)
    elif config['gamma_model_type'] == 'GammaEpsAdd':
        print('gamma_model_type: ', config['gamma_model_type'])
        print('gamma_eps: ', config['gamma_eps'])
        gamma_model = GammaEpsAdd(init_gamma=1/2.2, trainable=config['train_gamma'], eps=float(config['gamma_eps'])).to(device)
    elif config['gamma_model_type'] == 'GammaEpsClip':
        print('gamma_model_type: ', config['gamma_model_type'])
        print('gamma_eps: ', config['gamma_eps'])
        gamma_model = GammaEpsClip(init_gamma=1/2.2, trainable=config['train_gamma'], eps=float(config['gamma_eps'])).to(device)
    
    dropout_p = config.get('classification_dropout_p', 0.5)
    if config['classification_model'] == 'ResNet':
        classification_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
        num_ftrs = classification_model.fc.in_features
        classification_model.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(num_ftrs, num_classes)).to(device)
        # if config['dataset_name'] == 'HFD100_Flower':
        #     config['classifer_weights'] = '/mnt/hdd1/youta/ws/optimize_css_hfd100/src/runs/train/resnet1k_pretrain_flower_cam_sRGB_css_false_ccm_false_gamma_false/epoch_299/classification_model.pth'
        # elif config['dataset_name'] == 'HFD100_Leaves':
        #     config['classifer_weights'] = '/mnt/hdd1/youta/ws/optimize_css_hfd100/src/runs/train/resnet1k_pretrain_leaves_cam_sRGB_css_false_ccm_false_gamma_false/epoch_299/classification_model.pth'

        # if config['classifer_weights'] is not None:
        #     pretrained_state_dict = torch.load(config['classifer_weights'], map_location=device)
            
        #     current_model_state_dict = classification_model.state_dict()
            
        #     new_state_dict = {}
        #     for key, param in pretrained_state_dict.items():
        #         if key == 'fc.weight':
        #             new_state_dict['fc.1.weight'] = param
        #         elif key == 'fc.bias':
        #             new_state_dict['fc.1.bias'] = param
        #         elif key in current_model_state_dict:
        #             new_state_dict[key] = param
            
        #     current_model_state_dict.update(new_state_dict)
        #     classification_model.load_state_dict(current_model_state_dict)
    elif config['classification_model'] == 'ViT':
        classification_model = timm.create_model('vit_small_patch16_224.augreg_in21k', pretrained=True, img_size=64, num_classes=num_classes).to(device)
        # if config['dataset_name'] == 'HFD100_Flower':
        #     config['classifer_weights'] = '/mnt/hdd1/youta/ws/optimize_css_hfd100/src/runs/train/vits21k_pretrain_flower_cam_sRGB_css_false_ccm_false_gamma_false/epoch_299/classification_model.pth'
        # elif config['dataset_name'] == 'HFD100_Leaves':
        #     config['classifer_weights'] = '/mnt/hdd1/youta/ws/optimize_css_hfd100/src/runs/train/vit21k_pretrain_leaves_cam_sRGB_css_false_ccm_false_gamma_false/epoch_299/classification_model.pth'
        # pretrained_state_dict = torch.load(config['classifer_weights'], map_location=device)
        # classification_model.load_state_dict(pretrained_state_dict)
    elif config['classification_model'] == 'WideResNet':
        classification_model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device)
        num_ftrs = classification_model.fc.in_features
        classification_model.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(num_ftrs, num_classes)).to(device)
        # pretrained_state_dict = torch.load(config['classifer_weights'], map_location=device)
        # classification_model.load_state_dict(pretrained_state_dict)
    elif config['classification_model'] == 'SE_ResNet':
        classification_model = timm.create_model('seresnet50', pretrained=True).to(device)
        num_ftrs = classification_model.fc.in_features
        classification_model.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(num_ftrs, num_classes)).to(device)
        # if config['dataset_name'] == 'HFD100_Flower':
        #     config['classifer_weights'] = '/mnt/hdd1/youta/ws/optimize_css_hfd100/src/runs/train/seresnet50_pretrain_flower_cam_sRGB_css_false_ccm_false_gamma_false/epoch_299/classification_model.pth'
        # elif config['dataset_name'] == 'HFD100_Leaves':
        #     config['classifer_weights'] = '/mnt/hdd1/youta/ws/optimize_css_hfd100/src/runs/train/seresnet50_pretrain_leaves_cam_sRGB_css_false_ccm_false_gamma_false/epoch_299/classification_model.pth'
        # pretrained_state_dict = torch.load(config['classifer_weights'], map_location=device)
        # classification_model.load_state_dict(pretrained_state_dict)


    return css_model, gamma_model, ccm_model, classification_model

def setup_optimizer(config, css_model, gamma_model, ccm_model, classification_model):
    """オプティマイザのセットアップ"""
    optimizer_param_groups = []
    
    if config['train_css']:
        optimizer_param_groups.append({'params': css_model.parameters(), 'lr': config['css_lr'], 'weight_decay': 0.0})
    
    if config['train_gamma']:
        optimizer_param_groups.append({'params': gamma_model.parameters(), 'lr': config['gamma_lr'], 'weight_decay': 0.0})
    
    if config['train_ccm']:
        optimizer_param_groups.append({'params': ccm_model.parameters(), 'lr': config['ccm_lr'], 'weight_decay': 0.0})
    
    if config['train_classification']:
        weight_decay = 0.01
        optimizer_param_groups.append({'params': classification_model.parameters(), 'lr': config['classification_lr'], 'weight_decay': weight_decay})

    if not optimizer_param_groups:
        raise ValueError('最適化するパラメータがありません')

    # optimizer = optim.Adam(optimizer_param_groups)
    optimizer = optim.AdamW(optimizer_param_groups)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step_size'], gamma=config['lr_gamma'])
    
    return optimizer, lr_scheduler

def init_wandb(config):
    """WandBの初期化"""
    if config.get('wandb_project') is None:
        return None
    
    base_run_name_for_init = config.get('save_name') 
    if not base_run_name_for_init:
        base_run_name_for_init = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    return wandb.init(
        project=config['wandb_project'],
        entity=config.get('wandb_entity'),
        config=config,
        name=base_run_name_for_init,
        reinit=True 
    )

def main():
    args = parse_args()
    config = load_config(args.config)

    if args.camera_name is not None:
        config['camera_name'] = args.camera_name
    if args.train_css is not None:
        config['train_css'] = args.train_css
    if args.train_ccm is not None:
        config['train_ccm'] = args.train_ccm
    if args.css_lr is not None:
        config['css_lr'] = args.css_lr
    if args.ccm_lr is not None:
        config['ccm_lr'] = args.ccm_lr
    if args.gamma_lr is not None:
        config['gamma_lr'] = args.gamma_lr
    if args.gamma_model_type is not None:
        config['gamma_model_type'] = args.gamma_model_type
    if args.gamma_eps is not None:
        config['gamma_eps'] = args.gamma_eps
    if args.dataset_name is not None:
        config['dataset_name'] = args.dataset_name
    if args.train_gamma is not None:
        config['train_gamma'] = args.train_gamma
    if args.train_classification is not None:
        config['train_classification'] = args.train_classification
    if args.classification_lr is not None:
        config['classification_lr'] = args.classification_lr
    if args.classification_dropout_p is not None:
        config['classification_dropout_p'] = args.classification_dropout_p
    if args.classification_model is not None:
        config['classification_model'] = args.classification_model
    if args.classifer_weights is not None:
        config['classifer_weights'] = args.classifer_weights
    if args.save_name is not None:
        config['save_name'] = args.save_name

    if args.log_scale_input is not None:
        config['log_scale_input'] = args.log_scale_input
        gamma_base = 1.0 / 2.2
        input_val = max(args.log_scale_input, 0)
        calculated_grad_clip = gamma_base * math.pow(input_val, gamma_base - 1.0)
        config['gradient_clipping'] = calculated_grad_clip
        print(f"Calculated gradient_clipping: {config['gradient_clipping']} from log_scale_input: {args.log_scale_input}")
    
    
    wandb_run = init_wandb(config)
    
    if wandb_run is not None:
        config.update(wandb.config) # Sync config with W&B sweep parameters

        current_wandb_name = wandb_run.name 

        run_name_prefix_parts = []
        
        param_map = {
            'camera_name': 'cam',
            'train_css': 'css',
            'train_ccm': 'ccm',
            'train_gamma': 'gamma',
            'log_scale_input': 'lsi'
        }

        for config_key, prefix_key in param_map.items():
            value = config.get(config_key)
            if value is not None:
                if isinstance(value, float): 
                    value_str = str(value).replace('.', '_')
                elif isinstance(value, bool):
                    value_str = str(value).lower()
                else: 
                    value_str = str(value)
                run_name_prefix_parts.append(f"{prefix_key}_{value_str}")
        
        if run_name_prefix_parts:
            prefix = "_".join(run_name_prefix_parts)
            wandb_run.name = f"{prefix}_{current_wandb_name}"


    base_save_name_from_config = config.get('save_name') 
    if not base_save_name_from_config: 
        effective_base_save_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        effective_base_save_name = base_save_name_from_config
    
    save_name_suffix_parts = []
    
    # param_map_for_save = {
    #     'camera_name': 'cam',
    #     'train_css': 'css',
    #     'train_ccm': 'ccm',
    #     'train_gamma': 'gamma',
    #     'log_scale_input': 'lsi'
    # }
    param_map_for_save = {
        'log_scale_input': 'lsi'
    }

    for config_key, suffix_key in param_map_for_save.items():
        value = config.get(config_key)
        if value is not None:
            if isinstance(value, float):
                value_str = str(value).replace('.', '_')
            elif isinstance(value, bool):
                value_str = str(value).lower()
            else:
                value_str = str(value)
            save_name_suffix_parts.append(f"{suffix_key}_{value_str}")
            
    if save_name_suffix_parts:
        suffix = "_".join(save_name_suffix_parts)
        save_name = f"{effective_base_save_name}_{suffix}"
    else:
        save_name = effective_base_save_name
    
    save_dir = os.path.join('runs/train', save_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    train_loader, val_loader, num_classes = setup_dataset(config)
    device = torch.device(config['device'])
    css_model, gamma_model, ccm_model, classification_model = setup_models(config, num_classes, device)
    
    logger = TrainLogger(save_dir=save_dir, use_wandb=wandb_run is not None)
    
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

        # Check for gradient explosion via loss
        current_train_loss = train_metrics.get('train_loss') # Use .get for safety
        if current_train_loss is not None and not torch.isfinite(torch.tensor(current_train_loss)):
            explosion_msg = f"勾配爆発を検知しました！ epoch: {epoch}, train_loss: {current_train_loss}, gradient_clipping: {config.get('gradient_clipping')}"
            print(explosion_msg)
            logger.log_message(explosion_msg)
            if wandb_run is not None:
                wandb.log({
                    "gradient_exploded": 1,
                    "epoch_at_explosion": epoch,
                    "train_loss_at_explosion": current_train_loss,
                    "gradient_clipping_at_explosion": config.get('gradient_clipping')
                })
                # Mark run as crashed and exit to allow sweep to continue with next trial
                wandb.finish(exit_code=1) # Ensure WandB run is properly terminated as crashed
            sys.exit(1) # Exit the script, so sweep agent knows this run failed and proceeds
        
        val_metrics = model_wrapper.evaluate(val_loader)
        logger.log_val_loss(val_metrics)
        
        logger.log_epoch_metrics(train_metrics, val_metrics, epoch)
        logger.print_log(epoch, train_metrics, val_metrics)
        
        should_save = False
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            should_save = True
            print(f'最良モデルを保存しました (val_loss: {val_metrics["val_loss"]:.4f})')
            logger.save_checkpoint(
                epoch=epoch, 
                css_model=css_model, 
                gamma_model=gamma_model,
                ccm_model=ccm_model,
                classification_model=classification_model,
                is_best=True
            )
        
        if ((epoch + 1) % config['save_interval'] == 0) or (epoch == config['epochs'] - 1):
            should_save = True
        
        if should_save and val_metrics['val_loss'] >= best_val_loss:
            logger.save_checkpoint(
                epoch=epoch, 
                css_model=css_model, 
                gamma_model=gamma_model,
                ccm_model=ccm_model,
                classification_model=classification_model
            )
    
    if wandb_run is not None:
        wandb.finish()

if __name__ == '__main__':
    main()