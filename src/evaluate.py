import os
import json
import yaml
import torch
import timm
import random
import numpy as np
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from torchvision import models
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from hsi_dataset import HFD100_Dataset
from model_wrapper import ModelWrapper
from models.custom_css import CustomCSS
from models.custom_isp import GradientLimitedGammaV1, GradientLimitedGammaV2, ColorCorrectionMatrix, GammaEpsAdd, GammaEpsClip
from utils.visualization import save_classification_samples


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
    return config


def load_eval_config(eval_config_path):
    """評価設定ファイル (eval_config.yaml) を読み込む"""
    if not os.path.exists(eval_config_path):
        print(f"警告: 評価設定ファイル {eval_config_path} が見つかりません。全てのモデルの重みをロードします。")
        return {
            'load_weights': {
                'css': True,
                'gamma': True,
                'ccm': True,
                'base': True
            }
        }
    with open(eval_config_path, 'r') as f:
        eval_cfg = yaml.safe_load(f)
    
    # load_weights セクションのデフォルト値を設定
    if 'load_weights' not in eval_cfg:
        eval_cfg['load_weights'] = {}
    
    default_load_flags = {
        'css': True,
        'gamma': True,
        'ccm': True,
        'base': True
    }
    for component, default_flag in default_load_flags.items():
        if component not in eval_cfg['load_weights']:
            eval_cfg['load_weights'][component] = default_flag
            print(f"情報: eval_config.yaml の load_weights に {component} が未指定のため、デフォルト値 ({default_flag}) を使用します。")
            
    return eval_cfg


def parse_args():
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(description='Evaluate CSS model')
    parser.add_argument('--config', type=str, required=True, help='学習時の設定ファイルのパス')
    parser.add_argument('--eval_config', type=str, required=True, help='評価時の設定ファイル (eval_config.yaml) のパス')
    return parser.parse_args()


def setup_dataset(config, batch_size, num_workers):
    """テストデータセットのセットアップ"""
    
    if 'dataset_name' not in config:
        raise ValueError("設定ファイル (config.yaml) に 'dataset_name' フィールドが必要です。")
        
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
    """モデルのセットアップと重みの読み込み"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # CSSモデルのセットアップ
    if train_config['camera_name'] == 'sRGB' or train_config['camera_name'] == 'gray':
        init_css_weights_path = os.path.join(script_dir, 'camera_parameters/cmf', "cie1931_xyz_cmf.pt")
    else:
        init_css_weights_path = os.path.join(script_dir, 'camera_parameters/css', f"cmf_{train_config['camera_name']}.pt")
    
    if not os.path.exists(init_css_weights_path):
        raise FileNotFoundError(f"デフォルトCMFファイルが見つかりません: {init_css_weights_path}")
    init_css_weights = torch.load(init_css_weights_path).float()
    css_model = CustomCSS(init_weights=init_css_weights, trainable=False).to(device)

    if eval_cfg.get('load_weights', {}).get('css', True):
        weights_path = os.path.join(train_run_dir_path, 'epoch_99', 'css_model.pth')
        # weights_path = os.path.join(train_run_dir_path, 'epoch_299', 'css_model.pth')
        # weights_path = os.path.join(train_run_dir_path, 'best_model', 'css_model.pth')
        print(f"CSSモデルの重みをロードします: {weights_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"CSSモデルの重みファイルが見つかりません: {weights_path}")
        css_model.load_state_dict(torch.load(weights_path))
    else:
        print("CSSモデル: 学習済み重みをロードせず、デフォルトCMFを使用します。")
    css_model.eval()
    
    # ガンマモデルのセットアップ
    if train_config['gamma_model_type'] == 'GradientLimitedGammaV1':
        gamma_model = GradientLimitedGammaV1(init_gamma=1/2.2, grad_th=train_config['gradient_clipping'], trainable=train_config['train_gamma']).to(device)
    elif train_config['gamma_model_type'] == 'GradientLimitedGammaV2':
        gamma_model = GradientLimitedGammaV2(init_gamma=1/2.2, grad_th=train_config['gradient_clipping'], trainable=train_config['train_gamma']).to(device)
    elif train_config['gamma_model_type'] == 'GammaEpsAdd':
        gamma_model = GammaEpsAdd(init_gamma=1/2.2, trainable=train_config['train_gamma'], eps=float(train_config['gamma_eps'])).to(device)
    elif train_config['gamma_model_type'] == 'GammaEpsClip':
        gamma_model = GammaEpsClip(init_gamma=1/2.2, trainable=train_config['train_gamma'], eps=float(train_config['gamma_eps'])).to(device)
    if eval_cfg.get('load_weights', {}).get('gamma', True):
        weights_path = os.path.join(train_run_dir_path, 'epoch_99', 'gamma_model.pth')
        # weights_path = os.path.join(train_run_dir_path, 'epoch_299', 'gamma_model.pth')
        # weights_path = os.path.join(train_run_dir_path, 'epoch_289', 'gamma_model.pth')
        # weights_path = os.path.join(train_run_dir_path, 'best_model', 'gamma_model.pth')
        print(f"Gammaモデルの重みをロードします: {weights_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Gammaモデルの重みファイルが見つかりません: {weights_path}")
        gamma_model.load_state_dict(torch.load(weights_path))
        print("gamma_model.state_dict(): ", gamma_model.state_dict())
    else:
        print("Gammaモデル: 学習済み重みをロードせず、デフォルトのガンマ補正を使用します。")
    gamma_model.eval()
    
    # CCMモデルのセットアップ
    if train_config['camera_name'] == 'sRGB' or train_config['camera_name'] == 'gray':
        init_ccm_weights_path = os.path.join(script_dir, 'camera_parameters/ccm', "ccm_sRGB.pt")
    else:
        init_ccm_weights_path = os.path.join(script_dir, 'camera_parameters/ccm', f"ccm_{train_config['camera_name']}.pt")
    
    if os.path.exists(init_ccm_weights_path):
        init_ccm_weights = torch.load(init_ccm_weights_path).float()
    else:
        if train_config['camera_name'] != 'sRGB':
            print(f"警告: デフォルトCCMファイル {init_ccm_weights_path} が見つかりません。{train_config['camera_name']} 用に単位行列を使用します。")
        init_ccm_weights = torch.eye(3).float() # フォールバック
    print(init_ccm_weights)
    ccm_model = ColorCorrectionMatrix(init_ccm=init_ccm_weights, trainable=False).to(device)
    if eval_cfg.get('load_weights', {}).get('ccm', True):
        weights_path = os.path.join(train_run_dir_path, 'epoch_99', 'ccm_model.pth')
        # weights_path = os.path.join(train_run_dir_path, 'epoch_299', 'ccm_model.pth')
        # weights_path = os.path.join(train_run_dir_path, 'best_model', 'ccm_model.pth')
        print(f"CCMモデルの重みをロードします: {weights_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"CCMモデルの重みファイルが見つかりません: {weights_path}")
        ccm_model.load_state_dict(torch.load(weights_path))
        print(ccm_model.state_dict())
    else:
        print(f"CCMモデル: 学習済み重みをロードせず、{train_config['camera_name']} のデフォルトCCMを使用します。")
    ccm_model.eval()
    
    
    if eval_cfg.get('load_weights', {}).get('base', True):
        weights_path = os.path.join(train_run_dir_path, 'epoch_99', 'classification_model.pth')
        # weights_path = os.path.join(train_run_dir_path, 'epoch_299', 'classification_model.pth')
        # weights_path = os.path.join(train_run_dir_path, 'best_model', 'classification_model.pth')
        print(f"ベースモデルの学習済み重みをロードします: {weights_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"ベースモデルの重みファイルが見つかりません: {weights_path}")

        # --- モデルの選択 ---
        model_name = train_config.get('classification_model', 'ResNet')
        print(f"分類モデルをロードします: {model_name}")

        base_model = None
        if model_name == 'ResNet':
            base_model = models.resnet18(weights=None)
            num_ftrs = base_model.fc.in_features
            base_model.fc = nn.Linear(num_ftrs, num_classes)
            loaded_state_dict = torch.load(weights_path, map_location=device)
        
            # 学習時のモデル(fcがSequential)と評価時のモデル(fcがLinear)のキー名の違いを吸収する
            key_mappings = {
                'fc.1.weight': 'fc.weight',
                'fc.1.bias': 'fc.bias',
                'heads.head.1.weight': 'heads.head.weight',
                'heads.head.1.bias': 'heads.head.bias',
            }
            
            new_state_dict = loaded_state_dict.copy()
            remap_happened = False
            for saved_key, new_key in key_mappings.items():
                if saved_key in new_state_dict:
                    print(f"  キーをリマップします: {saved_key} -> {new_key}")
                    new_state_dict[new_key] = new_state_dict.pop(saved_key)
                    remap_happened = True
            
            if remap_happened:
                print("state_dict のキーリマップを適用しました。")

            # Dropout層など、学習時にのみ存在する層の重みは無視して読み込む
            base_model.load_state_dict(new_state_dict, strict=False)
            print("ベースモデルに重みを正常にロードしました（strict=False）。")

        elif model_name == 'ViT':
            base_model = timm.create_model('vit_small_patch16_224.augreg_in21k', pretrained=True, img_size=64, num_classes=num_classes).to(device)
            pretrained_state_dict = torch.load(weights_path, map_location=device)
            base_model.load_state_dict(pretrained_state_dict)
        elif model_name == 'WideResNet':
            base_model = models.wide_resnet50_2(weights=None)
            num_ftrs = base_model.fc.in_features
            base_model.fc = nn.Linear(num_ftrs, num_classes)
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
            remap_happened = False
            for saved_key, new_key in key_mappings.items():
                if saved_key in new_state_dict:
                    print(f"  キーをリマップします: {saved_key} -> {new_key}")
                    new_state_dict[new_key] = new_state_dict.pop(saved_key)
                    remap_happened = True
            
            if remap_happened:
                print("state_dict のキーリマップを適用しました。")

            base_model.load_state_dict(new_state_dict, strict=False)
        else:
            raise ValueError(f"サポートされていないモデルタイプです: {model_name}")

        base_model = base_model.to(device)

        # --- 重みのロードとキーの修正 ---
        
        base_model.eval()
    
    return css_model, gamma_model, ccm_model, base_model


def custom_evaluate(model_wrapper, test_loader, output_dir, num_visualization_samples=20):
    """拡張された評価メソッド"""
    css_model = model_wrapper.css_model
    gamma_model = model_wrapper.gamma_model
    ccm_model = model_wrapper.ccm_model
    base_model = model_wrapper.classification_model
    device = model_wrapper.device
    
    css_model.eval()
    gamma_model.eval()
    ccm_model.eval()
    base_model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    num_batches = len(test_loader)
    
    # 混同行列用の変数
    all_predictions = []
    all_targets = []
    
    # 可視化サンプルの保存用ディレクトリ
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 可視化用のインデックスをランダムに選択
    total_samples = num_batches * test_loader.batch_size
    vis_indices = random.sample(range(total_samples), min(num_visualization_samples, total_samples))
    
    sample_count = 0
    sample_images = []
    sample_predictions = []
    sample_targets = []
    
    with torch.no_grad():
        for i, (inputs_hsi, target) in enumerate(test_loader):
            print(f'\rEvaluation: {i+1}/{num_batches}', end='')
            
            # HSIをRGBに変換
            inputs_hsi = inputs_hsi.to(device)
            rgb_images = css_model(inputs_hsi)
            rgb_images = ccm_model(rgb_images)
            rgb_images = rgb_images.clamp(min=0)
            rgb_images = gamma_model(rgb_images)
            
            # 分類
            outputs = base_model(rgb_images)
            loss = model_wrapper.criterion(outputs, target.to(device))
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target.to(device)).sum().item()
            
            # 混同行列用のデータ収集
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.numpy())
            
            # 可視化用データの収集
            for j in range(len(rgb_images)):
                global_idx = i * test_loader.batch_size + j
                if global_idx in vis_indices:
                    sample_images.append(rgb_images[j])
                    sample_predictions.append(predicted[j].item())
                    sample_targets.append(target[j].item())
    
    print("\n評価完了")
    
    # 混同行列の作成と保存
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 分類レポートの保存
    report = classification_report(all_targets, all_predictions, output_dict=True)
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # 可視化サンプルの保存
    if sample_images:
        save_classification_samples(
            vis_dir,
            sample_images,
            sample_predictions,
            sample_targets,
            num_samples=len(sample_images)
        )
    
    # AA (Average Accuracy) の計算
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    aa = np.nanmean(class_accuracy)  # クラスのサンプルがない場合にnanになる可能性があるのでnanmeanを使用

    metrics = {
        'loss': total_loss / num_batches,
        'accuracy': 100. * correct / total,
        'average_accuracy': 100. * aa
    }
    
    return metrics


def save_eval_run_config(args, train_cfg, eval_cfg, output_dir):
    """評価実行時の設定（学習設定、評価設定、引数）を保存する"""
    full_eval_config = {
        'command_line_args': vars(args),
        'training_config': train_cfg,
        'evaluation_module_config': eval_cfg 
    }
    
    with open(os.path.join(output_dir, 'eval_run_config.yaml'), 'w') as f:
        yaml.dump(full_eval_config, f, default_flow_style=False, sort_keys=False)


def main():
    """メイン関数"""
    args = parse_args()
    train_config = load_config(args.config)
    eval_cfg = load_eval_config(args.eval_config)
    
    device = torch.device(train_config['device'])
    
    # output_dir は eval_cfg から取得し、'runs/eval' を付加する
    output_dir_resolved = os.path.join('runs/eval', eval_cfg['run_parameters']['output_dir'])
    os.makedirs(output_dir_resolved, exist_ok=True)
    
    save_eval_run_config(args, train_config, eval_cfg, output_dir_resolved)
    
    # batch_size, num_workers は eval_cfg から取得
    test_loader, num_classes = setup_dataset(
        train_config, 
        eval_cfg['run_parameters']['batch_size'], 
        eval_cfg['run_parameters']['num_workers']
    )
    
    # train_run_dir は eval_cfg から取得し、setup_models に渡す
    css_model, gamma_model, ccm_model, base_model = setup_models(
        train_config, 
        eval_cfg['run_parameters']['train_run_dir'], # args.train_run_dir から変更
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
        optimizer=None,  # 評価時は不要
        lr_schedule=None,  # 評価時は不要
        device=device,
        logger=None,
        camera_name=train_config['camera_name']
    )
    
    # 評価の実行
    print(f"評価を開始します: {output_dir_resolved}")
    metrics = custom_evaluate(
        model_wrapper=model_wrapper,
        test_loader=test_loader,
        output_dir=output_dir_resolved,
        num_visualization_samples=eval_cfg['run_parameters']['num_visualization_samples'] # args から変更
    )
    
    # 結果の保存
    with open(os.path.join(output_dir_resolved, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\n評価が完了しました。結果は {output_dir_resolved} に保存されています。")
    print(f"損失: {metrics['loss']:.4f}")
    print(f"精度 (OA): {metrics['accuracy']:.2f}%")
    print(f"平均精度 (AA): {metrics['average_accuracy']:.2f}%")


if __name__ == '__main__':
    main()
