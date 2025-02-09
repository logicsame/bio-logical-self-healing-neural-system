import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from bioneural.core.biololgicallayer import BioLogicalNeuron
import shutil



def create_results_directory():
    results_dir = 'cifar10_results_full_architecture'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'logs'), exist_ok=True)
    return results_dir


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBioBlock(nn.Module):
    """Enhanced Convolutional block with SE attention"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = weight_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(0.15)  # Increased dropout
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.se(x)
        x = self.activation(x)
        return self.dropout(x)

class ResidualBioBlock(nn.Module):
    """Enhanced Residual block with SE attention"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBioBlock(channels, channels)
        self.conv2 = ConvBioBlock(channels, channels)
        self.se = SEBlock(channels)
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        residual = x
        out = self.conv2(self.conv1(x))
        out = self.se(out)
        out = self.dropout(out)
        return out + residual
    
    
class ModernBionetwork(nn.Module):
    def __init__(self, num_classes=10,enable_monitoring=False,disable_monitoring=False,results_dir = 'cifar10_results_full_architecture'):
        super().__init__()
        monitoring_state = enable_monitoring and not disable_monitoring
        log_base_path = os.path.join(results_dir, 'logs')
        
        # Feature extractor remains the same
        self.features = nn.Sequential(
            ConvBioBlock(3, 64),
            ResidualBioBlock(64),
            ResidualBioBlock(64),
            nn.MaxPool2d(2),
            ConvBioBlock(64, 128),
            ResidualBioBlock(128),
            ResidualBioBlock(128),
            nn.MaxPool2d(2),
            ConvBioBlock(128, 256),
            ResidualBioBlock(256),
            ResidualBioBlock(256),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Only 2 bio layers
        self.bio_layers = nn.ModuleList([
            BioLogicalNeuron(4096, 1024, repair_threshold=0.8, log_file='Cifar10_neuron_1', repair_intensity=0.015, plasticity_rate=0.0015, enable_monitoring=monitoring_state,log_file = os.path.join(log_base_path, 'bio_layer_1.log')),
            BioLogicalNeuron(1024, 512, repair_threshold=0.8, log_file='Cifar10_neuron_2', repair_intensity=0.015, plasticity_rate=0.0015, enable_monitoring=monitoring_state,log_file = os.path.join(log_base_path, 'bio_layer_2.log'))
        ])
        
        # Classifier remains the same
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        health_reports = []
        for bio_layer in self.bio_layers:
            x, health_report = bio_layer(x)
            health_reports.append(health_report)
        
        x = self.classifier(x)
        return x
    


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        """
        Early stopping to stop the training when the loss does not improve after
        certain epochs.
        
        Args:
            patience (int): Number of epochs to wait before stopping after loss stops improving.
            min_delta (float): Minimum change in loss to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


import os
import torch
import numpy as np
import wandb
from datetime import datetime
import json
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_recall_fscore_support,
    roc_auc_score
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class CIFAR10Trainer:
    def __init__(self, n_splits=2, seed=42, wandb_logging=False,enable_monitoring=False,disable_monitoring=False):
        self.n_splits = n_splits
        self.seed = seed
        self.wandb_logging = wandb_logging
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_smoothing = 0.01
        self.gradient_clip = 0.5
        self.enable_monitoring = enable_monitoring
        self.disable_monitoring = disable_monitoring
        # Data augmentation and normalization for training
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Just normalization for validation/test
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Prepare dataset
        self.dataset = self._prepare_dataset()
        self.results_dir = create_results_directory()
    def _prepare_dataset(self):
        # Load full training dataset
        return datasets.CIFAR10(root='./data', train=True, download=True)
    
    def _create_data_splits(self):
        labels = np.array([self.dataset[i][1] for i in range(len(self.dataset))])
        indices = np.arange(len(labels))
        
        # Create stratified folds
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        splits = []
        
        for train_idx, temp_idx in skf.split(indices, labels):
            # Further split temp_idx into validation and test
            temp_labels = labels[temp_idx]
            sss2 = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.seed)
            val_idx, test_idx = next(sss2.split(temp_idx, temp_labels))
            
            val_idx = temp_idx[val_idx]
            test_idx = temp_idx[test_idx]
            
            splits.append((train_idx, val_idx, test_idx))
        
        return splits

    def train_and_evaluate(self):
        results = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'auc': []}
    
        if self.wandb_logging:
            wandb.init(project="CIFAR10_BioNetwork",
                      config={"n_splits": self.n_splits, "seed": self.seed})

        cv_progress = tqdm(list(self._create_data_splits()), desc="Cross-Validation Folds")
        
        for fold, (train_idx, val_idx, test_idx) in enumerate(cv_progress):
            # Create dataset subsets with appropriate transforms
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)
            test_subset = Subset(self.dataset, test_idx)
            
            # Create custom datasets with appropriate transforms
            train_dataset = TransformDataset(train_subset, self.train_transform)
            val_dataset = TransformDataset(val_subset, self.test_transform)
            test_dataset = TransformDataset(test_subset, self.test_transform)

            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=128, num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=128, num_workers=4, pin_memory=True)

            # Initialize model and training components
            model = ModernBionetwork(num_classes=10,enable_monitoring=self.enable_monitoring,disable_monitoring=self.disable_monitoring).to(self.device)
            criterion, optimizer, scheduler = self._get_training_components(model, len(train_loader))
            scaler = torch.cuda.amp.GradScaler()
            early_stopping = EarlyStopping(patience=25, min_delta=0.0005)

            
            # Training loop
            best_val_acc = 0
            best_val_loss = float('inf')
            epoch_progress = tqdm(range(200), desc=f"Fold {fold+1} Training", leave=False)

            for epoch in epoch_progress:
                # Training phase with mixup
                train_metrics = self._train_epoch(model, train_loader, optimizer, criterion, scheduler, scaler)
                
                # Validation phase
                val_metrics = self._validate(model, val_loader, criterion)

                # Update progress bar
                epoch_progress.set_postfix({
                    'Train Loss': f'{train_metrics["loss"]:.4f}',
                    'Val Acc': f'{val_metrics["accuracy"]:.4f}'
                })

                # Wandb logging
                if self.wandb_logging:
                    wandb.log({
                        f"Fold_{fold+1}/Train_Loss": train_metrics["loss"],
                        f"Fold_{fold+1}/Val_Loss": val_metrics["loss"],
                        f"Fold_{fold+1}/Val_Accuracy": val_metrics["accuracy"]
                    })

                # Early stopping check
                if early_stopping(val_metrics['loss']):
                    print(f"Early stopping triggered in fold {fold+1} at epoch {epoch}")
                    break

                # Save best model
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    model_path = os.path.join(self.results_dir, 'models', f'best_model_fold{fold}.pth')
                    torch.save(model.state_dict(), model_path)

            # Test evaluation
            model.load_state_dict(torch.load(model_path))
            test_metrics = self._evaluate(model, test_loader, criterion)

            # Store results
            for metric in results:
                results[metric].append(test_metrics[metric])

            # Log test metrics
            if self.wandb_logging:
                for metric, value in test_metrics.items():
                    wandb.log({f"Fold_{fold+1}/Test_{metric}": value})

        # Compute final statistics
        final_results = {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            } for metric, values in results.items()
        }

        # Save results
        with open('cifar10_results.json', 'w') as f:
            json.dump(final_results, f, indent=4)


        #Move bio_vis folder if it exists
        if self.enable_monitoring:
            bio_vis_src = 'bio_vis'
            if os.path.exists(bio_vis_src):
                bio_vis_dest = os.path.join(self.results_dir, 'bio_vis')
                if os.path.exists(bio_vis_dest):
                    shutil.rmtree(bio_vis_dest)
                shutil.move(bio_vis_src, bio_vis_dest)


        return final_results

    def _get_training_components(self, model, steps_per_epoch):
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0001,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
    
        # Calculate total steps correctly
        total_steps = steps_per_epoch * 100  # 100 epochs
    
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.001,
            epochs=200,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=100.0,
            total_steps=total_steps  # Explicitly set total steps
        )
    
        return criterion, optimizer, scheduler

    def _train_epoch(self, model, loader, optimizer, criterion, scheduler, scaler):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
    
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs, targets_a, targets_b, lam = self._mixup_data(inputs, targets, alpha=0.1)
        
            optimizer.zero_grad(set_to_none=True)
        
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = self._mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
                if torch.isnan(loss):
                    print(f"NaN loss detected at batch {batch_idx}")
                    continue
        
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        
            # Only step the scheduler if we haven't exceeded the total steps
            if scheduler.total_steps > scheduler._step_count:
                scheduler.step()
        
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a).float() + 
                       (1 - lam) * predicted.eq(targets_b).float()).sum().item()
        
            # Gradient norm monitoring
            if batch_idx % 100 == 0:
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f'Gradient norm: {total_norm}')

        return {
            'loss': total_loss / len(loader),
            'accuracy': 100. * correct / total
        }

    def _mixup_data(self, x, y, alpha=0.2):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def _mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def _validate(self, model, loader, criterion):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total
        return {
            'loss': total_loss / len(loader),
            'accuracy': accuracy
        }

    def _evaluate(self, model, loader, criterion):
        model.eval()
        total_loss = 0
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        accuracy = np.mean(y_true == y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # For multi-class ROC AUC, using one-vs-rest approach
        y_true_binary = np.eye(10)[y_true]
        y_pred_binary = np.eye(10)[y_pred]
        auc = roc_auc_score(y_true_binary, y_pred_binary, multi_class='ovr')
        
        return {
            'loss': total_loss / len(loader),
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'auc': auc * 100
        }

class TransformDataset:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.dataset)

import argparse
def main():
    parser = argparse.ArgumentParser(description='Publication Trainer with Biological Neural Network')
    
    monitoring_group = parser.add_mutually_exclusive_group()
    monitoring_group.add_argument('--enable-monitoring', action='store_true', 
                        help='Enable monitoring for biological layers')
    monitoring_group.add_argument('--disable-monitoring', action='store_true', 
                        help='Explicitly disable monitoring for biological layers')
    
  
    parser.add_argument('--n-splits', type=int, default=10, 
                        help='Number of cross-validation splits')
    parser.add_argument('--wandb', action='store_true', 
                        help='Enable Weights & Biases logging')
    
    args = parser.parse_args()
    
    trainer = CIFAR10Trainer(
        n_splits=args.n_splits, 
        wandb_logging=args.wandb,
        enable_monitoring=args.enable_monitoring,
        disable_monitoring=args.disable_monitoring
    )
    
    results = trainer.train_and_evaluate()
    print("Publication Results:", results)

if __name__ == "__main__":
    main()