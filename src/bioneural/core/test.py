import os
import torch
import numpy as np
import wandb
import datetime
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool, JumpingKnowledge, BatchNorm
from torch.nn.utils import spectral_norm
from torch_geometric.data import DataLoader
from torch.utils.data import random_split, Subset
from bioneural.core.biololgicallayer2 import BioLogicalNeuron
from torch_geometric.datasets import TUDataset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_recall_fscore_support,
    roc_auc_score
)
class GraphBiologicalNetwork(nn.Module):
    def __init__(self, num_node_features, num_classes=2):
        super().__init__()
        
        # Graph attention layers
        self.gat1 = GATConv(num_node_features, 64 // 4, heads=4, dropout=0.3)
        self.gat2 = GATConv(64, 128 // 4, heads=4, dropout=0.3)
        self.gat3 = GATConv(128, 256 // 4, heads=4, dropout=0.3)

        # Batch normalization
        self.bn1 = BatchNorm(64)
        self.bn2 = BatchNorm(128)
        self.bn3 = BatchNorm(256)
     
        # Jumping knowledge connection
        self.jk = JumpingKnowledge(mode='cat')
        
        # Biological layers
        jk_dim = (64 + 128 + 256) * 2
        # Biological layers
        self.bio_layers = nn.ModuleList([
            BioLogicalNeuron(jk_dim, 256,log_file='neuron_1'),
            BioLogicalNeuron(256, 128,log_file='neuron_2'),
            BioLogicalNeuron(128, 64,log_file='neuron_3')
        ])
        
        # Classifier
        self.classifier = nn.Sequential(
            spectral_norm(nn.Linear(64, 32)),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.LayerNorm(32),
            spectral_norm(nn.Linear(32, num_classes))
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Multi-scale feature extraction
        x1 = F.elu(self.bn1(self.gat1(x, edge_index)))
        x2 = F.elu(self.bn2(self.gat2(x1, edge_index)))
        x3 = F.elu(self.bn3(self.gat3(x2, edge_index)))
        
        # Jumping Knowledge connection
        x = self.jk([x1, x2, x3])
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x = torch.cat([x_mean, x_sum], dim=1)
        
        # Biological neural processing
        health_reports = []
        for bio_layer in self.bio_layers:
            x, health_report = bio_layer(x)
            health_reports.append(health_report)
        
        # Classifier
        x = self.classifier(x)
        return x, health_reports

def get_training_components(model):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.3)
    
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=0.005,  # Slightly increased learning rate
        weight_decay=0.05,  # Increased weight decay for better regularization
        betas=(0.9, 0.999)
    )
    
    # one cycle lr scheduler with more flexibility
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.005,
        epochs=250,  # Increased epochs
        steps_per_epoch=50,
        pct_start=0.3,  # Adjusted percentage of increase
        anneal_strategy='cos',
        div_factor=10.0,  # Reduced div factor
        final_div_factor=100.0  # Reduced final div factor
    )
    
    return criterion, optimizer, scheduler

class EarlyStopping:
    def __init__(self, patience=25, min_delta=0.0005):
        """
        Enhanced early stopping with more flexibility
        
        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float): Minimum change in loss to qualify as an improvement
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

class PublicationTrainer:
    def __init__(self, dataset_name='MUTAG', n_splits=15   , seed=42, wandb_logging=True):
        self.dataset_name = dataset_name
        self.n_splits = n_splits
        self.seed = seed
        self.wandb_logging = wandb_logging
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Added parameters for robust training
        self.label_smoothing = 0.3
        self.gradient_clip = 1.0
        
        # Prepare dataset
        self.dataset = self._prepare_dataset()
        
    def _prepare_dataset(self):
        dataset = TUDataset(root=f'data/{self.dataset_name}', name=self.dataset_name)
        
        # Add dummy features if no node features
        if dataset.num_node_features == 0:
            dataset.data.x = torch.ones((dataset.data.num_nodes, 1))
        
        return dataset
    
    def _create_data_splits(self):
        labels = [data.y.item() for data in self.dataset]
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        return list(skf.split(np.zeros(len(labels)), labels))
    
    def train_and_evaluate(self):
        # Initialize results tracking
        results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'auc': []
        }

        # Logging setup
        if self.wandb_logging:
            wandb.init(
                project=f"{self.dataset_name}_GraphBiological",
                config={
                    "dataset": self.dataset_name,
                    "n_splits": self.n_splits,
                    "seed": self.seed
                }
            )

        # Cross-validation with progress bar
        cv_progress = tqdm(list(self._create_data_splits()), desc="Cross-Validation Folds")
        for fold, (train_idx, val_idx) in enumerate(cv_progress):
            # Prepare data loaders
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=32)
            test_loader = DataLoader(val_subset, batch_size=32)

            # Model and training components
            model = GraphBiologicalNetwork(
                num_node_features=self.dataset.num_node_features, 
                num_classes=self.dataset.num_classes
            ).to(self.device)

            # Use get_training_components to get criterion, optimizer, and scheduler
            criterion, optimizer, scheduler = get_training_components(model)

            # Early Stopping
            early_stopping = EarlyStopping(patience=25, min_delta=0.0005)

            # Training loop with progress bar
            best_val_acc = 0
            best_val_loss = float('inf')
            epoch_progress = tqdm(range(250), desc=f"Fold {fold+1} Training", leave=False)

            for epoch in epoch_progress:
                # Training phase
                model.train()
                train_loss = self._train_epoch(model, train_loader, optimizer, criterion)

                # Validation phase
                val_metrics = self._validate(model, val_loader, criterion)

                # Update progress bar description
                epoch_progress.set_postfix({
                    'Train Loss': f'{train_loss:.4f}', 
                    'Val Acc': f'{val_metrics["accuracy"]:.4f}'
                })

                # Wandb logging if enabled
                if self.wandb_logging:
                    wandb.log({
                        f"Fold_{fold+1}/Train_Loss": train_loss,
                        f"Fold_{fold+1}/Val_Loss": val_metrics['loss'],
                        f"Fold_{fold+1}/Val_Accuracy": val_metrics['accuracy']
                    })

                # Early Stopping Check
                if early_stopping(val_metrics['loss']):
                    print(f"Early stopping triggered in fold {fold+1} at epoch {epoch}")
                    break

                # Model checkpoint
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    best_val_loss = val_metrics['loss']
                    torch.save(model.state_dict(), f'best_model_fold{fold}.pth')

                # Step the scheduler
                scheduler.step()

            # Test evaluation with best model
            model.load_state_dict(torch.load(f'best_model_fold{fold}.pth'))
            test_progress = tqdm(desc=f"Fold {fold+1} Testing", total=1)
            test_metrics = self._evaluate(model, test_loader, criterion)
            test_progress.update(1)
            test_progress.close()

            # Store results
            for metric in results:
                results[metric].append(test_metrics[metric])

            # Update cross-validation progress
            cv_progress.set_postfix({
                'Best Val Acc': f'{best_val_acc:.4f}',
                'Test Acc': f'{test_metrics["accuracy"]:.4f}'
            })

            # Wandb log test metrics if enabled
            if self.wandb_logging:
                wandb.log({
                    f"Fold_{fold+1}/Test_Accuracy": test_metrics['accuracy'],
                    f"Fold_{fold+1}/Test_Precision": test_metrics['precision'],
                    f"Fold_{fold+1}/Test_Recall": test_metrics['recall'],
                    f"Fold_{fold+1}/Test_F1_Score": test_metrics['f1_score'],
                    f"Fold_{fold+1}/Test_AUC": test_metrics['auc']
                })

        # Compute final statistics
        publication_results = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values)
            } for metric, values in results.items()
        }

        # Wandb summary if enabled
        if self.wandb_logging:
            for metric, stats in publication_results.items():
                wandb.summary[f"Overall_{metric}_Mean"] = stats['mean']
                wandb.summary[f"Overall_{metric}_Std"] = stats['std']
            wandb.finish()

        # Save results to JSON
        with open('publication_results.json', 'w') as f:
            json.dump(publication_results, f, indent=4)

        return publication_results
    
    def _train_epoch(self, model, loader, optimizer, criterion):
        model.train()
        total_loss = 0
        
        for batch in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            outputs, _ = model(batch)
            loss = criterion(outputs, batch.y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _validate(self, model, loader, criterion):
        model.eval()
        total_loss = 0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                outputs, _ = model(batch)
                loss = criterion(outputs, batch.y)
                total_loss += loss.item()
                
                pred = outputs.argmax(dim=1)
                y_true.extend(batch.y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
        
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        return {
            'loss': total_loss / len(loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def _evaluate(self, model, loader, criterion):
        model.eval()
        total_loss = 0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                outputs, _ = model(batch)
                loss = criterion(outputs, batch.y)
                total_loss += loss.item()
                
                pred = outputs.argmax(dim=1)
                y_true.extend(batch.y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
        
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        auc = roc_auc_score(y_true, y_pred, multi_class='ovr')
        
        return {
            'loss': total_loss / len(loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }


if __name__ == "__main__":
    trainer = PublicationTrainer()
    results = trainer.train_and_evaluate()
    print("Publication Results:", results)