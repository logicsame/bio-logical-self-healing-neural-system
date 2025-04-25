import os
import torch
import numpy as np
import wandb
import datetime
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from ogb.graphproppred import PygGraphPropPredDataset

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool, JumpingKnowledge, BatchNorm
import torch.nn.utils.parametrizations as parametrizations
from torch_geometric.loader import DataLoader  # Updated import
from torch.utils.data import random_split, Subset
from torch_geometric.datasets import TUDataset
from bioneural.core.biololgicallayer import BioLogicalNeuron

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_recall_fscore_support,
    roc_auc_score
)
import shutil

# Set default dtype instead of default tensor type
torch.set_default_dtype(torch.float32)

import random
def augment_batch(batch):
    """Enhanced augmentation strategy"""
    # Node feature augmentation
    if random.random() < 0.7:  # Increased probability
        noise_scale = random.uniform(0.01, 0.05)
        noise = torch.randn_like(batch.x) * noise_scale
        batch.x = batch.x + noise
    
    # Feature masking
    if random.random() < 0.3:
        mask = torch.bernoulli(torch.ones_like(batch.x) * 0.9)
        batch.x = batch.x * mask
    
    # Edge dropout
    if random.random() < 0.2:
        edge_mask = torch.bernoulli(torch.ones(batch.edge_index.size(1)) * 0.95)
        edge_mask = edge_mask.bool()
        batch.edge_index = batch.edge_index[:, edge_mask]
    
    return batch

def get_training_components(model, num_epochs=400, steps_per_epoch=50):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # Reduced smoothing
    
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=0.0005,  # Lower initial learning rate
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Multi-cycle learning rate scheduler
    total_steps = num_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.005,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # Shorter warmup
        anneal_strategy='cos',
        div_factor=50.0,
        final_div_factor=5000.0,
        three_phase=True  # Enable three-phase learning
    )
    
    return criterion, optimizer, scheduler

def create_results_directory():
    results_dir = 'hiv_results_full_architecture'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'logs'), exist_ok=True)
    return results_dir


class GraphBioNetwork(nn.Module):
    def __init__(self, num_node_features, num_classes=2, enable_monitoring=False, disable_monitoring=False, results_dir='hiv_results_full_architecture'):
        super().__init__()
        
        # Increased complexity and capacity
        hidden_dim = 512  # Doubled hidden dimension
        monitoring_state = enable_monitoring and not disable_monitoring
        log_base_path = os.path.join(results_dir, 'logs')
        
        # Deeper GAT layers with more heads
        self.gat1 = GATConv(num_node_features, hidden_dim // 16, heads=16, dropout=0.15)
        self.gat2 = GATConv(hidden_dim, hidden_dim // 16, heads=16, dropout=0.15)
        self.gat3 = GATConv(hidden_dim, hidden_dim // 16, heads=16, dropout=0.15)
        self.gat4 = GATConv(hidden_dim, hidden_dim // 16, heads=16, dropout=0.15)
        self.gat5 = GATConv(hidden_dim, hidden_dim // 16, heads=16, dropout=0.15)
        
        # Enhanced normalization
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm4 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm5 = nn.BatchNorm1d(hidden_dim)
        
        # Advanced JK connection
        self.jk = JumpingKnowledge(mode='max', channels=hidden_dim, num_layers=5)
        
        # Enhanced biological layers with finer-tuned parameters
        jk_dim = hidden_dim * 2
        self.bio_layers = nn.ModuleList([
            BioLogicalNeuron(jk_dim, 2048, repair_threshold=0.95, repair_intensity=0.02, plasticity_rate=0.001, enable_monitoring=monitoring_state, log_file=os.path.join(log_base_path, 'full_architecture_hiv.log')),
            BioLogicalNeuron(2048, 1024, repair_threshold=0.95, repair_intensity=0.02, plasticity_rate=0.001, enable_monitoring=monitoring_state, log_file=os.path.join(log_base_path, 'full_architecture_hiv.log')),
            BioLogicalNeuron(1024, 512, repair_threshold=0.95, repair_intensity=0.02, plasticity_rate=0.001, enable_monitoring=monitoring_state, log_file=os.path.join(log_base_path, 'full_architecture_hiv.log')),
            BioLogicalNeuron(512, 256, repair_threshold=0.95, repair_intensity=0.02, plasticity_rate=0.001, enable_monitoring=monitoring_state, log_file=os.path.join(log_base_path, 'full_architecture_hiv.log')),
            BioLogicalNeuron(256, 128, repair_threshold=0.95, repair_intensity=0.02, plasticity_rate=0.001, enable_monitoring=monitoring_state, log_file=os.path.join(log_base_path, 'full_architecture_hiv.log')),
        ])
        
        # Advanced classifier with skip connections using parametrizations.weight_norm instead of spectral_norm
        self.classifier = nn.Sequential(
            nn.LayerNorm(128),
            parametrizations.weight_norm(nn.Linear(128, 512)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(512),
            parametrizations.weight_norm(nn.Linear(512, 256)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(256),
            parametrizations.weight_norm(nn.Linear(256, num_classes))
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Enhanced feature extraction with skip connections
        x1 = self.batch_norm1(F.elu(self.gat1(x, edge_index)))
        x2 = self.batch_norm2(F.elu(self.gat2(x1, edge_index))) + x1
        x3 = self.batch_norm3(F.elu(self.gat3(x2, edge_index))) + x2
        x4 = self.batch_norm4(F.elu(self.gat4(x3, edge_index))) + x3
        x5 = self.batch_norm5(F.elu(self.gat5(x4, edge_index))) + x4
        
        # Advanced pooling with JK
        x = self.jk([x1, x2, x3, x4, x5])
        
        # Multi-scale pooling
        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x = torch.cat([x_mean, x_sum], dim=1)
        
        # Enhanced biological processing
        prev_x = x
        health_reports = []
        for i, bio_layer in enumerate(self.bio_layers):
            x, health_report = bio_layer(x)
            if x.shape == prev_x.shape:
                x = x + prev_x * (0.1 / (i + 1))  # Decaying residual connections
            prev_x = x
            health_reports.append(health_report)
        
        x = self.classifier(x)
        return x, health_reports

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0005):
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

class HIVTrainer:
    def __init__(self, dataset_name='ogbg-molhiv', n_splits=5, seed=42, wandb_logging=True,
                 enable_monitoring=False, disable_monitoring=False):
        self.dataset_name = dataset_name
        self.n_splits = n_splits
        self.seed = seed
        self.wandb_logging = wandb_logging
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_monitoring = enable_monitoring
        self.disable_monitoring = disable_monitoring
        # Added parameters for robust training
        self.label_smoothing = 0.1
        self.gradient_clip = 0.5
        self.results_dir = create_results_directory()
        # Prepare dataset
        self.dataset = self._prepare_dataset()
        
    def _prepare_dataset(self):
        dataset = PygGraphPropPredDataset(root=f'data/{self.dataset_name}', name=self.dataset_name)
    
        # Ensure y is a 1D tensor
        dataset.data.y = dataset.data.y.squeeze()
    
        # Add dummy features if no node features, ensuring float type
        if dataset.num_node_features == 0:
            dataset.data.x = torch.ones((dataset.data.num_nodes, 1), dtype=torch.float)
    
        return dataset
    
    def _create_data_splits(self):
        labels = [data.y.item() for data in self.dataset]
        indices = np.arange(len(labels))
        
        # Calculate graph properties for better stratification
        graph_properties = []
        for data in self.dataset:
            num_nodes = data.x.shape[0]
            num_edges = data.edge_index.shape[1] // 2
            avg_degree = num_edges / num_nodes
            graph_properties.append([num_nodes, avg_degree])
        
        graph_properties = np.array(graph_properties)
        
        # Normalize properties
        graph_properties = (graph_properties - graph_properties.mean(axis=0)) / graph_properties.std(axis=0)
        
        # Create composite stratification feature
        composite_labels = []
        for i, label in enumerate(labels):
            # Combine label with graph properties
            composite_label = f"{label}_{int(graph_properties[i][0] > 0)}_{int(graph_properties[i][1] > 0)}"
            composite_labels.append(composite_label)
        
        # Create stratified folds using composite labels
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        splits = []
        
        for train_idx, temp_idx in skf.split(indices, composite_labels):
            # Further stratify the validation/test split
            temp_composite_labels = [composite_labels[i] for i in temp_idx]
            sss2 = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.seed)
            val_idx, test_idx = next(sss2.split(temp_idx, temp_composite_labels))
            
            val_idx = temp_idx[val_idx]
            test_idx = temp_idx[test_idx]
            
            splits.append((train_idx, val_idx, test_idx))
        
        return splits
    
    def train_and_evaluate(self):
        results = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'auc': []}
    
        if self.wandb_logging:
            wandb.init(project=f"{self.dataset_name}_GraphBiological",
                    config={"dataset": self.dataset_name, "n_splits": self.n_splits, "seed": self.seed})

        cv_progress = tqdm(list(self._create_data_splits()), desc="Cross-Validation Folds")
        for fold, (train_idx, val_idx, test_idx) in enumerate(cv_progress):
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)
            test_subset = Subset(self.dataset, test_idx)

            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=32)
            test_loader = DataLoader(test_subset, batch_size=32)

            # Model and training components
            model = GraphBioNetwork(
                num_node_features=self.dataset.num_node_features, 
                num_classes=self.dataset.num_classes,
                enable_monitoring=self.enable_monitoring,
                disable_monitoring=self.disable_monitoring,
                results_dir=self.results_dir
            ).to(self.device)

            # Use get_training_components to get criterion, optimizer, and scheduler
            criterion, optimizer, scheduler = get_training_components(model)
 
            # Early Stopping
            early_stopping = EarlyStopping(patience=25, min_delta=0.0005)

            # Training loop with progress bar
            best_val_acc = 0
            best_val_loss = float('inf')
            model_path = os.path.join(self.results_dir, 'models', f'best_model_fold{fold}.pth')
            epoch_progress = tqdm(range(100), desc=f"Fold {fold+1} Training", leave=False)

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

                # Model checkpoint - FIXED: Save model before trying to load it
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    best_val_loss = val_metrics['loss']
                    torch.save(model.state_dict(), model_path)

                # Step the scheduler
                scheduler.step()

            # Test evaluation with best model
            try:
                model.load_state_dict(torch.load(model_path, weights_only=True))
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
                        f"Fold_{fold+1}/Test_AUC": test_metrics['auc'] if 'auc' in test_metrics else 0.0
                    })
            except FileNotFoundError:
                print(f"Warning: Model file not found for fold {fold}. Using current model state for testing.")
                test_metrics = self._evaluate(model, test_loader, criterion)
                # Continue with storing results as above...
                for metric in results:
                    results[metric].append(test_metrics[metric])

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
        with open(os.path.join(self.results_dir, 'publication_results.json'), 'w') as f:
            json.dump(publication_results, f, indent=4)

        # Move bio_vis folder if it exists
        if self.enable_monitoring:
            bio_vis_src = 'bio_vis'
            if os.path.exists(bio_vis_src):
                bio_vis_dest = os.path.join(self.results_dir, 'bio_vis')
                if os.path.exists(bio_vis_dest):
                    shutil.rmtree(bio_vis_dest)
                shutil.move(bio_vis_src, bio_vis_dest) 

        return publication_results
    
    def _train_epoch(self, model, loader, optimizer, criterion):
        model.train()
        total_loss = 0
    
        for batch in loader:
            batch = batch.to(self.device)
        
            # Ensure x is float
            batch.x = batch.x.float()
        
            # Apply augmentation (optional)
            if random.random() < 0.5:  # Apply augmentation with 50% probability
                batch = augment_batch(batch)
        
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
                
                # Ensure x is float
                batch.x = batch.x.float()
                
                outputs, _ = model(batch)
                loss = criterion(outputs, batch.y)
                total_loss += loss.item()
                
                pred = outputs.argmax(dim=1)
                y_true.extend(batch.y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
        
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        
        # Handle zero_division in precision/recall metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
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
        y_probs = []  # For AUC calculation
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                
                # Ensure x is float
                batch.x = batch.x.float()
                
                outputs, _ = model(batch)
                loss = criterion(outputs, batch.y)
                total_loss += loss.item()
                
                # Get predictions
                pred = outputs.argmax(dim=1)
                probs = F.softmax(outputs, dim=1)
                
                y_true.extend(batch.y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
                y_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays for metric calculations
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_probs = np.array(y_probs)
        
        accuracy = np.mean(y_true == y_pred)
        
        # Handle zero_division in precision/recall metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Calculate AUC if possible
        try:
            # For binary classification
            if y_probs.shape[1] == 2:
                auc = roc_auc_score(y_true, y_probs[:, 1])
            # For multiclass
            else:
                auc = roc_auc_score(
                    np.eye(y_probs.shape[1])[y_true.astype(int)], 
                    y_probs, 
                    multi_class='ovr',
                    average='weighted'
                )
        except ValueError:
            # If AUC calculation fails, set it to 0.5 (random chance)
            auc = 0.5
            print("Warning: AUC calculation failed, setting to 0.5")
        
        return {
            'loss': total_loss / len(loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }

import argparse

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Publication Trainer with Biological Neural Network')
    
    # Monitoring arguments with mutually exclusive group
    monitoring_group = parser.add_mutually_exclusive_group()
    monitoring_group.add_argument('--enable-monitoring', action='store_true', 
                        help='Enable monitoring for biological layers')
    monitoring_group.add_argument('--disable-monitoring', action='store_true', 
                        help='Explicitly disable monitoring for biological layers')
    

    parser.add_argument('--n-splits', type=int, default=5, 
                        help='Number of cross-validation splits')
    parser.add_argument('--wandb', action='store_true', 
                        help='Enable Weights & Biases logging')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create trainer with parsed arguments
    trainer = HIVTrainer(
        n_splits=args.n_splits, 
        wandb_logging=args.wandb,
        enable_monitoring=args.enable_monitoring,
        disable_monitoring=args.disable_monitoring
    )
    
    # Train and evaluate
    results = trainer.train_and_evaluate()
    print("Publication Results:", results)

if __name__ == "__main__":
    main()