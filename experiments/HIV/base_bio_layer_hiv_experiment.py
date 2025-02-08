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
import shutil
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool,GCNConv, global_add_pool, JumpingKnowledge, BatchNorm
from torch.nn.utils import spectral_norm
from torch_geometric.data import DataLoader
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

# Set default tensor type to float to prevent dtype issues
torch.set_default_tensor_type(torch.FloatTensor)

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
    results_dir = 'hiv_results_base_bio_layer'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'logs'), exist_ok=True)
    return results_dir

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
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, BatchNorm
from torch.nn.utils import spectral_norm
from torch_geometric.data import DataLoader
from torch.utils.data import random_split, Subset
from bioneural.core.biololgicallayer import BioLogicalNeuron
from torch_geometric.datasets import TUDataset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_recall_fscore_support,
    roc_auc_score
)

# Set default tensor type to float to prevent dtype issues
torch.set_default_tensor_type(torch.FloatTensor)

class GraphBioNetwork(nn.Module):
    def __init__(self, num_node_features, num_classes=2,enable_monitoring=False,disable_monitoring=False,results_dir = 'hiv_results_base_bio_layer'):
        super().__init__()
        
        
        
        # Increased complexity and capacity
        hidden_dim = 512  # Doubled hidden dimension
        monitoring_state = enable_monitoring and not disable_monitoring
        
        log_base_path = os.path.join(results_dir, 'logs')
        
        # Regular GCN layers instead of GAT
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
    
        
        # Enhanced normalization
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm4 = nn.BatchNorm1d(hidden_dim)
        
        
        # Enhanced biological layers with finer-tuned parameters
        pooled_dim = hidden_dim * 2  # For concatenated pooling
        self.bio_layers = nn.ModuleList([
            BioLogicalNeuron(pooled_dim, 2048, repair_threshold=0.95, repair_intensity=0.02, plasticity_rate=0.001, enable_monitoring=monitoring_state, log_file=os.path.join(log_base_path, 'base_bio_layer_hiv.log')),
            BioLogicalNeuron(2048, 1024, repair_threshold=0.95, repair_intensity=0.02, plasticity_rate=0.001, enable_monitoring=monitoring_state, log_file=os.path.join(log_base_path, 'base_bio_layer_hiv.log')),
            BioLogicalNeuron(1024, 512, repair_threshold=0.95, repair_intensity=0.02, plasticity_rate=0.001, enable_monitoring=monitoring_state, log_file=os.path.join(log_base_path, 'base_bio_layer_hiv.log')),
            BioLogicalNeuron(512, 128, repair_threshold=0.95, repair_intensity=0.02, plasticity_rate=0.001, enable_monitoring=monitoring_state, log_file=os.path.join(log_base_path, 'base_bio_layer_hiv.log')),
        ])
        
        # Advanced classifier with skip connections
        self.classifier = nn.Sequential(
            nn.LayerNorm(128),
            spectral_norm(nn.Linear(128, 512)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(512),
            spectral_norm(nn.Linear(512, 256)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(256),
            spectral_norm(nn.Linear(256, num_classes))
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Enhanced feature extraction with skip connections
        x1 = self.batch_norm1(F.elu(self.conv1(x, edge_index)))
        x2 = self.batch_norm2(F.elu(self.conv2(x1, edge_index))) + x1
        x3 = self.batch_norm3(F.elu(self.conv3(x2, edge_index))) + x2
        x4 = self.batch_norm4(F.elu(self.conv4(x3, edge_index))) + x3
        
        # Multi-scale pooling
        x_mean = global_mean_pool(x4, batch)
        x_sum = global_add_pool(x4, batch)
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
    def __init__(self, patience=10, min_delta=0.0005):
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
    def __init__(self, dataset_name='ogbg-molhiv', n_splits=5, seed=42, wandb_logging=False, enable_monitoring=False, disable_monitoring=False):
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
        
        # Prepare dataset
        self.dataset = self._prepare_dataset()
        self.results_dir = create_results_directory()
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
                disable_monitoring=self.disable_monitoring
            ).to(self.device)

            # Use get_training_components to get criterion, optimizer, and scheduler
            criterion, optimizer, scheduler = get_training_components(model)
 
            # Early Stopping
            early_stopping = EarlyStopping(patience=10, min_delta=0.0005)

            # Training loop with progress bar
            best_val_acc = 0
            best_val_loss = float('inf')
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

                # Model checkpoint
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    best_val_loss = val_metrics['loss']
                    model_path = os.path.join(self.results_dir, 'models', f'best_model_fold{fold}.pth')
                    torch.save(model.state_dict(), model_path)

                # Step the scheduler
                scheduler.step()

            # Test evaluation with best model
            model.load_state_dict(torch.load(model_path))
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
                
                # Ensure x is float
                batch.x = batch.x.float()
                
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

import argparse

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
    
    trainer = HIVTrainer(
        n_splits=args.n_splits, 
        wandb_logging=args.wandb,
        enable_monitoring=args.enable_monitoring,
        disable_monitoring=args.disable_monitoring
    )
    
    results = trainer.train_and_evaluate()
    print("Publication Results:", results)

if __name__ == "__main__":
    main()