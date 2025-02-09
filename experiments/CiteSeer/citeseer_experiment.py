import os
import torch
import numpy as np
import wandb
import datetime
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import  argparse
import torch.nn as nn
from bioneural.core.biololgicallayer import BioLogicalNeuron
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool, JumpingKnowledge, BatchNorm, GCNConv
from torch.nn.utils import spectral_norm
from torch_geometric.datasets import Planetoid
from sklearn.model_selection import KFold
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_recall_fscore_support,
    roc_auc_score
)
from sklearn.model_selection import train_test_split

import os
import torch
import numpy as np
import wandb
import datetime
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool, JumpingKnowledge, BatchNorm, GCNConv
from torch.nn.utils import spectral_norm
from torch_geometric.datasets import Planetoid
from sklearn.model_selection import KFold
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_recall_fscore_support,
    roc_auc_score
)
from sklearn.model_selection import train_test_split




def create_results_directory():
    results_dir = 'citeseer_results_full_architecture'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'logs'), exist_ok=True)
    return results_dir


class GraphBioNetwork(nn.Module):
    def __init__(self, num_node_features, num_classes=7,enable_monitoring=False, disable_monitoring=False,results_dir = 'citeseer_results_full_architecture'):
        super().__init__()
        
        # Increased capacity with wider hidden dimensions
        hidden_dim = 768
        monitoring_state = enable_monitoring and not disable_monitoring
        log_base_path = os.path.join(results_dir, 'logs')
        # Deeper architecture with residual connections
        self.gat1 = GATConv(num_node_features, hidden_dim // 8, heads=8, dropout=0.2)
        self.gat2 = GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.2)
        self.gat3 = GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.2)
        self.gat4 = GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.2)
        self.gat5 = GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.2)
        self.gat6 = GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.2)
        
        # Layer normalization instead of batch normalization for better stability
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)
        self.layer_norm4 = nn.LayerNorm(hidden_dim)
        self.layer_norm5 = nn.LayerNorm(hidden_dim)
        self.layer_norm6 = nn.LayerNorm(hidden_dim)
        
        # Added concatenation-based JK connection
        self.jk = JumpingKnowledge(mode='cat', channels=hidden_dim, num_layers=6)
        
        # More robust biological layers with adjusted parameters
        self.bio_layers = nn.ModuleList([
            BioLogicalNeuron(hidden_dim * 6, 1024, repair_threshold=0.95, repair_intensity=0.015, plasticity_rate=0.0015,summary_interval=5,enable_monitoring=monitoring_state,log_file=os.path.join(log_base_path, 'bio_layer_1.log')),
            BioLogicalNeuron(1024, 512, repair_threshold=0.95, repair_intensity=0.015, plasticity_rate=0.0015,summary_interval=5,enable_monitoring=monitoring_state,log_file=os.path.join(log_base_path, 'bio_layer_2.log')),
            BioLogicalNeuron(512, 256, repair_threshold=0.95, repair_intensity=0.015, plasticity_rate=0.0015,summary_interval=5,enable_monitoring=monitoring_state,log_file=os.path.join(log_base_path, 'bio_layer_3.log')),
        ])
        
        # Enhanced classifier with deeper architecture
        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            spectral_norm(nn.Linear(256, 128)),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(128),
            spectral_norm(nn.Linear(128, 64)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(64),
            nn.Linear(64, num_classes)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Enhanced forward pass with stronger residual connections
        x1 = self.layer_norm1(F.elu(self.gat1(x, edge_index)))
        x2 = self.layer_norm2(F.elu(self.gat2(x1, edge_index))) + x1
        x3 = self.layer_norm3(F.elu(self.gat3(x2, edge_index))) + x2
        x4 = self.layer_norm4(F.elu(self.gat4(x3, edge_index))) + x3
        x5 = self.layer_norm5(F.elu(self.gat5(x4, edge_index))) + x4
        x6 = self.layer_norm6(F.elu(self.gat6(x5, edge_index))) + x5
        
        # Concatenative aggregation of multi-scale features
        x = self.jk([x1, x2, x3, x4, x5, x6])
        
        # Biological processing
        health_reports = []
        for bio_layer in self.bio_layers:
            x, health_report = bio_layer(x)
            health_reports.append(health_report)
        
        # Final classification
        x = self.classifier(x)
        return x, health_reports

def get_training_components(model):
    # Use weighted cross entropy if classes are imbalanced
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=0.0005,  # Lower learning rate
        weight_decay=0.02,  # Increased weight decay
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.0005,
        epochs=300,  # Increased epochs
        steps_per_epoch=1,
        pct_start=0.3,  # Longer warmup
        anneal_strategy='cos',
        div_factor=15.0,
        final_div_factor=150.0
    )
    
    return criterion, optimizer, scheduler

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001):
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

class CiteSeerTrainer:
    def __init__(self, n_splits=10, seed=42, wandb_logging=False, enable_monitoring=False, disable_monitoring=False):
        self.n_splits = n_splits
        self.seed = seed
        self.wandb_logging = wandb_logging
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_monitoring = enable_monitoring
        self.disable_monitoring = disable_monitoring
        
        # Load dataset
        self.dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
        self.data = self.dataset[0].to(self.device)
        self.results_dir = create_results_directory()
        # Training parameters
        self.gradient_clip = 0.5
        self.epochs = 200
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    def prepare_cv_split(self):
        """Prepare cross-validation splits with 80/10/10 ratio"""
        num_nodes = self.data.x.size(0)
        indices = np.arange(num_nodes)
        
        # First, separate out 10% for final test set
        train_val_idx, test_idx = train_test_split(
            indices, 
            test_size=0.1, 
            random_state=self.seed,
            stratify=self.data.y.cpu().numpy()
        )
        
        # Create KFold cross-validation for the remaining 90%
        self.kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        return train_val_idx, test_idx
    
    def _train_epoch(self, model, train_mask, criterion, optimizer):
        model.train()
        optimizer.zero_grad()
        
        output, _ = model(self.data)
        loss = criterion(output[train_mask], self.data.y[train_mask])
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
        
        optimizer.step()
        
        # Calculate accuracy
        pred = output[train_mask].argmax(dim=1)
        acc = (pred == self.data.y[train_mask]).float().mean()
        
        return loss.item(), acc.item()
    
    def _validate(self, model, val_mask, criterion):
        model.eval()
        with torch.no_grad():
            output, _ = model(self.data)
            loss = criterion(output[val_mask], self.data.y[val_mask])
            
            pred = output[val_mask].argmax(dim=1)
            true = self.data.y[val_mask]
            
            accuracy = (pred == true).float().mean()
            precision, recall, f1, _ = precision_recall_fscore_support(
                true.cpu(), pred.cpu(), average='weighted'
            )
            
            return {
                'loss': loss.item(),
                'accuracy': accuracy.item(),
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
    
    def _evaluate(self, model, test_mask, criterion):
        model.eval()
        with torch.no_grad():
            output, _ = model(self.data)
            loss = criterion(output[test_mask], self.data.y[test_mask])
            
            pred = output[test_mask].argmax(dim=1)
            true = self.data.y[test_mask]
            
            accuracy = (pred == true).float().mean()
            precision, recall, f1, _ = precision_recall_fscore_support(
                true.cpu(), pred.cpu(), average='weighted'
            )
            
            try:
                auc = roc_auc_score(
                    true.cpu(),
                    F.softmax(output[test_mask], dim=1).cpu(),
                    multi_class='ovr'
                )
            except:
                auc = 0.0
            
            return {
                'loss': loss.item(),
                'accuracy': accuracy.item(),
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc
            }
    
    def train_fold(self, fold_idx, train_idx, val_idx, test_idx):
        # Create masks
        train_mask = torch.zeros(self.data.x.size(0), dtype=torch.bool)
        val_mask = torch.zeros(self.data.x.size(0), dtype=torch.bool)
        test_mask = torch.zeros(self.data.x.size(0), dtype=torch.bool)
    
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
    
        self.data.train_mask = train_mask.to(self.device)
        self.data.val_mask = val_mask.to(self.device)
        self.data.test_mask = test_mask.to(self.device)
    
        # Initialize model and training components
        model = GraphBioNetwork(
            num_node_features=self.dataset.num_node_features,
            num_classes=self.dataset.num_classes,
            enable_monitoring=self.enable_monitoring,
            disable_monitoring=self.disable_monitoring,
            results_dir=self.results_dir  # Pass results_dir to model
        ).to(self.device)
    
        criterion, optimizer, scheduler = get_training_components(model)
        early_stopping = EarlyStopping(patience=20, min_delta=0.001)
    
        # Training loop with progress tracking
        best_val_acc = 0
        best_model_state = None
        epoch_progress = tqdm(range(self.epochs), desc=f"Fold {fold_idx + 1}/{self.n_splits}")
    
        for epoch in epoch_progress:
            # Training phase
            train_loss, train_acc = self._train_epoch(model, train_mask, criterion, optimizer)
        
            # Validation phase
            val_metrics = self._validate(model, val_mask, criterion)
        
            # Update progress bar
            epoch_progress.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Train Acc': f'{train_acc:.4f}',
                'Val Acc': f'{val_metrics["accuracy"]:.4f}'
            })
        
            # Wandb logging
            if self.wandb_logging:
                wandb.log({
                    f'fold_{fold_idx}_train_loss': train_loss,
                    f'fold_{fold_idx}_train_acc': train_acc,
                    f'fold_{fold_idx}_val_loss': val_metrics['loss'],
                    f'fold_{fold_idx}_val_acc': val_metrics['accuracy']
                })
        
            # Learning rate scheduling
            scheduler.step()
        
            # Early stopping check
            if early_stopping(val_metrics['loss']):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
        
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_model_state = model.state_dict()
                # Save model with correct fold_idx
                model_path = os.path.join(self.results_dir, 'models', f'best_model_fold_{fold_idx}.pth')
                torch.save(best_model_state, model_path)
    
        # Load best model for final evaluation
        model.load_state_dict(best_model_state)
        test_metrics = self._evaluate(model, test_mask, criterion)
    
        print(f"\nFold {fold_idx + 1} Results:")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
    
        return test_metrics, best_model_state
    
    def train_and_evaluate(self):
        """
        Performs full training and evaluation using cross-validation and a held-out test set.
        Returns final results including cross-validation statistics and final test performance.
        """
        if self.wandb_logging:
            wandb.init(project="Cora_GraphBiological_CV")

        # First, keep a completely separate test set
        train_val_indices, final_test_idx = train_test_split(
            np.arange(self.data.x.size(0)),
            test_size=0.1,
            random_state=self.seed,
            stratify=self.data.y.cpu().numpy()
        )

        # Initialize cross validation results storage
        cv_results = {
            'accuracy': [], 
            'precision': [], 
            'recall': [], 
            'f1_score': [], 
            'auc': []
        }
    
        # Initialize best model tracking
        best_val_metrics = {'accuracy': 0}
        best_model_state = None
    
        # Perform cross-validation on training data
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        cv_progress = tqdm(enumerate(kfold.split(train_val_indices)), 
                      total=self.n_splits, 
                        desc="Cross-Validation Progress")
    
        for fold_idx, (train_idx, val_idx) in cv_progress:
            # Map indices back to original dataset indices
            train_idx = train_val_indices[train_idx]
            val_idx = train_val_indices[val_idx]
        
            # Train and evaluate fold
            fold_metrics, fold_model_state = self.train_fold(
                fold_idx, 
                train_idx, 
                val_idx, 
                final_test_idx  # Pass final test indices for consistent evaluation
            )
        
            # Store cross-validation results
            for metric in cv_results:
                cv_results[metric].append(fold_metrics[metric])
        
            # Update best model if current fold performs better
            if fold_metrics['accuracy'] > best_val_metrics['accuracy']:
                best_val_metrics = fold_metrics.copy()
                best_model_state = fold_model_state
        
            # Update progress bar with current fold's performance
            cv_progress.set_postfix({
                'Best Val Acc': f'{best_val_metrics["accuracy"]:.4f}',
                'Current Val Acc': f'{fold_metrics["accuracy"]:.4f}'
            })

        # Initialize and load best model for final evaluation
        final_model = GraphBioNetwork(
            num_node_features=self.dataset.num_node_features,
            num_classes=self.dataset.num_classes,
            enable_monitoring=self.enable_monitoring,
            disable_monitoring=self.disable_monitoring,
            results_dir=self.results_dir
        ).to(self.device)
        final_model.load_state_dict(best_model_state)

        # Create final test mask
        test_mask = torch.zeros(self.data.x.size(0), dtype=torch.bool)
        test_mask[final_test_idx] = True
        test_mask = test_mask.to(self.device)

        # Evaluate on final test set
        criterion = nn.CrossEntropyLoss()
        final_test_metrics = self._evaluate(final_model, test_mask, criterion)

        # Calculate cross-validation statistics
        cv_statistics = {
            metric: {
                'mean': float(np.mean(values)),  # Convert to float for JSON serialization
                'std': float(np.std(values))
            } for metric, values in cv_results.items()
        }

        # Combine all results
        final_results = {
            'cross_validation': cv_statistics,
            'final_test': {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                for k, v in final_test_metrics.items()
            }
        }

        # Log results to wandb if enabled
        if self.wandb_logging:
            for metric, stats in cv_statistics.items():
                wandb.summary[f"cv_{metric}_mean"] = stats['mean']
                wandb.summary[f"cv_{metric}_std"] = stats['std']
            for metric, value in final_test_metrics.items():
                wandb.summary[f"test_{metric}"] = value
            wandb.finish()

        # Save results to JSON
        results_path = os.path.join(self.results_dir, 'cv_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=4)

        # Save best model state
        best_model_path = os.path.join(self.results_dir, 'models', 'best_model_final.pth')
        torch.save(best_model_state, best_model_path)

        # Print final results
        print("\nCross-Validation Results")
        print("=" * 50)
        for metric, stats in cv_statistics.items():
            print(f"CV {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
        print("\nFinal Test Set Results")
        print("=" * 50)
        for metric, value in final_test_metrics.items():
            print(f"Test {metric}: {value:.4f}")

        return final_results


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Publication Trainer with Biological Neural Network')
    
    # Monitoring arguments with mutually exclusive group
    monitoring_group = parser.add_mutually_exclusive_group()
    monitoring_group.add_argument('--enable-monitoring', action='store_true', 
                        help='Enable monitoring for biological layers')
    monitoring_group.add_argument('--disable-monitoring', action='store_true', 
                        help='Explicitly disable monitoring for biological layers')
    
    # Other existing arguments can be added here
    parser.add_argument('--n-splits', type=int, default=10, 
                        help='Number of cross-validation splits')
    parser.add_argument('--wandb', action='store_true', 
                        help='Enable Weights & Biases logging')
    
    # Parse arguments
    args = parser.parse_args()

    trainer = CiteSeerTrainer(
        n_splits=args.n_splits, 
        wandb_logging=args.wandb, 
        enable_monitoring=args.enable_monitoring, 
        disable_monitoring=args.disable_monitoring
    )
    results = trainer.train_and_evaluate()
    print('Results:', results)

if __name__ == "__main__":
    main()