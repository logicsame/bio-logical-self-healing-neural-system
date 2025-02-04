import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import torch
from typing import Dict, List
import pandas as pd
from datetime import datetime

class BioNeuronVisualizer:
    """Visualizer for biological neuron metrics"""
    def __init__(self, save_dir: str = "bio_visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize history containers
        self.health_history = []
        self.stability_history = []
        self.calcium_history = []
        self.repair_events = []
        self.steps = []
        
        # Set style properly
        plt.style.use('default')  # Reset to default style
        sns.set_theme(style="whitegrid")  # Use seaborn's whitegrid theme
        sns.set_palette("husl")  # Set the color palette
        
    def update(self, step: int, health_report: Dict, calcium_level: torch.Tensor):
        """Update histories with new data"""
        self.steps.append(step)
        self.health_history.append(health_report['current_health'].mean().item())
        self.stability_history.append(health_report['stability'].mean().item())
        self.calcium_history.append(calcium_level.mean().item())
        
        if health_report.get('repair_performed', False):
            self.repair_events.append((step, health_report['current_health'].mean().item()))
            
    def plot_health_metrics(self):
        """Plot health and stability trends"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot health trend
        ax.plot(self.steps, self.health_history, label='Health', linewidth=2)
        ax.plot(self.steps, self.stability_history, label='Stability', linewidth=2)
        
        # Mark repair events
        if self.repair_events:
            repair_steps, repair_health = zip(*self.repair_events)
            ax.scatter(repair_steps, repair_health, color='red', marker='*', 
                      s=100, label='Repair Events', zorder=5)
            
        ax.set_title('Neuron Health and Stability Over Time', pad=20)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Metric Value')
        ax.legend(frameon=True)
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(self.save_dir / f'health_metrics_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_calcium_dynamics(self):
        """Plot calcium level trends"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot calcium levels
        ax.plot(self.steps, self.calcium_history, label='Calcium Level', 
                color='purple', linewidth=2)
        
        # Add rolling average
        window = min(50, len(self.calcium_history))
        if window > 1:
            rolling_avg = pd.Series(self.calcium_history).rolling(window=window).mean()
            ax.plot(self.steps, rolling_avg, label=f'{window}-step Average',
                   color='blue', linestyle='--', linewidth=1.5)
        
        ax.set_title('Calcium Dynamics Over Time', pad=20)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Calcium Level')
        ax.legend(frameon=True)
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(self.save_dir / f'calcium_dynamics_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_phase_diagram(self):
        """Generate phase diagram showing relationship between health and calcium levels"""
        if not self.health_history or not self.calcium_history:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create phase plot
        plt.scatter(self.health_history, self.calcium_history, 
                   c=self.steps, cmap='viridis', 
                   alpha=0.6, s=50)
        
        # Add arrows to show direction of time
        step_interval = max(1, len(self.steps) // 20)  # Show arrows at 20 points
        for i in range(0, len(self.steps) - 1, step_interval):
            plt.arrow(self.health_history[i], self.calcium_history[i],
                     (self.health_history[i+1] - self.health_history[i]) * 0.2,
                     (self.calcium_history[i+1] - self.calcium_history[i]) * 0.2,
                     head_width=0.01, head_length=0.02, fc='gray', ec='gray',
                     alpha=0.5)
        
        plt.colorbar(label='Training Steps')
        plt.xlabel('Health Level')
        plt.ylabel('Calcium Level')
        plt.title('Health-Calcium Phase Diagram', pad=20)
        
        # Add repair event markers if they exist
        if self.repair_events:
            repair_steps, repair_health = zip(*self.repair_events)
            repair_calcium = [self.calcium_history[self.steps.index(step)] 
                            for step in repair_steps]
            plt.scatter(repair_health, repair_calcium, 
                       color='red', marker='*', s=200,
                       label='Repair Events', zorder=5)
            plt.legend()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(self.save_dir / f'phase_diagram_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_summary_plot(self):
        """Generate comprehensive summary plot"""
        fig, axs = plt.subplots(3, 1, figsize=(15, 12), height_ratios=[2, 2, 2])
        
        # Health and Stability
        axs[0].plot(self.steps, self.health_history, label='Health', linewidth=2)
        axs[0].plot(self.steps, self.stability_history, label='Stability', linewidth=2)
        if self.repair_events:
            repair_steps, repair_health = zip(*self.repair_events)
            axs[0].scatter(repair_steps, repair_health, color='red', marker='*',
                          s=100, label='Repair Events', zorder=5)
        axs[0].set_title('Health and Stability Metrics', pad=20)
        axs[0].legend(frameon=True)
        
        # Calcium Dynamics
        axs[1].plot(self.steps, self.calcium_history, label='Calcium Level',
                   color='purple', linewidth=2)
        window = min(50, len(self.calcium_history))
        if window > 1:
            rolling_avg = pd.Series(self.calcium_history).rolling(window=window).mean()
            axs[1].plot(self.steps, rolling_avg, label=f'{window}-step Average',
                       color='blue', linestyle='--', linewidth=1.5)
        axs[1].set_title('Calcium Dynamics', pad=20)
        axs[1].legend(frameon=True)
        
        # Phase Diagram
        scatter = axs[2].scatter(self.health_history, self.calcium_history,
                               c=self.steps, cmap='viridis', alpha=0.6, s=50)
        if self.repair_events:
            repair_steps, repair_health = zip(*self.repair_events)
            repair_calcium = [self.calcium_history[self.steps.index(step)]
                            for step in repair_steps]
            axs[2].scatter(repair_health, repair_calcium,
                          color='red', marker='*', s=200,
                          label='Repair Events', zorder=5)
        plt.colorbar(scatter, ax=axs[2], label='Training Steps')
        axs[2].set_xlabel('Health Level')
        axs[2].set_ylabel('Calcium Level')
        axs[2].set_title('Health-Calcium Phase Diagram', pad=20)
        axs[2].legend()
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(self.save_dir / f'summary_plot_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_all_plots(self):
        """Generate and save all visualization plots"""
        self.plot_health_metrics()
        self.plot_calcium_dynamics()
        self.plot_phase_diagram()
        self.generate_summary_plot()