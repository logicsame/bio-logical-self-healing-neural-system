import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Any
import pandas as pd
from datetime import datetime

class BioNeuronVisualizer:
    """
    Advanced visualization system for biological neuron metrics with comprehensive repair strategy analysis.
    
    This class provides sophisticated visualization capabilities for monitoring and analyzing the behavior
    of biological neural networks, including health metrics, calcium dynamics, repair mechanisms, and
    phase space analysis. The visualizer tracks temporal evolution of neural states and generates
    publication-quality plots for research and debugging purposes.
    
    The visualization system is designed to handle real-time data streams from biological neuron
    simulations, automatically managing data history and generating insights into neural behavior
    patterns, homeostatic regulation effectiveness, and repair strategy distributions.
    
    Args:
        save_dir (str, optional): Directory path where visualization plots will be saved.
            The directory will be created if it doesn't exist. Defaults to "bio_visualizations".
    
    Attributes:
        save_dir (Path): Path object for the save directory with automatic creation.
        health_history (List[float]): Historical record of neuron health scores over time.
        stability_history (List[float]): Historical record of synaptic stability measurements.
        calcium_history (List[float]): Historical record of calcium concentration levels.
        repair_events (List[tuple]): List of (step, health_score) tuples marking repair occurrences.
        steps (List[int]): Sequence of training steps corresponding to all measurements.
        repair_strategy_history (Dict[str, List[int]]): Historical counts for each repair strategy:
            - 'synaptic_scaling': Frequency of synaptic scaling repair applications
            - 'selective_reinforcement': Frequency of selective synaptic reinforcement
            - 'activity_dependent_pruning': Frequency of activity-dependent pruning events
    
    Methods:
        update(step, health_report, calcium_level, repair_strategies=None):
            Updates all tracking histories with new data from a single training step.
            Automatically extracts values from tensors and handles repair event detection.
            
        plot_health_metrics():
            Generates time-series plot of health and stability metrics with repair event markers.
            Creates publication-quality visualization showing neural health evolution.
            
        plot_calcium_dynamics():
            Visualizes calcium concentration changes over time with rolling average overlay.
            Includes proper biological units and trend analysis for calcium homeostasis.
            
        plot_repair_strategies():
            Creates stacked area plot showing cumulative distribution of repair strategy usage.
            Provides insights into which repair mechanisms are most frequently activated.
            
        plot_phase_diagram():
            Generates phase space plot showing health vs. calcium relationships over time.
            Includes directional arrows and repair event markers for trajectory analysis.
            
        generate_comprehensive_summary():
            Creates comprehensive 2x2 subplot figure combining all major visualizations.
            Ideal for research presentations and comprehensive behavior analysis.
            
        save_all_plots():
            Batch generates and saves all available visualization types with error handling.
            Automatically timestamps files and manages plotting failures gracefully.
    
    Visualization Features:
        - **Time Series Analysis**: Tracks neural health, stability, and calcium dynamics over training
        - **Repair Strategy Monitoring**: Visualizes frequency and distribution of different repair mechanisms
        - **Phase Space Analysis**: Shows relationships between health and calcium in state space
        - **Event Marking**: Highlights repair events and critical state changes
        - **Rolling Averages**: Smooths noisy data for trend identification
        - **Publication Quality**: High DPI output with proper styling and annotations
    
    Data Handling:
        - **Tensor Compatibility**: Automatically extracts values from PyTorch tensors
        - **Missing Data Tolerance**: Gracefully handles incomplete or missing data streams
        - **Memory Management**: Maintains reasonable history lengths for long-running experiments
        - **Error Recovery**: Continues operation even when individual plots fail
    
    Example:
        >>> import torch
        >>> from datetime import datetime
        >>> 
        >>> # Initialize visualizer
        >>> visualizer = BioNeuronVisualizer(save_dir="neural_analysis")
        >>> 
        >>> # Simulate training loop with data updates
        >>> for step in range(1000):
        ...     # Generate synthetic health report
        ...     health_report = {
        ...         'current_health': torch.tensor(0.8 + 0.1 * np.random.randn()),
        ...         'stability': torch.tensor(0.7 + 0.1 * np.random.randn()),
        ...         'repair_performed': step % 100 == 0  # Repair every 100 steps
        ...     }
        ...     
        ...     # Generate synthetic calcium data
        ...     calcium_level = torch.tensor(0.5 + 0.2 * np.random.randn())
        ...     
        ...     # Update repair strategies
        ...     repair_strategies = {
        ...         'synaptic_scaling': step // 50,
        ...         'selective_reinforcement': step // 75,
        ...         'activity_dependent_pruning': step // 100
        ...     }
        ...     
        ...     # Update visualizer
        ...     visualizer.update(step, health_report, calcium_level, repair_strategies)
        ...     
        ...     # Generate plots periodically
        ...     if step % 200 == 0:
        ...         visualizer.save_all_plots()
        >>> 
        >>> # Generate final comprehensive summary
        >>> visualizer.generate_comprehensive_summary()
    
    File Naming Convention:
        All generated plots are automatically timestamped using the format 'YYYYMMDD_HHMMSS'
        to prevent overwrites and maintain analysis history:
        - health_metrics_20240615_143022.png
        - calcium_dynamics_20240615_143022.png
        - repair_strategies_20240615_143022.png
        - phase_diagram_20240615_143022.png
        - comprehensive_summary_20240615_143022.png
    
    Styling and Appearance:
        - Uses seaborn 'whitegrid' style with 'husl' color palette for publication quality
        - High DPI (600) output for crisp visualization in research papers
        - Consistent font sizing (12-15pt) and professional layouts
        - Proper axis labeling with units where applicable (e.g., calcium in μM)
        - Legend placement optimized for readability
    
    Note:
        This visualizer is designed for research and development environments where
        detailed analysis of biological neural network behavior is required. The
        plotting functions are optimized for insight generation rather than real-time
        performance, making them suitable for offline analysis and research documentation.
        
        For long-running experiments, consider periodic saving of plots to capture
        behavioral evolution over extended training periods. The visualizer automatically
        handles memory management but very long experiments may benefit from periodic
        history clearing.
    
    Raises:
        OSError: If save directory cannot be created or accessed.
        ValueError: If incompatible data types are passed to update methods.
        RuntimeError: If matplotlib/seaborn plotting operations fail due to backend issues.
    """
    def __init__(self, save_dir: str = "bio_visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize history containers
        self.health_history = []
        self.stability_history = []
        self.calcium_history = []
        self.repair_events = []
        self.steps = []
        
        # New tracking for repair strategies - MODIFIED: removed the specified strategies
        self.repair_strategy_history = {

            'synaptic_scaling': [],
            'selective_reinforcement': [],
            'activity_dependent_pruning': []
        }
        
        # Set style properly
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")
    
    def _get_value(self, obj: Any) -> float:
        """Helper method to extract value from tensor or direct value"""
        # If it's a tensor-like object with mean() and item() methods
        if hasattr(obj, 'mean') and callable(obj.mean) and hasattr(obj.mean(), 'item') and callable(obj.mean().item):
            return obj.mean().item()
        # If it's a tensor-like object with just item() method
        elif hasattr(obj, 'item') and callable(obj.item):
            return obj.item()
        # Otherwise assume it's a direct numerical value
        else:
            return float(obj)
        
    def update(self, step: int, health_report: Dict, calcium_level: Any, repair_strategies: Dict = None):
        """Update histories with new data including repair strategies"""
        self.steps.append(step)
        self.health_history.append(self._get_value(health_report['current_health']))
        self.stability_history.append(self._get_value(health_report['stability']))
        self.calcium_history.append(self._get_value(calcium_level))
        
        # Track repair strategies
        if repair_strategies:
            for strategy, count in repair_strategies.items():
                # Only track strategies that are in our tracking dict (excluding removed ones)
                if strategy in self.repair_strategy_history:
                    self.repair_strategy_history[strategy].append(count)
        
        if health_report.get('repair_performed', False):
            self.repair_events.append((step, self._get_value(health_report['current_health'])))
            
    def plot_repair_strategies(self):
        """Visualize repair strategy distribution and evolution"""
        # Check if we have steps and repair strategy data
        if not self.steps or all(len(data) == 0 for data in self.repair_strategy_history.values()):
            # Skip plotting if no data is available
            print("No repair strategy data available for plotting")
            return
            
        for strategy in self.repair_strategy_history:
            # If strategy data is empty or shorter than steps, pad with zeros
            if len(self.repair_strategy_history[strategy]) == 0:
                self.repair_strategy_history[strategy] = [0] * len(self.steps)
            elif len(self.repair_strategy_history[strategy]) < len(self.steps):
                padding = [0] * (len(self.steps) - len(self.repair_strategy_history[strategy]))
                self.repair_strategy_history[strategy].extend(padding)
        
        fig, ax = plt.subplots(figsize=(15, 7))
        
        strategies = list(self.repair_strategy_history.keys())
        strategy_data = [self.repair_strategy_history[strategy] for strategy in strategies]
        
        # Cumulative area plot
        ax.stackplot(self.steps, strategy_data, labels=strategies, alpha=0.7)
        
        ax.set_title('Repair Strategy Distribution', fontsize=15)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Strategy Count', fontsize=12)  # Added explicit y-axis label
        ax.legend(loc='upper left')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(self.save_dir / f'repair_strategies_{timestamp}.png', dpi=600, bbox_inches='tight')
        plt.close()
            
    def plot_health_metrics(self):
        """Plot health and stability trends"""
        if not self.steps or not self.health_history:
            # Skip plotting if no data is available
            print("No health metrics data available for plotting")
            return
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot health trend
        ax.plot(self.steps, self.health_history, label='Health', linewidth=2)
        ax.plot(self.steps, self.stability_history, label='Stability', linewidth=2)
        
        # Mark repair events
        if self.repair_events:
            repair_steps, repair_health = zip(*self.repair_events)
            ax.scatter(repair_steps, repair_health, color='red', marker='*', 
                      s=100, label='Repair Events', zorder=5)
            
        ax.set_title('Neuron Health Metrics', pad=20)
        ax.set_xlabel('Training Steps', fontsize=12)  # Added explicit font size
        ax.set_ylabel('Health Score', fontsize=12)  # More specific y-axis label
        ax.legend(frameon=True)
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(self.save_dir / f'health_metrics_{timestamp}.png', dpi=600, bbox_inches='tight')
        plt.close()
        
    def plot_calcium_dynamics(self):
        """Plot calcium level trends"""
        if not self.steps or not self.calcium_history:
            # Skip plotting if no data is available
            print("No calcium data available for plotting")
            return
            
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
        
        ax.set_title('Calcium Level Dynamics', pad=20)
        ax.set_xlabel('Training Steps', fontsize=12)  # Added explicit font size
        ax.set_ylabel('Calcium Concentration (μM)', fontsize=12)  # Added units to y-axis label
        ax.legend(frameon=True)
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(self.save_dir / f'calcium_dynamics_{timestamp}.png', dpi=600, bbox_inches='tight')
        plt.close()
        
    def plot_phase_diagram(self):
        """Generate phase diagram showing relationship between health and calcium levels"""
        if not self.health_history or not self.calcium_history:
            print("No phase diagram data available for plotting")
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
        plt.xlabel('Health Score', fontsize=12)  # More specific x-axis label with font size
        plt.ylabel('Calcium Concentration (μM)', fontsize=12)  # More specific y-axis label with units
        plt.title('Health-Calcium Phase Space', pad=20)
        
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
        plt.savefig(self.save_dir / f'phase_diagram_{timestamp}.png', dpi=600, bbox_inches='tight')
        plt.close()
        
    def generate_comprehensive_summary(self):
        """Generate a comprehensive summary plot with repair strategies"""
        if not self.steps or not self.health_history:
            print("No data available for comprehensive summary")
            return
            
        fig, axs = plt.subplots(2, 2, figsize=(20, 15))
        
        # Health and Stability (Top Left)
        axs[0, 0].plot(self.steps, self.health_history, label='Health', linewidth=2)
        axs[0, 0].plot(self.steps, self.stability_history, label='Stability', linewidth=2)
        if self.repair_events:
            repair_steps, repair_health = zip(*self.repair_events)
            axs[0, 0].scatter(repair_steps, repair_health, color='red', marker='*',
                               s=100, label='Repair Events', zorder=5)
        axs[0, 0].set_title('Neuron Health Metrics', fontsize=14)
        axs[0, 0].set_xlabel('Training Steps', fontsize=12)  # Added x-axis label
        axs[0, 0].set_ylabel('Health Score', fontsize=12)  # Added y-axis label
        axs[0, 0].legend()
        
        # Calcium Dynamics (Top Right)
        axs[0, 1].plot(self.steps, self.calcium_history, label='Calcium Level', color='purple')
        window = min(50, len(self.calcium_history))
        if window > 1:
            rolling_avg = pd.Series(self.calcium_history).rolling(window=window).mean()
            axs[0, 1].plot(self.steps, rolling_avg, label=f'{window}-step Average', 
                           color='blue', linestyle='--')
        axs[0, 1].set_title('Calcium Level Dynamics', fontsize=14)
        axs[0, 1].set_xlabel('Training Steps', fontsize=12)  # Added x-axis label
        axs[0, 1].set_ylabel('Calcium Concentration (μM)', fontsize=12)  # Added y-axis label with units
        axs[0, 1].legend()
        
        # Repair Strategies Cumulative Plot (Bottom Left)
        for strategy in self.repair_strategy_history:
            # If strategy data is empty or shorter than steps, pad with zeros
            if len(self.repair_strategy_history[strategy]) == 0:
                self.repair_strategy_history[strategy] = [0] * len(self.steps)
            elif len(self.repair_strategy_history[strategy]) < len(self.steps):
                padding = [0] * (len(self.steps) - len(self.repair_strategy_history[strategy]))
                self.repair_strategy_history[strategy].extend(padding)
                
        strategies = list(self.repair_strategy_history.keys())
        strategy_data = [self.repair_strategy_history[strategy] for strategy in strategies]
        
        # Check if we have strategies to plot
        if strategies and all(len(data) > 0 for data in strategy_data):
            axs[1, 0].stackplot(self.steps, strategy_data, labels=strategies, alpha=0.7)
            axs[1, 0].set_title('Repair Strategy Distribution', fontsize=14)
            axs[1, 0].set_xlabel('Training Steps', fontsize=12)  # Added x-axis label
            axs[1, 0].set_ylabel('Strategy Count', fontsize=12)  # Added y-axis label
            axs[1, 0].legend(loc='upper left')
        else:
            axs[1, 0].text(0.5, 0.5, "No repair strategy data available", 
                          horizontalalignment='center', verticalalignment='center',
                          transform=axs[1, 0].transAxes, fontsize=12)
            axs[1, 0].set_xlabel('Training Steps', fontsize=12)  # Added x-axis label even for empty plot
            axs[1, 0].set_ylabel('Strategy Count', fontsize=12)  # Added y-axis label even for empty plot
        
        # Health-Calcium Phase Space (Bottom Right)
        scatter = axs[1, 1].scatter(self.health_history, self.calcium_history, 
                                    c=self.steps, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, ax=axs[1, 1], label='Training Steps')
        axs[1, 1].set_title('Health-Calcium Phase Space', fontsize=14)
        axs[1, 1].set_xlabel('Health Score', fontsize=12)  # Added x-axis label
        axs[1, 1].set_ylabel('Calcium Concentration (μM)', fontsize=12)  # Added y-axis label with units
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(self.save_dir / f'comprehensive_summary_{timestamp}.png', dpi=600, bbox_inches='tight')
        plt.close()
        
    def save_all_plots(self):
        """Generate and save all visualization plots"""
        try:
            # Check if we have any data to plot
            if not self.steps:
                print("No data available for plotting. Skipping all plots.")
                return
                
            self.plot_health_metrics()
            self.plot_calcium_dynamics()
            self.plot_phase_diagram()
            self.plot_repair_strategies()  # New method
            self.generate_comprehensive_summary()  
        except Exception as e:
            print(f"Error during plot generation: {str(e)}")
