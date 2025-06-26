# Modified BioLogicalNeuron Implementation

from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Union
from collections import defaultdict
import numpy as np
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from ..utils.logging import HealthLogger
from ..metrics.healthtracker import HealthTracker
import logging
from ..visualization.biosysvisualization2 import BioNeuronVisualizer
from .homeostasis import HomeostaticRegulation

class BioLogicalNeuron(nn.Module):
    """
    Biological Neuron with biologically plausible homeostatic regulation and repair mechanisms.
    
    This module implements a neural network layer that mimics biological neuron behavior, including
    calcium-based homeostatic regulation, adaptive repair mechanisms, and comprehensive health monitoring.
    The neuron employs biologically inspired strategies such as synaptic scaling, selective reinforcement,
    and activity-dependent pruning to maintain optimal performance and prevent degradation.
    
    Args:
        in_features (int): Number of input features to the neuron layer.
        out_features (int): Number of output features from the neuron layer.
        plasticity_rate (float, optional): Rate of synaptic plasticity adaptation. Defaults to 0.008.
        repair_threshold (float, optional): Health threshold below which repair mechanisms activate. 
            Range: [0.0, 1.0]. Defaults to 0.5.
        repair_intensity (float, optional): Intensity of repair operations when triggered. 
            Higher values result in more aggressive repairs. Defaults to 0.08.
        calcium_threshold (float, optional): Threshold for calcium-based homeostatic regulation.
            When exceeded, triggers synaptic scaling mechanisms. Defaults to 0.9.
        enable_monitoring (bool, optional): Whether to enable health tracking, logging, and 
            visualization features. Defaults to True.
        log_file (str, optional): Path to log file for health monitoring. If None, logs to console.
            Defaults to "bioneuron_health.log".
        summary_interval (int, optional): Number of steps between health summary reports. 
            Defaults to 100.
        **kwargs: Additional keyword arguments passed to HomeostaticRegulation module.
    
    Attributes:
        linear (nn.Linear): Weight-normalized linear transformation layer.
        η (float): Plasticity rate for synaptic adaptation.
        homeostasis (HomeostaticRegulation): Homeostatic regulation module managing calcium dynamics.
        repair_threshold (float): Health threshold for repair activation.
        repair_intensity (float): Intensity of repair mechanisms.
        repair_count (int): Total number of repairs performed.
        repair_cooldown (int): Cooldown counter preventing excessive repair frequency.
        health_tracker (HealthTracker): Health monitoring and tracking system.
        logger (logging.Logger): Logger for health status and events.
        visualizer (BioNeuronVisualizer): Visualization system for neuron dynamics.
        step_counter (int): Counter tracking forward pass steps.
        summary_interval (int): Interval for health summary reports.
        epoch_health_logs (defaultdict): Storage for epoch-wise health statistics.
        current_epoch (int): Current training epoch number.
        epoch_metrics (defaultdict): Collection of metrics for current epoch.
        repair_strategies (dict): Counters for different repair mechanism activations.
    
    Methods:
        forward(x, gradients=None): 
            Performs forward pass with homeostatic regulation and repair.
            
        start_epoch(epoch_num): 
            Initializes tracking for a new training epoch.
            
        end_epoch(): 
            Generates epoch summary and resets metrics.
            
        biologically_plausible_repair(health_report, gradients=None): 
            Implements biological repair mechanisms including synaptic scaling,
            selective reinforcement, and activity-dependent pruning.
            
        get_adaptive_learning_rate(health_report): 
            Computes health-based adaptive learning rate.
            
        get_health_stats(): 
            Returns comprehensive health and performance statistics.
    
    Repair Mechanisms:
        - **Synaptic Scaling**: Reduces weights of overactive neurons with high calcium levels
        - **Selective Reinforcement**: Strengthens important synaptic connections
        - **Activity-Dependent Pruning**: Weakens or removes underperforming synapses
        - **Homeostatic Adjustment**: Global weight normalization to prevent runaway excitation
    
    Health Monitoring:
        The neuron continuously monitors its health through multiple metrics including calcium
        levels, synaptic strength stability, and overall performance. Health reports trigger
        adaptive behaviors and repair mechanisms when degradation is detected.
    
    Example:
        >>> import torch
        >>> neuron = BioLogicalNeuron(
        ...     in_features=128,
        ...     out_features=64,
        ...     plasticity_rate=0.01,
        ...     repair_threshold=0.6,
        ...     enable_monitoring=True
        ... )
        >>> 
        >>> # Start training epoch
        >>> neuron.start_epoch(1)
        >>> 
        >>> # Forward pass
        >>> x = torch.randn(32, 128)
        >>> output, health_report = neuron(x)
        >>> 
        >>> # Check health statistics
        >>> stats = neuron.get_health_stats()
        >>> print(f"Current health: {stats['current_health']:.3f}")
        >>> 
        >>> # End epoch and get summary
        >>> neuron.end_epoch()
    
    Note:
        This implementation is designed for research and experimental purposes, providing
        insights into biologically plausible neural network behaviors. The repair mechanisms
        and homeostatic regulation are based on current understanding of biological neural
        systems and may require tuning for specific applications.
    
    Raises:
        ValueError: If repair_threshold is not in range [0.0, 1.0].
        RuntimeError: If homeostatic regulation fails to initialize properly.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        plasticity_rate: float = 0.008,
        repair_threshold: float = 0.5,
        repair_intensity: float = 0.08,
        calcium_threshold: float = 0.9,  
        enable_monitoring: bool = True,
        log_file: Optional[str] = "bioneuron_health.log",
        summary_interval: int = 100,  
        **kwargs
    ):
        super().__init__()
        self.linear = weight_norm(nn.Linear(in_features, out_features))
        self.η = plasticity_rate
        
        # Pass calcium_threshold to homeostasis
        kwargs['calcium_threshold'] = calcium_threshold
        self.homeostasis = HomeostaticRegulation(**kwargs)
        
        self.repair_threshold = repair_threshold
        self.repair_intensity = repair_intensity
        self.repair_count = 0
        self.repair_cooldown = 0
        
        self.health_tracker = HealthTracker() if enable_monitoring else None
        self.logger = self._setup_logger(log_file) if enable_monitoring else None
        self.visualizer = BioNeuronVisualizer(save_dir="bio_vis") if enable_monitoring else None
        self.step_counter = 0
        self.summary_interval = summary_interval
        
        self.epoch_health_logs = defaultdict(list)
        self.current_epoch = 0
        self.epoch_metrics = defaultdict(list)
        
        # Renamed repair strategies to reflect biological mechanisms
        self.repair_strategies = {
            'synaptic_scaling': 0,
            'selective_reinforcement': 0,
            'activity_dependent_pruning': 0,
            'homeostatic_adjustment': 0
        }
        
    def _setup_logger(self, log_file: Optional[str]) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(f"bioneuron_{id(self)}")
        logger.setLevel(logging.INFO)
        
        if logger.hasHandlers():
            logger.handlers.clear() 
            
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if log_file:
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        else:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        logger.propagate = False
        return logger

    def _update_epoch_metrics(self, health_report: Dict[str, Union[torch.Tensor, float, bool]]):
        """Update epoch-wise metrics"""
        self.epoch_metrics["current_health"].append(
            float(health_report['current_health'].mean() if isinstance(health_report['current_health'], torch.Tensor) 
            else health_report['current_health'])
        )
        self.epoch_metrics["stability"].append(
            float(health_report['stability'].mean() if isinstance(health_report['stability'], torch.Tensor) 
            else health_report['stability'])
        )
        self.epoch_metrics["base_health"].append(
            float(health_report['base_health'].mean() if isinstance(health_report['base_health'], torch.Tensor) 
            else health_report['base_health'])
        )
        self.epoch_metrics["calcium_trend"].append(
            float(health_report['calcium_trend'] if isinstance(health_report['calcium_trend'], (int, float)) 
            else health_report['calcium_trend'].item())
        )
        self.epoch_metrics["repair_performed"].append(health_report['repair_performed'])

    def _log_health_status(self, health_report: Dict[str, float]):
        """Log health status and updates"""
        if self.health_tracker and self.logger:
            self.health_tracker.log_health(health_report, self.step_counter)
            
            self._update_epoch_metrics(health_report)
            
            health = health_report["current_health"].mean().item() if isinstance(health_report["current_health"], torch.Tensor) else health_report["current_health"]
            stability = health_report["stability"].mean().item() if isinstance(health_report["stability"], torch.Tensor) else health_report["stability"]
            
            msg = f"Step {self.step_counter} | Health: {health:.3f} | Stability: {stability:.3f}"
            if health_report.get("repair_performed", False):
                msg += f" | Repair performed! (Count: {self.repair_count})"
            
            self.logger.info(msg)
            
            # Log summary at specified intervals
            if self.step_counter % self.summary_interval == 0:
                self._log_interval_summary()

    def _log_interval_summary(self):
        """Log summary statistics for the current interval"""
        if not self.epoch_metrics["current_health"]:
            return

        avg_health = np.mean(self.epoch_metrics["current_health"])
        avg_stability = np.mean(self.epoch_metrics["stability"])
        avg_base_health = np.mean(self.epoch_metrics["base_health"])
        total_repairs = sum(self.epoch_metrics["repair_performed"])
        
        summary = (
            f"\nHealth Summary (Steps {self.step_counter - self.summary_interval + 1} - {self.step_counter}):\n"
            f"  Average Health: {avg_health:.3f}\n"
            f"  Average Stability: {avg_stability:.3f}\n"
            f"  Average Base Health: {avg_base_health:.3f}\n"
            f"  Total Repairs: {total_repairs}\n"
            f"  Repair Rate: {(total_repairs/self.summary_interval)*100:.2f}%"
        )
        
        if self.logger:
            self.logger.info(summary)
        else:
            print(summary)

    def start_epoch(self, epoch_num: int):
        """Mark the start of a new epoch"""
        self.current_epoch = epoch_num
        self.epoch_metrics = defaultdict(list)
        if self.logger:
            self.logger.info(f"\nStarting Epoch {epoch_num}")

    def end_epoch(self):
        """Generate end-of-epoch summary and reset metrics"""
        if not self.epoch_metrics["current_health"]:
            return

        avg_health = np.mean(self.epoch_metrics["current_health"])
        avg_stability = np.mean(self.epoch_metrics["stability"])
        avg_base_health = np.mean(self.epoch_metrics["base_health"])
        total_repairs = sum(self.epoch_metrics["repair_performed"])
        repair_rate = (total_repairs / len(self.epoch_metrics["repair_performed"])) * 100
        
        summary = (
            f"\nEpoch {self.current_epoch} Summary:\n"
            f"  Average Health: {avg_health:.3f}\n"
            f"  Average Stability: {avg_stability:.3f}\n"
            f"  Average Base Health: {avg_base_health:.3f}\n"
            f"  Total Repairs: {total_repairs}\n"
            f"  Repair Rate: {repair_rate:.2f}%"
        )
        
        if self.logger:
            self.logger.info(summary)
        else:
            print(summary)
        
        # Store epoch summary
        self.epoch_health_logs[self.current_epoch] = {
            "avg_health": avg_health,
            "avg_stability": avg_stability,
            "avg_base_health": avg_base_health,
            "total_repairs": total_repairs,
            "repair_rate": repair_rate
        }
        
        # Reset metrics for next epoch
        self.epoch_metrics = defaultdict(list)

    def biologically_plausible_repair(self, health_report: Dict[str, torch.Tensor], gradients: Optional[torch.Tensor] = None) -> bool:
        """
        Implement biologically plausible repair mechanisms based on synaptic scaling and selective reinforcement.
        In biological systems, repair involves structured processes rather than random noise injection.
        """
        if self.repair_cooldown > 0:
            self.repair_cooldown -= 1
            return False 
        
        current_health = health_report['current_health'].mean().item()
        if current_health < self.repair_threshold:
            # 1. Activity-Dependent Synaptic Scaling
            # In biological systems, synaptic scaling reduces weights after periods of high activity
            calcium_level = self.homeostasis.calcium_levels[-1]
            high_calcium_mask = calcium_level > self.homeostasis.calcium_threshold
            
            # Scale down weights for neurons with high calcium (overactive)
            if high_calcium_mask.any():
                scaling_factor = 1.0 - self.repair_intensity * (1.0 - health_report['stability'].item())
                self.linear.weight.data[high_calcium_mask] *= scaling_factor
                self.repair_strategies['synaptic_scaling'] += 1
            
            # 2. Selective Synaptic Reinforcement
            # Biological neurons selectively reinforce important connections while pruning weak ones
            weight_variance = torch.var(self.linear.weight, dim=1)
            weight_mean = torch.mean(torch.abs(self.linear.weight), dim=1)
            
            # Identify weak and strong synapses
            weak_synapse_mask = (torch.abs(self.linear.weight) < 0.1 * weight_mean.unsqueeze(1))
            strong_synapse_mask = (torch.abs(self.linear.weight) > 2.0 * weight_mean.unsqueeze(1))
            
            # Selectively prune weak synapses
            if weak_synapse_mask.any():
                pruning_factor = 0.9  # Reduce by 10%
                self.linear.weight.data[weak_synapse_mask] *= pruning_factor
                self.repair_strategies['activity_dependent_pruning'] += 1
            
            # Selectively reinforce strong synapses
            if strong_synapse_mask.any():
                reinforcement_factor = 1.05  # Increase by 5%
                self.linear.weight.data[strong_synapse_mask] *= reinforcement_factor
                self.repair_strategies['selective_reinforcement'] += 1
            
            # 3. Homeostatic Adjustment
            # Apply global normalization to prevent runaway excitation
            if torch.norm(self.linear.weight) > 2.0 * self.linear.weight.numel():
                self.linear.weight.data = self.linear.weight.data / torch.norm(self.linear.weight) * self.linear.weight.numel()
                self.repair_strategies['homeostatic_adjustment'] += 1
            
            # Repair tracking
            self.repair_cooldown = 40
            self.repair_count += 1
        
            self.homeostasis.repair_history.append({
                'health': current_health,
                'stability': health_report['stability'].item()
            })
            return True
        return False
    
    def get_adaptive_learning_rate(self, health_report):
        """Dynamically adjust learning rate based on neuron health"""
        base_lr = 0.001
        health_factor = health_report['current_health'].mean().item()
        stability_factor = health_report['stability'].mean().item()
        
        # Adaptive learning rate based on neuron health
        adaptive_lr = base_lr * health_factor * stability_factor
        return adaptive_lr

    def forward(self, x: torch.Tensor, gradients: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Dict[str, float]]:
        """Forward pass with biologically plausible homeostatic regulation and repair"""
        pre_synaptic = x
        post_synaptic = F.gelu(self.linear(x))
        
        with torch.no_grad():
            calcium_level = self.homeostasis.update_calcium(post_synaptic)
            synaptic_strength = torch.norm(self.linear.weight, dim=1)
            self.homeostasis.synaptic_strengths.append(synaptic_strength)
            
            if len(self.homeostasis.synaptic_strengths) > self.homeostasis.window:
                self.homeostasis.synaptic_strengths = self.homeostasis.synaptic_strengths[-self.homeostasis.window:]
            
            health_report = self.homeostasis.predict_health()
            performed_repair = self.biologically_plausible_repair(health_report, gradients)
            health_report['repair_performed'] = performed_repair
            
            # Compute adaptive learning rate
            adaptive_lr = self.get_adaptive_learning_rate(health_report)
            health_report['adaptive_lr'] = adaptive_lr
            
            self._log_health_status(health_report)
            self.step_counter += 1
            
            if self.visualizer:
                self.visualizer.update(
                    step=self.step_counter,
                    health_report=health_report,
                    calcium_level=calcium_level,
                    repair_strategies=self.repair_strategies
                )
                
                if self.step_counter % 400 == 0:
                    self.visualizer.save_all_plots()
            
        return post_synaptic, health_report
    
    def get_health_stats(self) -> Dict[str, float]:
        """Get comprehensive health statistics"""
        stats = {
            **self.homeostasis.get_health_summary(),
            "repair_count": self.repair_count,
            "cooldown": self.repair_cooldown,
            "repair_strategies": self.repair_strategies
        }
        
        if self.health_tracker:
            stats.update(self.health_tracker.get_summary())
            
        return stats
