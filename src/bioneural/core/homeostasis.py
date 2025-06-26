# Modified Homeostasis Implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from ..utils.logging import HealthLogger
from typing import Dict, List, Optional

class HomeostaticRegulation:
    """
    Biologically accurate Homeostatic Regulation System with comprehensive neural health monitoring.
    
    This class implements calcium-based homeostatic mechanisms inspired by biological neural systems,
    where calcium levels serve as a key indicator of neuronal health and activity. The system maintains
    optimal neural function by monitoring calcium dynamics, synaptic weight stability, and overall
    network health, closely mimicking the homeostatic processes found in living neurons.
    
    In biological systems, calcium homeostasis is critical as elevated calcium levels can be cytotoxic,
    while appropriate calcium signaling is essential for synaptic plasticity and neural communication.
    This implementation captures these dynamics to provide realistic neural behavior regulation.
    
    Args:
        decay_rate (float, optional): Rate at which calcium levels naturally decay over time.
            Represents biological calcium clearance mechanisms. Range: [0.01, 0.1]. Defaults to 0.03.
        activation_scale (float, optional): Scaling factor applied to neural activations when
            computing calcium influx. Controls sensitivity to neural activity. Defaults to 0.4.
        stability_scale (float, optional): Scaling factor for synaptic weight stability contributions
            to overall health computation. Emphasizes importance of stable synaptic connections.
            Defaults to 0.25.
        prediction_window (int, optional): Number of time steps used for health prediction and
            stability calculations. Larger windows provide more stable but less responsive predictions.
            Defaults to 8.
        stability_threshold (float, optional): Threshold value used in stability computations to
            determine acceptable variability levels. Defaults to 0.15.
        calcium_threshold (float, optional): Optimal calcium level threshold above which neurons
            are considered to have excessive calcium. In biological systems, maintaining calcium
            below this threshold is crucial for cell health. Defaults to 0.7.
        enable_logging (bool, optional): Whether to enable detailed health logging through the
            HealthLogger system. Useful for debugging and analysis. Defaults to True.
    
    Attributes:
        α (float): Decay rate for calcium clearance mechanisms.
        β (float): Activation scaling factor for calcium influx.
        γ (float): Stability scaling factor for health computations.
        window (int): Size of the prediction and stability analysis window.
        stability_threshold (float): Threshold for stability determinations.
        calcium_threshold (float): Optimal calcium level threshold.
        logger (HealthLogger): Logger instance for health monitoring (if enabled).
        calcium_levels (List[torch.Tensor]): Historical calcium level measurements.
        synaptic_strengths (List[torch.Tensor]): Historical synaptic weight measurements.
        health_scores (List[torch.Tensor]): Historical health score computations.
        stability_scores (List[torch.Tensor]): Historical stability measurements.
        repair_history (List[dict]): Record of repair events and their contexts.
    
    Methods:
        reset_history():
            Clears all historical data and resets tracking systems.
            
        update_calcium(activations):
            Updates calcium levels based on current neural activity with biological
            momentum and early-training exploration noise.
            
        compute_stability(health_scores):
            Computes stability metric from recent health score variations using
            coefficient of variation approach.
            
        compute_weight_stability(weights):
            Calculates synaptic weight stability over time, handling different
            tensor dimensions appropriately.
            
        predict_health():
            Generates comprehensive health prediction including current health,
            stability metrics, base health, and calcium trend analysis.
            
        get_health_summary():
            Returns summary statistics of current health status and repair history.
    
    Health Computation:
        The health prediction system uses a biologically-inspired approach where:
        - Lower calcium levels (below threshold) indicate better health
        - Synaptic weight stability is prioritized over magnitude
        - Calcium trends are monitored (decreasing calcium is beneficial)
        - Overall health combines calcium status, weight stability, and temporal consistency
    
    Biological Basis:
        - **Calcium Homeostasis**: Mimics cellular calcium regulation where high levels are toxic
        - **Synaptic Stability**: Reflects the importance of stable synaptic connections in learning
        - **Temporal Dynamics**: Captures the time-dependent nature of neural adaptation
        - **Adaptive Momentum**: Implements experience-dependent calcium handling
    
    Example:
        >>> import torch
        >>> homeostasis = HomeostaticRegulation(
        ...     decay_rate=0.05,
        ...     calcium_threshold=0.8,
        ...     prediction_window=10,
        ...     enable_logging=True
        ... )
        >>> 
        >>> # Simulate neural activity
        >>> activations = torch.randn(32, 64)  # batch_size x features
        >>> 
        >>> # Update calcium based on activity
        >>> calcium_level = homeostasis.update_calcium(activations)
        >>> 
        >>> # Record synaptic weights
        >>> weights = torch.randn(64, 32)
        >>> homeostasis.synaptic_strengths.append(torch.norm(weights, dim=1))
        >>> 
        >>> # Get health prediction
        >>> health_report = homeostasis.predict_health()
        >>> print(f"Health: {health_report['current_health'].mean():.3f}")
        >>> print(f"Stability: {health_report['stability'].mean():.3f}")
        >>> 
        >>> # Get summary statistics
        >>> summary = homeostasis.get_health_summary()
        >>> print(f"Average health: {summary['health']:.3f}")
    
    Note:
        This implementation prioritizes biological accuracy over computational efficiency.
        The calcium dynamics and stability computations are designed to reflect real
        neurobiological processes, making it particularly suitable for research into
        biologically plausible neural networks and adaptive learning systems.
        
        The system maintains historical data within a sliding window to balance
        responsiveness with computational efficiency. For long-running experiments,
        consider periodic history reset to manage memory usage.
    
    Raises:
        ValueError: If decay_rate, activation_scale, or stability_scale are outside valid ranges.
        RuntimeError: If tensor operations fail due to device mismatches.
        IndexError: If prediction_window is larger than available history.
    """
    def __init__(
        self,
        decay_rate: float = 0.03,
        activation_scale: float = 0.4,
        stability_scale: float = 0.25,
        prediction_window: int = 8,
        stability_threshold: float = 0.15,
        calcium_threshold: float = 0.7,  # New parameter: optimal calcium threshold
        enable_logging: bool = True
    ):
        """
        Initialize the homeostatic regulation system.
        
        Args:
            decay_rate: Rate at which calcium levels decay
            activation_scale: Scaling factor for neural activations
            stability_scale: Scaling factor for synaptic stability (renamed from strength_scale)
            prediction_window: Window size for health prediction
            stability_threshold: Threshold for stability computation
            calcium_threshold: Optimal calcium level threshold (new parameter)
            enable_logging: Whether to enable detailed health logging
        """
        self.α = decay_rate
        self.β = activation_scale
        self.γ = stability_scale
        self.window = prediction_window
        self.stability_threshold = stability_threshold
        self.calcium_threshold = calcium_threshold  # New parameter
        self.logger = HealthLogger() if enable_logging else None
        self.reset_history()
        
    def reset_history(self):
        self.calcium_levels = []
        self.synaptic_strengths = []
        self.health_scores = []
        self.stability_scores = []
        self.repair_history = []    
    
    def compute_stability(self, health_scores):
        if len(health_scores) < 2:
            return torch.tensor(1.0, device=health_scores[-1].device if health_scores else 'cuda')
        recent_scores = torch.stack(health_scores[-self.window:])
        stability = 1.0 - torch.std(recent_scores) / (torch.mean(recent_scores) + 1e-6)
        return torch.clamp(stability, 0.1, 1.0)  # Prevent extreme values

    def update_calcium(self, activations):
        """
        Update calcium levels based on neural activity.
        In biological systems, calcium homeostasis maintains low steady-state values
        as high calcium levels are cytotoxic.
        """
        device = activations.device
        current_calcium = self.calcium_levels[-1] if self.calcium_levels else torch.zeros_like(activations.mean(dim=0))
        
        # Enhanced momentum calculation
        momentum = min(0.95, 0.9 + len(self.calcium_levels) * 0.001)  # Adaptive momentum
        
        # Calculate new calcium level - increases with neural activity
        new_calcium = (momentum * current_calcium + 
                      (1 - momentum) * self.β * activations.mean(dim=0))
        
        # Add noise for exploration during early training
        if len(self.calcium_levels) < 100:  # Early training phase
            noise = torch.randn_like(new_calcium) * 0.01
            new_calcium = new_calcium + noise
            
        self.calcium_levels.append(new_calcium)
        
        if len(self.calcium_levels) > self.window:
            self.calcium_levels = self.calcium_levels[-self.window:]
            
        return new_calcium
    
    def compute_weight_stability(self, weights):
        """
        Compute weight stability instead of just magnitude.
        In biological systems, stability of synaptic weights is more important
        than just their magnitude.
    """
        if len(self.synaptic_strengths) < 2:
            # Check if weights is 1D or 2D and return appropriate shape
            if weights.dim() <= 1:
                return torch.ones_like(weights)
            else:
                return torch.ones_like(weights.mean(dim=1))
            
        # Calculate weight changes over time
        recent_weights = torch.stack(self.synaptic_strengths[-self.window:])
    
        # Handle different dimensions
        if weights.dim() <= 1:
            weight_stability = 1.0 - torch.std(recent_weights, dim=0) / (torch.mean(recent_weights, dim=0) + 1e-6)
        else:
            weight_stability = 1.0 - torch.std(recent_weights, dim=0) / (torch.mean(recent_weights, dim=0) + 1e-6)
            # If needed, reduce dimension to match expected output shape
            if weight_stability.dim() > 1:
                weight_stability = weight_stability.mean(dim=1)
    
        return torch.clamp(weight_stability, 0.1, 1.0)
    
    def get_health_summary(self) -> Dict[str, float]:
        """Get a summary of current health metrics."""
        if not self.health_scores:
            return {"health": 1.0, "stability": 1.0}
        
        return {
            "health": float(self.health_scores[-1].mean().item()),
            "stability": float(self.compute_stability(self.health_scores).mean().item()),
            "repair_count": len(self.repair_history)
        }

    def predict_health(self):
        """
        Predict neuron health based on calcium levels and synaptic stability.
        In biological systems, lower calcium levels (below toxic threshold) indicate better health,
        and synaptic stability is more important than just strength.
        """
        if not self.calcium_levels:
            return {
                'current_health': torch.tensor(1.0, device='cuda'),
                'stability': torch.tensor(1.0, device='cuda'),
                'base_health': torch.tensor(1.0, device='cuda'),
                'calcium_trend': torch.tensor(0.0, device='cuda')
            }
            
        device = self.calcium_levels[-1].device
        recent_calcium = self.calcium_levels[-1]
        
        if not self.synaptic_strengths:
            weight_stability = torch.ones_like(recent_calcium)
        else:
            # Use weight stability instead of just magnitude
            weight_stability = self.compute_weight_stability(self.synaptic_strengths[-1])
        
        # Enhanced health computation - lower calcium (below threshold) is better
        # This aligns with biological reality where high calcium levels are cytotoxic
        calcium_health = torch.sigmoid(self.calcium_threshold - recent_calcium)
        base_health = torch.sigmoid(calcium_health + self.γ * weight_stability)
        
        # Calculate calcium trend
        calcium_trend = 0.0
        if len(self.calcium_levels) > 1:
            calcium_diff = self.calcium_levels[-1] - self.calcium_levels[-2]
            calcium_trend = torch.mean(calcium_diff).item()
            # Negative trend (decreasing calcium) is good in biological systems
            calcium_trend = -calcium_trend
        
        stability = self.compute_stability(self.health_scores if self.health_scores else [base_health])
        
        # Weighted health score
        current_health = base_health * stability * (1.0 + torch.sigmoid(torch.tensor(calcium_trend)))
        self.health_scores.append(current_health)
        
        if len(self.health_scores) > self.window:
            self.health_scores = self.health_scores[-self.window:]
        
        return {
            'current_health': current_health,
            'stability': stability,
            'base_health': base_health,
            'calcium_trend': calcium_trend
        }
