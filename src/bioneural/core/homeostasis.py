# Modified Homeostasis Implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from ..utils.logging import HealthLogger
from typing import Dict, List, Optional

class HomeostaticRegulation:
    """Biologically accurate Homeostatic Regulation System with improved monitoring"""
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
