"""
BioNeural - Biological Self-Healing Neuron System
"""

__version__ = "1.0.0"
__author__ = "logicsame"
__email__ = "useforprofessional@gmail.com"

from .core.biololgicallayer import BioLogicalNeuron
from .metrics.healthtracker import HealthTracker
from .visualization.biosysvisualization2 import BioNeuronVisualizer

__all__ = [
    'BioLogicalNeuron', 
    'HealthTracker', 
    'BioNeuronVisualizer'
]