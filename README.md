# BioLogicalNeuron Layer

A sophisticated biological neural network layer implementing advanced homeostatic regulation and self-repair mechanisms, designed for robust and adaptive deep learning systems.

## Table of Contents
- [Overview](#overview)
- [Experimental Results](#experimental-results)
- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
- [Architecture](#architecture)
- [Monitoring System](#monitoring-system)
- [Research Applications](#research-applications)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Overview

BioLogicalNeuron is a novel neural network layer that incorporates biological principles of homeostasis and self-repair into deep learning architectures. This implementation provides robust learning capabilities with built-in adaptation mechanisms, making it particularly suitable for complex, long-running applications and research in neuromorphic computing.

## Experimental Results

Our comprehensive evaluation demonstrates the effectiveness of BioLogicalNeuron across various datasets and tasks. The following tables present our experimental results.

### Comprehensive Performance Analysis On Graph, Molecular, Protein structure, Node classification Datasets

| Dataset | Fold | BioLogicalNeuron | BioLogicalNeuron + Attn + Jumping | Previous SOTA | vs. Base | vs. SOTA |
|---------|------|------------------|-----------------------------------|---------------|-----------|-----------|
| AIDS | 10 | **99.80 ± 0.004** | **99.63 ± 0.007** | 99.55 | +0.10 | +0.15 |
| HIV | 5 | **96.95 ± 0.0013** | **97.15 ± 0.001** | 96.86 | +0.20 | +0.29 |
| COX2 | 5 | 79.57 ± 0.017 | **83.25 ± 0.031** | 82.86 | +3.68 | +0.39 |
| Protein | 10 | **75.89 ± 0.04** | **74.65 ± 0.045** | 72.07 | -1.24 | +3.89 |
| DD | 10 | 80.33 ± 0.06 | 76.94 ± 0.059 | **95.67** | -4.00 | -19.00 |
| MUTAG | 10 | 83.33 ± 0.07 | 78.00 ± 0.12 | **100.00** | -5.33 | -22.00 |
| Cora | 15 | -- | **90.48 ± 0.019** | 90.16 | -- | +0.32 |
| Citeseer | 10 | -- | 78.58 ± 0.02 | **82.07** | -- | -4.00 |
| PubMed | 10 | -- | 87.88 ± 0.01 | **91.64** | -- | -4.64 |



### Performance Analysis for Image Datasets

| Dataset | Fold | Without BioLogicalNeuron | BioLogicalNeuron + Attention | Performance Gain (Base) |
|---------|------|-------------------------|----------------------------|----------------------|
| CIFAR-10 | 2 | 86.43 ± 0.064 | **90.42 ± 0.196** | +0.33 |
| MNIST | 2 | -- | **99.43 ± 0.002** | -- |
| Fashion-MNIST | 2 | -- | **93.27 ± 0.20** | -- |



**Key Findings:**
- Achieved state-of-the-art performance on multiple molecular datasets (AIDS, HIV, COX2)
- Significant improvements on protein structure prediction tasks
- Competitive performance on standard computer vision benchmarks
- Mixed results on graph classification tasks, with room for improvement on DD and MUTAG datasets

## Installation

### System Requirements
- Python 3.7 or higher
- CUDA-capable GPU (recommended)
- 4GB RAM minimum
- 1GB free disk space

### Dependencies
```
torch>=1.9.0
numpy>=1.19.0
matplotlib>=3.3.0
logging>=0.5.1.2
typing>=3.7.4
dataclasses>=0.6
pandas>=2.0.0
```

### Step-by-Step Installation
1. **Create a Virtual Environment (Optional)**
```bash
# Using venv
python -m venv biolayer-env

# Activate the environment
# On Windows
biolayer-env\Scripts\activate
# On Unix or MacOS
source biolayer-env/bin/activate
```

2. **Install Required Dependencies**
```bash
pip install -r requirements.txt
```

3. **Clone the Repository**
```bash
git clone https://github.com/yourusername/Bioneural.git
cd Bioneural
```

4. **Install the Package**
```bash
# Install in development mode
pip install -e .

# Or install directly
pip install .
```

5. **Verify Installation**
```python
import torch
from bioneural.core.biololgicallayer import BioLogicalNeuron


# Test installation
layer = BioLogicalNeuron(in_features=64, out_features=32)
x = torch.randn(10, 64)
output, health = layer(x)
print("Installation successful!")
```

### Docker Installation
```bash
# Build the Docker image
docker build -t biolayer .

# Run the container
docker run -it --gpus all biolayer
```

## Features

### Core Components
- Homeostatic regulation system
- Adaptive repair mechanisms
- Real-time health monitoring
- Dynamic learning rate adjustment
- Comprehensive logging system
- Advanced visualization tools

### Technical Specifications
- Multi-strategy repair system
- Calcium-based homeostasis
- Adaptive noise injection
- Targeted repair zones
- Stability-aware learning

## Usage

### Basic Implementation
```python
from Bioneural.core import BioLogicalNeuron
import torch

# Initialize layer
bio_layer = BioLogicalNeuron(
    in_features=64,
    out_features=32,
    plasticity_rate=0.008,
    repair_threshold=0.5,
    enable_monitoring=True
)

# Forward pass
input_data = torch.randn(32, 64)
output, health_report = bio_layer(input_data)
```

### Advanced Configuration
```python
bio_layer = BioLogicalNeuron(
    in_features=64,
    out_features=32,
    plasticity_rate=0.008,
    repair_threshold=0.5,
    repair_intensity=0.08,
    enable_monitoring=True,
    log_file="custom_log.log",
    summary_interval=100
)
```


## Monitoring System

### Health Tracking
```python
# Enable monitoring
bio_layer = BioLogicalNeuron(
    in_features=64,
    out_features=32,
    enable_monitoring=True
)

# Access health statistics
health_stats = bio_layer.get_health_stats()
```

### Visualization System
```python
# Visualizations are automatically saved to 'bio_vis' directory
bio_layer.visualizer.save_all_plots()
```

### Logging System
```python
# Configure logging
bio_layer = BioLogicalNeuron(
    in_features=64,
    out_features=32,
    enable_monitoring=True,
    log_file="health_metrics.log"
)
```

## Research Applications

### Suitable for:
- Neuromorphic computing research
- Stability analysis
- Homeostatic regulation studies
- Neural plasticity investigation

## Troubleshooting

### Common Issues

1. **Installation Failures**
```bash
# Update pip
pip install --upgrade pip

# Clear pip cache
pip cache purge
```

2. **CUDA Issues**
- Ensure CUDA toolkit matches PyTorch version
- Verify GPU compatibility

3. **Memory Issues**
- Reduce batch size
- Enable gradient checkpointing

## Run Experiments

After installing the BioLogicalNeuron layer, follow these steps to run experiments:

1. Open cmd or bash on cloned reprository folder

2. The experiments can be run with different configurations using command-line arguments. Here are the available experiment options:

### AIDS Dataset Experiments

Run the full architecture experiment on the AIDS dataset with monitoring enabled:
```bash
python experiments/aids/full_architecture_aids_experiment.py --enable-monitoring
```

Run the full architecture experiment on the AIDS dataset without monitoring:
```bash
python experiments/aids/full_architecture_aids_experiment.py
```

Run the only baseitecture experiment on the AIDS dataset with monitoring enabled:
```bash
python experiments/aids/full_architecture_aids_experiment.py --enable-monitoring
```

Run the full architecture experiment on the AIDS dataset without monitoring:
```bash
python experiments/aids/full_architecture_aids_experiment.py
```


### COX2 Dataset Experiments

Run the full architecture experiment on the COX2 dataset with monitoring enabled:
```bash
python experiments/cox2/full_architecture_cox2_experiment.py --enable-monitoring
```

### Additional Dataset Experiments

Based on the performance analysis tables in the documentation, experiments can also be run on other datasets such as HIV, COX2, Protein, and DD. The corresponding experiment scripts would follow a similar pattern:

```bash
python experiments/<dataset_name>/full_architecture_<dataset_name>_experiment.py [--enable-monitoring]
```

### Experiment Configuration

You can customize various parameters for the experiments:

```bash
python experiments/aids/full_architecture_aids_experiment.py \
    --enable-monitoring \
    --batch-size 32 \
    --learning-rate 0.001 \
    --epochs 100 \
    --plasticity-rate 0.008 \
    --repair-threshold 0.5
```

The experiment results and monitoring data (if enabled) will be saved in the respective output directories under each experiment folder.

### Notes
- The `--enable-monitoring` flag is optional and disabled by default
- Experiment results are automatically logged and saved
- For reproducibility, use the same random seed across experiments
- Monitor GPU memory usage when running large-scale experiments


## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/Bioneural.git

# Set up development environment
python -m venv dev-env
source dev-env/bin/activate
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@article{biologicalneuron2024,
    title={BioLogicalNeuron: A Biologically-Inspired Self-Regulating Neural Network Layer},
    author={Your Name},
    journal={Nature Machine Intelligence},
    year={2024},
    volume={},
    number={},
    pages={},
    publisher={Nature Publishing Group},
    doi={}
}
```

## Contact

- **Main Developer**: [MD. Azizul Hakim]
- **Email**: your.email@institution.edu
- **Research Group**: [Institution Name]
- **Project Website**: https://github.com/yourusername/Bioneural

## Acknowledgments

This work was supported by [funding sources/institutions].