[![Published](https://img.shields.io/badge/Published-Scientific%20Reports-blue)](https://doi.org/10.1038/s41598-025-09114-8)
[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41598--025--09114--8-blue)](https://doi.org/10.1038/s41598-025-09114-8)

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
- [Run Experiments](#run-experiments)
- [License](#license)
- [Contact](#contact)


## Overview

BioLogicalNeuron is a **peer-reviewed, published** neural network layer that incorporates biological principles of homeostasis and self-repair into deep learning architectures. **Published in Scientific Reports (Nature Portfolio, 2025)**, this implementation provides robust learning capabilities with built-in adaptation mechanisms, making it particularly suitable for complex, long-running applications and research in neuromorphic computing.

**Published Research:** This implementation is based on peer-reviewed research published in Scientific Reports, demonstrating significant improvements over state-of-the-art methods across multiple benchmark datasets.


## Publication

This research has been published in **Scientific Reports** (Nature Portfolio):
**Title:** Biologically inspired neural network layer with homeostatic regulation and adaptive repair mechanisms

If you use this work in your research, please cite our paper:
```bash
@article{hakim2025biological,
  title={Biologically inspired neural network layer with homeostatic regulation and adaptive repair mechanisms},
  author={Hakim, MD Azizul and Alam, Mohammad Ifazul},
  journal={Scientific Reports},
  volume={15},
  pages={9114},
  year={2025},
  publisher={Nature Publishing Group},
  doi={10.1038/s41598-025-09114-8}
}
```




## Experimental Results

Our comprehensive evaluation demonstrates the effectiveness of BioLogicalNeuron across various datasets and tasks. The following tables present our experimental results.

### Comprehensive Performance Analysis On Graph, Molecular, Protein structure, Node classification Datasets

| Dataset  | Fold | BioLogicalNeuron   | BioLogicalNeuron + Attn + Jumping | Previous SOTA | vs. SOTA |
|----------|------|--------------------|-----------------------------------|---------------|----------|
| AIDS     | 10   | **99.80 ± 0.004**  | **99.63 ± 0.007**                | 99.55         | +0.25    |
| HIV      | 5    | **96.95 ± 0.0013** | **97.15 ± 0.001**                | 96.86         | +0.29    |
| COX2     | 5    | 80.85 ± 0.017      | **82.71 ± 0.031**                | 82.6         | +0.11    |
| Protein  | 10   | **75.89 ± 0.04**   | **74.65 ± 0.045**                | 72.07         | +3.89    |
| DD       | 10   | 80.33 ± 0.06       | 76.94 ± 0.059                    | **95.67**     | -19.00   |
| MUTAG    | 10   | 83.33 ± 0.07       | 78.00 ± 0.12                     | **100.00**    | -22.00   |




### Performance Analysis for Image Datasets

| Dataset | Fold | Without BioLogicalNeuron | BioLogicalNeuron + Attention | Performance Gain (Base) |
|---------|------|-------------------------|----------------------------|----------------------|
| CIFAR-10 | 2 | 86.65 ± 0.064 | **90.42 ± 0.196** | +0.77 |
| MNIST | 2 | -- | **99.43 ± 0.002** | -- |
| Fashion-MNIST | 2 | -- | **93.27 ± 0.20** | -- |

### performance Analysis for Citation Datasets

| Dataset | Fold | BaseBioLayer | BioLayer + GAT Attention | Performance Gain | SOTA Result | Performance vs. Base | Performance vs. SOTA |
|---------|------|--------------|-------------------------|------------------|-------------|---------------------|---------------------|
| Cora | 15 | 74.53 | **88.56** | +14.02 | 90.26 | +14.02 | -1.70 |
| Citeseer | 10 | 72.37 | **76.87** | +3.67 | 82.07 | +3.67 | -5.20 |
| PubMed | 10 | 88.18 | **88.28** | +0.10 | 91.67 | +0.10 | -3.39 |


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

2. **Clone the Repository**
```bash
git clone https://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@github.com/logicsame/bio-logical-self-healing-neural-system.git

cd bio-logical-self-healing-neural-system
```

3. **Download and install Microsoft Visual C++ Redistributable**
```bash
https://aka.ms/vs/16/release/vc_redist.x64.exe
```


3. **Install Required Dependencies**
```bash
pip install -r requirements.txt
```


4. **Install the Package**
```bash
# Install in development mode
pip install -e .
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
python experiments/aids/full_architecture_aids_experiment.py --disable-monitoring
```

Run the only base bio layer experiment on the AIDS dataset with monitoring enabled:
```bash
python experiments/aids/aids_experiments_base_bio_layer.py --enable-monitoring
```

Run the only base bio layer experiment on the AIDS dataset without monitoring:
```bash
python experiments/aids/aids_experiments_base_bio_layer.py --disable-monitoring
```


### COX2 Dataset Experiments

Run the full architecture experiment on the COX2 dataset with monitoring enabled:
```bash
python experiments/cox2/full_architecture_cox2_experiment.py --enable-monitoring
```
Run the full architecture experiment on the COX2 dataset without monitoring :
```bash
python experiments/cox2/full_architecture_cox2_experiment.py --disable-monitoring 
```

Run the base bio layer experiment on the COX2 dataset with monitoring enabled:
```bash
python experiments/cox2/base_bio_layer_cox2_experiment.py --enable-monitoring
```
Run the base bio layer experiment on the COX2 dataset without monitoring :
```bash
python experiments/cox2/base_bio_layer_cox2_experiment.py --disable-monitoring 
```

### HIV Dataset Experiments

Run the full architecture experiment on the HIV dataset with monitoring enabled:
```bash
python experiments/HIV/full_architecture_bio_layer_hiv_experiment.py --enable-monitoring
```

Run the full architecture experiment on the HIV dataset without monitoring:
```bash
python experiments/HIV/full_architecture_bio_layer_hiv_experiment.py --disable-monitoring 
```

Run the base bio layer experiment on the HIV dataset with monitoring enabled:
```bash
python experiments/HIV/base_bio_layer_hiv_experiment.py --enable-monitoring
```

Run the base bio layer experiment on the HIV dataset without monitoring:
```bash
python experiments/HIV/base_bio_layer_hiv_experiment.py --disable-monitoring
```

### PROTEINS Dataset Experiments

Run the full architecture experiment on the PROTEINS dataset with monitoring enabled:
```bash
python experiments/PROTEINS/full_architecture_PROTEINS_experiment.py --enable-monitoring
```

Run the full architecture experiment on the PROTEINS dataset without monitoring:
```bash
python experiments/PROTEINS/full_architecture_PROTEINS_experiment.py --disable-monitoring 
```

Run the base bio layer experiment on the PROTEINS dataset with monitoring enabled:
```bash
python experiments/PROTEINS/base_bio_protrein_experiment.py --enable-monitoring
```

Run the base bio layer experiment on the PROTEINS dataset without monitoring:
```bash
python experiments/PROTEINS/base_bio_protrein_experiment.py --disable-monitoring
```


### D&D Dataset Experiments

Run the full architecture experiment on the D&D dataset with monitoring enabled:
```bash
python experiments/D&D/full_architecture_dd_experiment.py --enable-monitoring
```

Run the full architecture experiment on the D&D dataset without monitoring:
```bash
python experiments/D&D/full_architecture_dd_experiment.py --disable-monitoring 
```

Run the base bio layer experiment on the D&D dataset with monitoring enabled:
```bash
python experiments/D&D/base_bio_layer_dd_experiment.py --enable-monitoring
```

Run the base bio layer experiment on the D&D dataset without monitoring:
```bash
python experiments/D&D/base_bio_layer_dd_experiment.py --disable-monitoring
```


### Mutag Dataset Experiments

Run the full architecture experiment on the Mutag dataset with monitoring enabled:
```bash
python experiments/mutag/full_architecture_mutag_experiment.py --enable-monitoring
```

Run the full architecture experiment on the Mutag dataset without monitoring:
```bash
python experiments/mutag/full_architecture_mutag_experiment.py --disable-monitoring 
```

Run the base bio layer experiment on the Mutag dataset with monitoring enabled:
```bash
python experiments/mutag/base_bio_layer_mutag_experiment.py --enable-monitoring
```

Run the base bio layer experiment on the Mutag dataset without monitoring:
```bash
python experiments/mutag/base_bio_layer_mutag_experiment.py --disable-monitoring
```

### Cora Dataset Experiments

Run the full architecture experiment on the Cora dataset:
```bash
python experiments/cora/cora_experiment.py --enable-monitoring
```

Run only with base_bio layer experiment on the Cora dataset:
```bash
python experiments/cora/base_bio_layer_cora_experiment.py --enable-monitoring
```



### Citeseer Dataset Experiment

Run the full architecture experiment on the Citeseer dataset:
```bash
python experiments/CiteSeer/citeseer_experiment_with_full_architecture.py --enable-monitoring
```

Run the only base bio layer experiment on the Citeseer dataset:
```bash
python experiments/CiteSeer/citeseer_experiment_with_full_architecture.py --enable-monitoring
```

### Pubmed Dataset Experiment

Run the full architecture experiment on the Pubmed dataset:
```bash
python experiments/PubMed/full_architecture_pubmed_experiment.py --enable-monitoring
```
Run the full architecture experiment on the Pubmed dataset:
```bash
python experiments/PubMed/base_bio_layer_pubmed_experiment.py --enable-monitoring
```
*Note: For the citation dataset experiment, the biological layer's behavior visualization will not be generated, but the biological layer's log file will be generated.*

### Cifar10 Dataset Experiments

Run the full architecture experiment on the Cifar10 dataset with monitoring :
```bash
python experiments/ciraf10/full_architecture_cifar10_experiment.py --enable-monitoring
```
Run only with attention mechanism on cifar10 dataset 
```bash
python experiments/ciraf10/with_attention_cifar10_experiment.py
```

## FashionMNIST dataset Experiments

Run only with attention mechanism on FashionMNIST dataset 
```bash
python experiments/Fashion mnist/full_architecture_fashionmnist_experiment.py --enable-monitoring
```
### MNIST Dataset Experiment

Run only with attention mechanism on FashionMNIST dataset 
```bash
python experiments/mnist/full_architecture_mnist_experiment.py --enable-monitoring
```

### Notes
- The `--enable-monitoring` flag is optional and disabled by default
- Experiment results are automatically logged and saved
- The experiment results, Bio Layer log file during training and  monitoring data (if enabled) will be saved in the respective output directories under each experiment folder.

- For reproducibility, use the same random seed across experiments
- Monitor GPU memory usage when running large-scale experiments




## License

MIT License. See [LICENSE](LICENSE) for details.


## Contact

- **Lead Author**: MD. Azizul Hakim
- **Email**: azizulhakim8291@gmail.com
- **Co-Author**: Mohammad Ifazul Alam
- **Email**: efazulalam05@gmail.com
- **Published Paper**: [Scientific Reports](https://doi.org/10.1038/s41598-025-09114-8)

