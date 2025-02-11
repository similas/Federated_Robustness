# Robust-FL: A Framework for Analyzing Attack Strategies and Robustness in Federated Learning

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*An elegant framework for exploring the vulnerabilities and resilience of federated learning systems through systematic attack implementations.*

[Key Features](#key-features) • [Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Contributing](#contributing)

</div>

## About The Project

Robust-FL Framework provides researchers and practitioners with a comprehensive toolkit for investigating the security aspects of federated learning systems. By implementing three distinct attack strategies – label flipping, backdoor injection, and model replacement – this framework enables systematic analysis of federated learning vulnerabilities and model resilience.

Using the CIFAR-10 dataset and ResNet18 architecture as our foundation, we offer a clean, modular implementation that supports both research exploration and educational understanding of federated learning security challenges.

## Key Features

- **Clean Implementation**: Pure PyTorch implementation of federated learning with ResNet18 on CIFAR-10
- **Multiple Attack Strategies**: 
  - Label Flipping: Strategic manipulation of training labels
  - Backdoor: Subtle injection of targeted patterns
  - Model Replacement: Direct corruption of model weights
- **Comprehensive Analysis**: Built-in evaluation tools with visual insights
- **Mac Optimization**: Integrated support for Metal Performance Shaders (MPS)

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/fl-attack.git

# Navigate to the project directory
cd fl-attack

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
# Run with clean and malicoius clients
python main.py
```

## Contributing

Contributions make the open-source community thrive. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- Inspired by advances in federated learning security research
- Built with PyTorch and torchvision
- Tested and refined with help from the FL research community

<div align="center">

Made with ❤️ for the federated learning community by Ali Sadr

</div>
