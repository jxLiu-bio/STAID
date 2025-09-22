# STAID: iterative deconvolution in spatial transcriptomics via deep learning and graph Fourier transform

<img src="https://img.shields.io/badge/Platform-Linux-green"> <img src="https://img.shields.io/badge/Language-python3-green"> <img src="https://img.shields.io/badge/License-MIT-green"> <img src="https://img.shields.io/badge/notebooks-passing-green">

## System Requirements

### OS requirements

```STAID``` can run on Linux and Windows. The package has been tested on the following systems:

- Linux: Ubuntu 18.04 with NVIDIA RTX 4090 GPU.
- Windows: Windows Server 2022 with NVIDIA RTX A5000 GPU

```STAID``` requires python version >= 3.8.

All results in this tutorial were obtained using Python 3.9.0 with CUDA 12.4 on Linux.


## Installation Guide

### 1. Create a virtual environment

Users can install ```anaconda``` by following this tutorial if there is no [Anaconda](https://www.anaconda.com/).

Create a separate virtual environment:

```shell
conda create -n STAID_env python=3.9
conda activate STAID_env
```
### 2. Install ```STAID```

Install ```STAID``` from [Github](https://github.com/jxLiu-bio/STAID) and install it by:
```bash
git clone https://github.com/jxLiu-bio/STAID.git
cd STAID
pip install .
```

### 3. Install ```Torch```

Install the torch according to the tutorial in its official website: [Pytorch](https://pytorch.org/) to install the 
appropriate version for your system.

For example, to install PyTorch with CUDA 12.4 support, you can run:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 4. Test installation and set up Jupyter Notebook (optional)
To verify that STAID is installed correctly, start Python:
```bash
python
```
and 
```Python
import staid
```
If no error appears, the installation is successful.

Optionally, you can install Jupyter Notebook and create a kernel for your STAID environment:

```bash
conda install jupyter
python -m ipykernel install --user --name STAID_env --display-name STAID_env
```

## Tutorial
For a step-by-step tutorial, please refer to: [STAID tutorial](https://staid.readthedocs.io/en/latest/).
