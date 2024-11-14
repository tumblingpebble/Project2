# DeepShell

This project implements a **deep Gaussian Mixture Model (GMM) clustering** using TURTLE's method with foundational model representations. The goal is to cluster CIFAR-10 data and evaluate clustering performance with various metrics.

## Setup

### 1. Install Required Dependencies

Install the required packages by running:

```bash
pip install -r requirements.txt
```

### 2. Precompute Representations and Ground Truth Labels

Run the following commands to precompute representations and ground truth labels:

```bash
python precompute_representations.py --dataset cifar10 --phis clipvitL14
python precompute_representations.py --dataset cifar10 --phis dinov2
python precompute_labels.py --dataset cifar10
```

### 3. Run the Project

To execute the main clustering pipeline, simply run:

```bash
python run_deepshell.py
```

This command will handle training, evaluation, and saving of results automatically.
