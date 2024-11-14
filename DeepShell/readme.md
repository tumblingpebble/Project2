#Deep GMM with TURTLE's Method
This project implements a deep Gaussian Mixture Model (GMM) clustering using TURTLE's method with representations.  The goal is to cluster CIFAR-10 data and evaluate the clustering performance using several metrics.
# Setup
##1. Install required dependencies:
```pip install -r requirements.txt```

##2. Precompute Representations and ground truth Labels
```python precompute_representations.py --dataset cifar10 --phis clipvitL14```
```python precompute_representations.py --dataset cifar10 --phis dinov2```
```python precompute_labels.py --dataset cifar10```

##3. Run the project
```python run_deepshell.py```
