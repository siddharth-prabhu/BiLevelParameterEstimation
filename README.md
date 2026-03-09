# Bi-Level optimization for parameter estimation of differential equations using interpolation

## Overview

This project is based on our paper (https://arxiv.org/abs/2506.00720)

## Features

- Bi-Level optimizaiton (inner convex + outer nonconvex) for parameter estimation of differential equations
- Differentiable optimization in JAX
- Differentiable cubic spline interpolatin in JAX


## Installation

Clone the repository:

```
git clone https://github.com/siddharth-prabhu/BiLevelParameterEstimation.git
cd BiLevelParameterEstimation
python pe_calcium.py
```

## Citation

If you found this code useful in academic research, please cite:

```bibtex
@misc{prabhu2025bileveloptimizationparameterestimation,
      title={Bi-Level optimization for parameter estimation of differential equations using interpolation}, 
      author={Siddharth Prabhu and Srinivas Rangarajan and Mayuresh Kothare},
      year={2025},
      eprint={2506.00720},
      archivePrefix={arXiv},
      primaryClass={eess.SY},
      url={https://arxiv.org/abs/2506.00720}, 
}
```