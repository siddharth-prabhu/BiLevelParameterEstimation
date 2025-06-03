# Bi-Level optimization for parameter estimation of differential equations using interpolation

## Overview

This project is based on our paper (https://arxiv.org/abs/2506.00720)

## Features

- BiLevel optimization in JAX
- Accelerated sensitivity calculation across ODE's using interpolation
- Accelearted derivative calculations across the KKT system by exploiting problem formulation
  
## Installation

Clone the repository:

```
git clone https://github.com/siddharth-prabhu/BiLevelParameterEstimation.git
cd BiLevelParameterEstimation
python pe_calcium.py # For parameter estimation of Calcium Ion system 
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
