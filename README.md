# Brownian Bug Simulation Module

A Python package for simulating and analyzing the **Competitive Brownian Bug Model**, a stochastic interacting particle system exhibiting birth, death, diffusion, and density-dependent competition.

The package provides tools for

- Brownian bug simulations
- Monte Carlo ensemble simulations
- Master equation analysis
- Statistical analysis of population dynamics


---

## Scientific Background

The Competitive Brownian Bug Model is a stochastic interacting particle system in which particles

- diffuse via Brownian motion,
- reproduce,
- die,
- interact through local density-dependent competition.

The birth and death rates of the *i*th particle are

```
b_i = max(0, b₀ − b₁N_R)
d_i = d₀ + d₁N_R
```

where

- `N_R` is the number of neighboring particles within a radius `R`,
- `b₀` and `d₀` are intrinsic birth/death rates,
- `b₁` and `d₁` are competition parameters.

Under the homogeneous approximation, the package also provides tools to study the corresponding Markov chain and Master equation describing the population dynamics.

---

## Features


- Brownian Bug Monte Carlo simulations
- Ensemble averaging
- Master Equation solver
- Statistical observables

  - Mean population
  - Variance
  - Probability distributions

- Parameter sweeps
- Plot generation
- Configurable simulation parameters

---

## Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Brownian-Bug-Simulation-Module-.git
```

Install locally

```bash
pip install .
```

or

```bash
pip install -e .
```

for development.

---

## Package Structure

```
src/
└── bug_engine/
    ├── __init__.py
    ├── engine.py
    ├── ensemble.py
    ├── Master_equation.py
    ├── stat.py
    ├── plot.py
    └── config.py
```

---

## Example

```python
from bug_engine import *

# Example simulation
# (example code coming soon)
```

---

## Applications

This package can be used to investigate

- Active matter
- Population dynamics
- Nonequilibrium statistical mechanics
- Stochastic birth-death processes
- Mean-field approximations
- Master equation dynamics

---

## Project Background

This software was developed during my Summer Research Project under the supervision of

**Dr. Emilio Hernández García**

Institute for Cross-Disciplinary Physics and Complex Systems (IFISC), Spain.

The implementation accompanies the theoretical development presented in the internship report on the Competitive Brownian Bug Model, including simulations, homogeneous-state approximation, and Master Equation formulation. 
---

## Future Work

- Gillespie algorithm implementation
- Spatial correlation functions
- Cluster analysis

---

## License

MIT License

---

## Author

**Karmukilan Somasundaram**

Integrated MSc, School of Mathematical Sciences

National Institute of Science Education and Research (NISER), India