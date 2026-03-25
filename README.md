# PINN Ill-Posed Problem Capabilities Test

This repository explores the application of **Physics-Informed Neural Networks (PINNs)** to solve **ill-posed inverse problems** in the context of thermal modeling. Specifically, it focuses on identifying physical parameters of a wall structure by leveraging an analogy with electrical RC circuits.

## 🚀 Key Features

- **Physics-Informed Learning**: Utilizes Neural Networks that respect physical laws defined by ordinary differential equations (ODEs).
- **Inverse Problem Solving**: Discovers unknown physical parameters (like thermal dissipation or time constants) from noisy sparse observational data.
- **Circuit-to-Heat Analogy**: Models 1D heat transfer through wall layers as Resistance-Capacitance (RC) circuits.
- **Automated Hyperparameter Tuning**: Integrated with **Ray Tune** and **Optuna** for efficient model optimization.
- **High-Fidelity Comparison**: Uses **PySpice** for industrial-grade SPICE circuit simulations to validate PINN results.
- **Robustness Testing**: Evaluates performance under varying noise levels in experimental observations.

## 🛠 Technical Stack

- **Deep Learning**: PyTorch
- **Physics Modeling**: PySpice, SciPy (solve_ivp)
- **Visualization**: Matplotlib, Seaborn, Schemdraw (for circuit diagrams), Torchviz/Torchview
- **Optimization**: Ray (Tune), Optuna, Hyperopt

## 📖 Theoretical Background

The project uses the general RC circuit differential equation:

$$\frac{dV_C(t)}{dt} + \frac{1}{RC}V_C(t) = \frac{V_{in}(t)}{RC}$$

In the thermal analogy:
- $V_C(t)$ represents the temperature (node voltage).
- $V_{in}(t)$ represents the external temperature source (input voltage).
- $R$ and $C$ represent thermal resistance and capacitance respectively.

The PINN minimizes a composite loss function:
$$ \mathcal{L} = \mathcal{L}_{physics} + \lambda \mathcal{L}_{data} $$
where the physics loss ensures the network satisfies the ODE residual, and the data loss matches observational points.

## 📂 Project Structure

- `PINN wall demo-inverse.ipynb`: Main research and implementation notebook.
- `nonoise_single.pt`: Saved model weights for a single-layer wall simulation.
- `nonoise_two_nodes.pt`: Saved model weights for a two-node (complex) simulation.
- `.env/`: Local environment configuration (standard for this repo).

## 🚦 Getting Started

1. **Prerequisites**:
   - Python 3.11+
   - NgSpice (required for PySpice simulations)
   - PyTorch (MPS/CUDA supported)

2. **Running the Demo**:
   Open `PINN wall demo-inverse.ipynb` in your preferred Jupyter environment. The notebook is self-contained and walks through:
   - Initial PINN architecture setup.
   - Solving a standard analytical example.
   - Simulating single and multi-node RC circuits.
   - Executing the inverse solver to "discover" physical constants.

## 📊 Results Summary

The tests demonstrate that PINNs are highly capable of reconstructing underlying dynamics and identifying parameters even when data is sparse and corrupted with noise, outperforming standard data-driven interpolation in physically constrained domains. Also shows that when missing more than a single parameter the PINN cannot retrieve the parameters. Hence it cannot solve ill-posed problems. As oposed to what is found in the scientific literature.