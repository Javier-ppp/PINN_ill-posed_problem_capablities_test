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

The project investigates the identification of physical parameters in systems governed by conservation laws, modeled via the electrical RC circuit analogy. 

### 1. Mathematical Models
The system is evaluated across two primary configurations:

- **Single-Stage RC Filter (1st Order)**:
  \[
  v_{in}(t) - v_{out}(t) - RC \frac{dv_{out}(t)}{dt} = 0
  \]
  The system reaches a pole at:
  \[
  p = -\frac{1}{RC}
  \]

- **Two-Stage RC Filter (2nd Order)**:
  The two-stage configuration can be represented by a first-order system (identifying partial information):
  \[
  v_{in}(t) - v_1(t) - R_1 C_1 \frac{dv_1(t)}{dt} - R_1 C_2 \frac{dv_{out}(t)}{dt} = 0
  ]\
  Alternatively, it can be combined into a single second-order residual form:
  $$
  R_1 C_1 R_2 C_2 \frac{d^2 v_{out}(t)}{dt^2} + (R_2 C_2 + R_1(C_1 + C_2)) \frac{dv_{out}(t)}{dt} + v_{out}(t) - v_{in}(t) = 0
  $$
  This formulation allows the PINN to fit a single output signal ($v_{out}$) to discover multiple underlying parameters ($R_i, C_i$).

### 2. The Loss Function & Constraints
To ensure physical consistency and parameter positivity (e.g., $R, C > 0$), the PINN employs a constrained loss function:
$$
\mathcal{L} = (1 - \lambda) \mathcal{L}_{data} + \lambda \mathcal{L}_{physics} + \sum Swish(\mu_i) + \sum Swish(\rho_i)
$$
where:
- $\mathcal{L}_{physics}$ is the mean squared residual of the ODE.
- $Swish(x) = x \cdot Sigmoid(x)$ acts as a soft penalty to prevent parameters from becoming negative during optimization.
- $\mu_i$ and $\rho_i$ represent the network's approximations of $C_i$ and $R_i$.

### 3. The Curse of Dimensionality & Uniqueness
A core finding of this research is the **identifiability crisis** in ill-posed inverse problems. While the PINN can accurately reconstruct the system's dynamics (poles $p_i$), the individual components ($R_i, C_i$) are not unique for a given set of poles in multi-stage systems. As the number of parameters increases, the optimization landscape becomes increasingly complex, rendering the discovery of exact physical constants fundamentally unstable without additional boundary information.

## 📂 Project Structure

- `PINN wall demo-inverse.ipynb`: Main research and implementation notebook.
- `2511.08561v1.pdf`: Research paper: "The curse of dimensionality: what lies beyond the capabilities of physics-informed neural networks".
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