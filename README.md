
# Learnable B-Spline Activations for Mitigating Catastrophic Forgetting

This repository contains the implementation of my undergraduate thesis:

**Title**:  
**"Using Learnable B-Spline Activation Functions to Address Catastrophic Forgetting in Neural Networks"**  
**Author**: Mahdi Razi Gandomani  
**Supervisors**: Dr. Ali Mohaddeseh Khorasani  
**Institution**: Amirkabir University of Technology (Tehran Polytechnic)  
**Date**: April 2025 (Farvardin 1404)

---

## 📚 Project Overview

**Catastrophic forgetting** remains a major challenge in continual learning systems, where neural networks forget previously learned tasks when adapting to new ones.  
This project proposes a novel **locally learnable activation function** based on **B-spline basis functions** to mitigate forgetting without sacrificing plasticity.

**Key Contributions:**
- Developed a **learnable B-spline activation function** module.
- Integrated B-spline activations into a feedforward neural network architecture.
- Conducted experiments on:
  - **Permuted MNIST continual learning** tasks.
  - **Toy regression tasks** involving **separated Gaussian peaks**.
- Compared models using traditional activations (**ReLU**) versus **learnable B-spline activations**.

---

## 📁 Repository Structure

```bash
.
├── bspline_activation.py      # Learnable B-spline activation module
├── model.py                   # Neural network models with/without learnable activations
├── permuted_mnist.py           # Permuted MNIST dataset creation
├── train_eval_mnist.py         # Training and evaluation on continual MNIST tasks
├── plot_mnist_results.ipynb       # Visualization of forgetting curves (MNIST)
├── toy_regression.ipynb           # Regression task with Gaussian peaks (Toy dataset)
└── README.md                   
```

---

## 🧠 Learnable B-Spline Activation Function

Traditional activation functions (e.g., ReLU, Tanh) are **global** and affect the entire input space. In contrast, B-splines are **locally supported**: a change in one region can leave other regions intact, helping to mitigate forgetting of previously learned tasks.

In this implementation:
- The activation is a **trainable weighted sum of B-spline basis functions**.
- We optimize the **control points** during network training.
- Enables **local adaptation** to new tasks with **minimal interference** to old tasks.
---

## 🔥 Experiments

### 1. Permuted MNIST (Continual Learning)

- **Task**: Sequential learning of 3 different permutations of MNIST digits.
- **Comparison**:  
  - **Baseline**: MLP with ReLU activations  
  - **Proposed**: MLP with Learnable B-Spline activations
- **Metric**: Classification Accuracy across tasks after each training stage.

**Findings**:
- **Forgetting curves**: Plot the evolution of accuracy after learning each task.
![Forgetting curves](results/forgetting1_mnist.png)
![Training timeline](results/forgetting2_mnist.png)
- **Training timeline**: Plot per-task accuracies epoch-by-epoch to observe forgetting behavior.
![Training timeline](results/training_timeline.png)
- ReLU networks **forget** old tasks significantly after learning new ones.
- B-Spline networks **preserve** higher accuracy on previous tasks.

### 2. Gaussian Peaks Regression (Toy Task)

- **Task**: Sequential modeling of 5 separated Gaussian peaks.
- **Comparison**:
  - **Baseline**: MLP with ReLU activations
  - **Proposed**: MLP with Learnable B-Spline activations
- **Metric**: Regression accuracy on earlier peaks after training new ones.

**Findings**:
- **Regression fitting**: Visualize learned functions after each sequential regression task.
![Regression fitting](results/regression_fitting.png)
![Regression fitting](results/regression_fitting2.png)
- B-Spline model maintain **better retention** of older peaks.
- ReLU model **overfit** to new peaks and **forget** previous ones almost entirely.

---

## 🛠️ Installation

**Requirements**:
- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- matplotlib

```bash
pip install torch torchvision numpy matplotlib
```

---

## 🚀 Running the Code

Train and compare on **Permuted MNIST**:
```bash
python train_eval_mnist.py
```

Run the **Toy Gaussian Peaks** regression:
```bash
python toy_regression.py
```

---


## 📬 Contact

For any questions or suggestions, please reach out to:

- **Email**: [mahdi.razi@aut.ac.ir]

---

## ⭐ Acknowledgments

Special thanks to my supervisors **Dr. Ali Mohaddeseh Khorasani** for their invaluable guidance and support throughout this project.

---
