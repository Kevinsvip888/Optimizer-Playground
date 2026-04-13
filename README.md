# Optimizer Playground

Interactive visualization of how deep learning optimizers converge on objective functions.

<p align="center">
  <img src="optimizer_animation.gif" alt="Optimizer Playground animation" width="420">
</p>

<p align="center">
  <a href="https://kevinsvip888.github.io/Optimizer-Playground/"><strong>Live Demo</strong></a>
</p>

## Overview

**Optimizer Playground** is an interactive web application for visualizing and comparing how popular deep learning optimizers move across objective functions and converge toward minima. By showing optimizer trajectories and convergence behavior directly, the project helps build intuition for optimization in a way that formulas alone often cannot.

This project is designed for students, educators, and practitioners who want a more visual understanding of gradient-based optimization in deep learning.

## Screenshot

<p align="center">
  <img src="https://github.com/user-attachments/assets/558cd1ba-572c-4198-8067-68b37eaa62d7" alt="Optimizer Playground interface" width="900">
</p>

## Supported Optimizers

- **SGD** — vanilla gradient descent
- **Momentum** — SGD with velocity
- **Nesterov** — lookahead momentum
- **AdaGrad** — adaptive per-parameter learning rate
- **RMSProp** — exponentially weighted adaptive learning rate
- **Adam** — adaptive optimization with momentum
- **AdamW** — Adam with decoupled weight decay

## Features

- Interactive comparison of multiple optimizers
- Visual optimizer trajectories over objective functions
- Side-by-side intuition for convergence speed and stability
- Clean, educational interface for learning optimization concepts
- Lightweight static site deployable with GitHub Pages

## Why This Project?

Different optimizers can behave very differently on the same loss surface. **Optimizer Playground** helps users understand:

- how quickly optimizers converge
- how stable or oscillatory their updates are
- how momentum changes optimization paths
- how adaptive methods differ from standard gradient descent
- how optimizer choice affects movement across a landscape

## Learn More

For readers who want to go deeper into the theory and intuition behind these methods, the following papers and tutorials are useful starting points.

### Papers and Original Sources

- **SGD**  
  Robbins & Monro, *A Stochastic Approximation Method*  
  [Read the paper](https://www.columbia.edu/~ww2040/8100F16/RM51.pdf)

- **Momentum**  
  A practical introduction:  
  [Dive into Deep Learning — Momentum](https://d2l.ai/chapter_optimization/momentum.html)

- **Nesterov**  
  Yurii Nesterov, *A Method for Solving the Convex Programming Problem with Convergence Rate O(1/k^2)*  
  [Read the paper](https://hengshuaiyao.github.io/papers/nesterov83.pdf)

- **AdaGrad**  
  Duchi, Hazan, and Singer, *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*  
  [Read the paper](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

- **RMSProp**  
  Geoffrey Hinton, *Neural Networks for Machine Learning, Lecture 6*  
  [Read the lecture notes](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

- **Adam**  
  Kingma and Ba, *Adam: A Method for Stochastic Optimization*  
  [Read the paper](https://arxiv.org/abs/1412.6980)

- **AdamW**  
  Loshchilov and Hutter, *Decoupled Weight Decay Regularization*  
  [Read the paper](https://arxiv.org/abs/1711.05101)

### Tutorials and Notes

- [CS231n — Neural Networks Part 3](https://cs231n.github.io/neural-networks-3/)
- [Dive into Deep Learning — SGD](https://d2l.ai/chapter_optimization/sgd.html)
- [Dive into Deep Learning — Momentum](https://d2l.ai/chapter_optimization/momentum.html)
- [Dive into Deep Learning — AdaGrad](https://d2l.ai/chapter_optimization/adagrad.html)
- [Dive into Deep Learning — RMSProp](https://d2l.ai/chapter_optimization/rmsprop.html)
- [Dive into Deep Learning — Adam](https://d2l.ai/chapter_optimization/adam.html)

### Suggested Learning Order

1. SGD  
2. Momentum  
3. Nesterov  
4. AdaGrad  
5. RMSProp  
6. Adam  
7. AdamW

## Tech Stack

- HTML
- CSS
- JavaScript

## Live Demo

Explore the project here:  
[https://kevinsvip888.github.io/Optimizer-Playground/](https://kevinsvip888.github.io/Optimizer-Playground/)

## License

This project is open source and available under the MIT License.
