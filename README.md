# Optimizer-Playground
An interactive web app for visualizing and comparing how deep learning optimizers converge on objective functions.

<p align="center">
  <img src="optimizer_animation.gif" alt="OptiScope optimizer visualization" width="400">
</p>


**Optimizer-Playground** is an interactive web application for visualizing how deep learning optimizers such as **Adam** converge on objective functions. It helps users build intuition about optimization dynamics by comparing optimizer trajectories, convergence speed, and behavior across different loss landscapes.

## Live Demo

https://kevinsvip888.github.io/Optimizer-Playground/

<img width="1058" height="942" alt="image" src="https://github.com/user-attachments/assets/558cd1ba-572c-4198-8067-68b37eaa62d7" />

## Overview

Optimization is at the heart of deep learning, but the behavior of different optimizers can be hard to understand from formulas alone. **Optimizer-Playground** makes optimization easier to grasp by visualizing how different algorithms move across an objective function and approach minima over time.

Instead of only reading about learning rates, momentum, or adaptive updates, users can directly observe how each optimizer behaves and compare them side by side.

## Supported Optimizers

This project currently includes the following optimizers:

- **SGD** — vanilla gradient descent
- **Momentum** — SGD with velocity
- **Nesterov** — lookahead momentum
- **AdaGrad** — adaptive per-parameter learning rate
- **RMSProp** — leaky AdaGrad
- **Adam** — adaptive optimization with momentum
- **AdamW** — Adam with decoupled weight decay

## Features

- Interactive optimizer comparison
- Visual trajectories over objective functions
- Easy switching between multiple optimizers
- Educational interface for learning optimization concepts
- Clear and visually distinct optimizer selection UI

## Why This Project?

Different optimizers often behave very differently even on the same objective function. This project helps users build intuition about:

- convergence speed
- stability of updates
- oscillation and overshooting
- effect of momentum
- adaptive versus non-adaptive methods
- differences in optimization paths across algorithms

It is designed for students, educators, and practitioners who want a more visual understanding of optimization in deep learning.

## Learn More

For users who want to study the theory and intuition behind these optimizers, here are some helpful references.

### Papers and Original Sources

- **SGD**
  - Robbins & Monro, *A Stochastic Approximation Method*  
    https://www.columbia.edu/~ww2040/8100F16/RM51.pdf

- **Momentum**
  - Momentum is commonly taught as an extension of gradient descent in optimization and deep learning courses. A good starting reference is:
    - Dive into Deep Learning — Momentum  
      https://d2l.ai/chapter_optimization/momentum.html

- **Nesterov**
  - Yurii Nesterov, *A Method for Solving the Convex Programming Problem with Convergence Rate O(1/k^2)*  
    https://hengshuaiyao.github.io/papers/nesterov83.pdf

- **AdaGrad**
  - Duchi, Hazan, and Singer, *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*  
    https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf

- **RMSProp**
  - Geoffrey Hinton, *Neural Networks for Machine Learning, Lecture 6*  
    https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

- **Adam**
  - Kingma and Ba, *Adam: A Method for Stochastic Optimization*  
    https://arxiv.org/abs/1412.6980

- **AdamW**
  - Loshchilov and Hutter, *Decoupled Weight Decay Regularization*  
    https://arxiv.org/abs/1711.05101

### Blogs, Notes, and Tutorials

- **General optimizer overview**
  - CS231n — Neural Networks Part 3  
    https://cs231n.github.io/neural-networks-3/

- **SGD**
  - Dive into Deep Learning — Stochastic Gradient Descent  
    https://d2l.ai/chapter_optimization/sgd.html

- **Momentum**
  - Dive into Deep Learning — Momentum  
    https://d2l.ai/chapter_optimization/momentum.html

- **AdaGrad**
  - Dive into Deep Learning — AdaGrad  
    https://d2l.ai/chapter_optimization/adagrad.html

- **RMSProp**
  - Dive into Deep Learning — RMSProp  
    https://d2l.ai/chapter_optimization/rmsprop.html

- **Adam**
  - Dive into Deep Learning — Adam  
    https://d2l.ai/chapter_optimization/adam.html

### Suggested Learning Path

If you are new to optimization, a good order is:

1. Gradient Descent / SGD
2. Momentum
3. Nesterov
4. AdaGrad
5. RMSProp
6. Adam
7. AdamW
