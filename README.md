# Optimizer-Playground
An interactive web app for visualizing and comparing how deep learning optimizers converge on objective functions.


**Optimizer-Playground** is an interactive web application for visualizing how deep learning optimizers such as **Adam** converge on objective functions. It helps users build intuition about optimization dynamics by comparing optimizer trajectories, convergence speed, and behavior across different loss landscapes.

## Live Demo

https://kevinsvip888.github.io/Optimizer-Playground/


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

## Tech Stack

- **HTML**
- **CSS**
- **JavaScript**

## Project Structure

```bash
optiscope/
├── index.html
├── style.css
├── app.js
├── assets/
│   └── screenshot.png
└── README.md
