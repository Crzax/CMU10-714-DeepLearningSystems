# CMU 10-714: Deep Learning Systems

This repository contains my complete implementations for the assignments of the [CMU 10-714: Deep Learning Systems](https://dlsyscourse.org/) course. All tasks across all homework assignments have been fully implemented and tested.

## Project Overview

The goal of this course is to build a deep learning library called `needle` (Necessary Elements of Deep LEarning) from scratch, covering everything from automatic differentiation to hardware backends and high-level model architectures.

## Assignments Summary

### [Homework 0: Basic ML and C++ Extensions](./hw0/)
- Implemented softmax regression for MNIST digit classification.
- Developed a C++ extension for matrix operations to improve performance.

### [Homework 1: Automatic Differentiation](./hw1/)
- Built the core `needle` autograd engine.
- Implemented forward and backward passes for basic mathematical operations.
- Developed the computational graph system for automatic gradient calculation.

### [Homework 2: Neural Network Library](./hw2/)
- Implemented high-level neural network modules (`nn.Module`, `nn.Linear`, `nn.ReLU`, `nn.LayerNorm`, etc.).
- Developed optimizers including SGD with momentum and Adam.
- Built a data loading pipeline with `Dataset` and `DataLoader` classes.
- Implemented and trained MLP and ResNet architectures.

### [Homework 3: Array Library and Backends](./hw3/)
- Developed a custom `ndarray` library to handle memory management and broadcasting.
- Implemented efficient CPU (C++) and GPU (CUDA) backends for tensor operations.
- Integrated the backends with the `needle` autograd system.

### [Homework 4: Convolutional and Sequence Models](./hw4/)
- Implemented Convolutional layers and pooling operations.
- Developed sequence models including Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks.
- Trained models on CIFAR-10 (image classification) and Penn Treebank (language modeling) datasets.

### [Homework 4 Extra: Transformers](./hw4_extra/)
- Implemented the Transformer architecture from scratch.
- Developed Multi-Head Attention, Transformer layers, and positional embeddings.
- Built a decoder-only Transformer for language modeling tasks.

## Implementation Status
All assignments (HW0 - HW4 Extra) are **fully implemented**. This includes:
- Core autograd engine.
- CPU and CUDA backends.
- Neural network modules and optimizers.
- Advanced architectures (ResNet, LSTM, Transformers).
