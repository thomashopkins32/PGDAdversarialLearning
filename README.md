# PGDAdversarialLearning
Projected Gradient Descent for Adversarial Learning

Based on the paper https://arxiv.org/pdf/1706.06083.pdf

## Empirical Evaluation of PGD Attack on ImageNet (not in order):
- [ ] Create scripts for downloading and preprocessing ImageNet dataset https://www.image-net.org/download.php
    - [ ] download script (tensorflow or keras may have these already)
    - [ ] preprocess script
- [ ] Find neural network architecture that achieves high accuracy on ImageNet
    - [ ] Find paper that describes architecture (and hopefully hyperparameters)
    - [ ] Implement architecture and train it without adversarial examples
    
    **or**
    
    - [ ] Find architecture and model parameters available to download
    - [ ] Write load script for model weights
- [ ] Run PGD attack on network with standard training
    - [ ] Edit the script from either https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py or https://github.com/MadryLab/cifar10_challenge/blob/master/pgd_attack.py
    - [ ] Run the attack, gathering statistics
    - [ ] Display statistics in plots similar to the paper
- [ ] Analyze network performance when trained with adversarial examples
    - [ ] Train network with PGD adversarially perturbed examples
    - [ ] Run a PGD attack, gathering statistics
    - [ ] Display statistics in plots similar to the paper

## Problem Set
- [ ] Develop a problem set similar to homeworks that go through the math
- [ ] Develop a set of simple experiments that show PGD attacks in action
    - [ ] Fashion-MNIST dataset?
    - [ ] UCI Adult dataset?
- [ ] This should be in a Jupyter notebook
