# PGD Adversarial Learning
Projected Gradient Descent for Adversarial Learning

Based on the paper https://arxiv.org/pdf/1706.06083.pdf

## Issues
- ImageNet is too big. We should use CIFAR-100 instead. It is available using torchvision as follows:
```python3
torchvision.datasets.CIFAR100(root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
```

## Empirical Evaluation of PGD Attack on CIFAR-100 (not in order):
- [ ] Create scripts for downloading and preprocessing CIFAR-100 dataset (see above)
    - [ ] download script
    - [ ] preprocess script
- [ ] Find neural network architecture that achieves high accuracy on CIFAR-100
    - [ ] Find paper that describes architecture (and hopefully hyperparameters)
    - [] Implement architecture and train it without adversarial examples
    
    **or**
    
    - [ ] Find architecture and model parameters available to download
    - [ ] Write load script for model weights
- [ ] Run PGD attack on network with standard training
    - [x] Edit the script from https://github.com/MadryLab/cifar10_challenge/blob/master/pgd_attack.py
    - [ ] Run the attack, gathering statistics
    - [ ] Display statistics in plots similar to the paper
- [ ] Analyze network performance when trained with adversarial examples
    - [ ] Train network with PGD adversarially perturbed examples (see https://github.com/MadryLab/cifar10_challenge/blob/master/train.py for example)
    - [ ] Run a PGD attack, gathering statistics
    - [ ] Display statistics in plots similar to the paper

## Problem Set
- [ ] Develop a problem set similar to homeworks that go through the math
- [ ] Develop a set of simple experiments that show PGD attacks in action
    - [ ] MNIST dataset?
    - [ ] Fashion-MNIST dataset?
    - [ ] UCI Adult dataset?
- [ ] This should be in a Jupyter notebook


## Presentation
Who are the authors, and the date and venue of publication?

What is the problem that is addressed (pick one, if the paper addresses more than one), and why is it interesting or useful?

What is the main result of the paper?

Describe the result or algorithm and motivate it intuitively.

What is the cost (time, space, or some other metric) of this algorithm, and how does it compare to prior algorithms for the same problem? (and similarly, for non-algorithmic results)
What performance guarantees, if any, are provided for the algorithm?

Give an accurate description of the analysis given in the paper: in simple cases this may be a tour through the entire argument; when this is not possible, focus on explaining a core lemma/theorem that supports the claim of the paper.

Provide an empirical evaluation of the algorithm: compare its performance to reasonable baselines, and explore relevant aspects of the algorithm (its variability, sensitivity to relevant properties of the input, etc.). If presenting a non-algorithmic result and it is possible, provide some experimental evidence of its sharpness or lack thereof.

