'''
Runs a Projected Gradient Descent adversarial attack
by perturbing images based on first-order information
from the neural network
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


import numpy as np


class LinfPGDAttack:

    def __init__(self, model, epsilon, num_steps, step_size, random_start, device):
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start
        self.loss = nn.CrossEntropyLoss()
        self.device = device

    def perturb(self, x_nat, y):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            x = torch.clip(x, 0, 1) # ensure valid pixel range
        else:
            x = x_nat.astype(torch.float)

        for i in range(self.num_steps):
            x = x.to(self.device)
            x.requires_grad = True
            preds = self.model(x)
            loss = self.loss(preds, y)
            loss.backward()
            grad = x.grad.cpu()
            x = x.cpu().detach() + self.step_size * torch.sign(grad)
            x = torch.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = torch.clip(x, 0, 1) # ensure valid pixel range
        return x.to(self.device)


if __name__ == '__main__':
    import json
    import sys
    import math
    import time

    from model import ResNet, BasicBlock

    with open('config.json') as config_file:
        config = json.load(config_file)

    model = ResNet(BasicBlock, [1, 2, 3, 4]).double()
    model.eval()
    attack = LinfPGDAttack(model,
                           config['epsilon'],
                           config['num_steps'],
                           config['step_size'],
                           config['random_start'])

    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = [] # adv accumulator

    print('Iterating over {} batches'.format(num_batches))

    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)
        print('batch size: {}'.format(bend - bstart))

        x_batch = cifar.eval_data.xs[bstart:bend, :].reshape(-1, 3, 32, 32)
        y_batch = cifar.eval_data.ys[bstart:bend]
        print(x_batch.shape)

        start = time.time()
        x_batch_adv = attack.perturb(x_batch, y_batch)
        end = time.time()
        print(f'Time for perturbing: {end - start}')

        x_adv.append(x_batch_adv)

        print('Storing examples')
        path = config['store_adv_path']
        x_adv = np.concatenate(x_adv, axis=0)
        np.save(path, x_adv)
        print('Examples stored in {}'.format(path))
