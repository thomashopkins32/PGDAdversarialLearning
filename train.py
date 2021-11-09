import json
import os
from timeit import default_timer as timer
from datetime import datetime
import shutil

import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.tensorboard as tb
import torchmetrics as tm
import torchvision
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import ResNet, BasicBlock, LeNet5
from pgd_attack import LinfPGDAttack

# get configuration for attack and learning
with open('config.json') as config_file:
    config = json.load(config_file)

# randomness control
torch.manual_seed(config['torch_random_seed'])
np.random.seed(config['np_random_seed'])

# training parameters
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_start = config['step_size_start']
step_size_schedule = config['step_size_schedule']
step_size_decay = config['step_size_decay']
weight_decay = config['weight_decay']
data_path = config['data_path']
momentum = config['momentum']
batch_size = config['training_batch_size']

# set up data and model
random_crop = torchvision.transforms.RandomCrop(32, padding=4)
random_flip = torchvision.transforms.RandomVerticalFlip()
to_tensor = torchvision.transforms.PILToTensor()
to_double = torchvision.transforms.ConvertImageDtype(torch.double)
cifar_transform = torchvision.transforms.Compose([random_crop,
                                                  random_flip,
                                                  to_tensor,
                                                  to_double])
cifar_train_data = torchvision.datasets.CIFAR100(data_path,
                                                 transform=cifar_transform,
                                                 train=True,
                                                 download=True)
cifar_test_data = torchvision.datasets.CIFAR100(data_path,
                                                transform=cifar_transform,
                                                train=False,
                                                download=True)

train_data = torch.utils.data.DataLoader(cifar_train_data,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=5)
test_data = torch.utils.data.DataLoader(cifar_test_data,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=5)
train_data_iter = iter(train_data)
test_data_iter = iter(test_data)
print(f'Training on {len(train_data)*batch_size} data points for {max_num_training_steps} batch steps...')



# TODO: option to load model from checkpoint
model = LeNet5(100).double()
# model = ResNet(BasicBlock, [1, 2, 3, 4], num_classes=100).double()
model.eval()

# set up optimizer and learning rate schedule
# need to try a couple different optimizers but the paper uses SGD w/ momentum
optimizer = optim.SGD(model.parameters(), lr=step_size_start,
                      momentum=momentum, weight_decay=weight_decay)
lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=step_size_schedule,
                                        gamma=step_size_decay)
loss_func = nn.CrossEntropyLoss()

# set up adversary
attack = LinfPGDAttack(model,
                       config['epsilon'],
                       config['num_steps'],
                       config['step_size'],
                       config['random_start'])

# set up checkpoint output directory
model_dir = config['model_dir']
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
shutil.copy('config.json', model_dir)
writer = tb.SummaryWriter(model_dir)

natural_accuracy = tm.Accuracy()
adversarial_accuracy = tm.Accuracy()

training_time = 0.0
for ii in tqdm(range(max_num_training_steps+1), total=max_num_training_steps+1):
    x_batch, y_batch = next(train_data_iter)
    start = timer()
    # compute adversarial perturbations
    x_batch_adv = attack.perturb(x_batch, y_batch)

    # training step
    model.train()
    optimizer.zero_grad()
    adv_outputs = model(x_batch_adv)
    loss = loss_func(adv_outputs, y_batch)
    adv_loss = loss.item()
    loss.backward()
    optimizer.step()

    end = timer()
    training_time += end - start
    model.eval()
    # evaluation
    if ii % num_output_steps == 0:
        nat_outputs = model(x_batch)
        nat_acc = natural_accuracy(nat_outputs, y_batch)
        adv_acc = adversarial_accuracy(adv_outputs, y_batch)

        print(f'Step {ii}: ({datetime.now()})')
        print(f'    training nat accuracy {nat_acc:.4}%')
        print(f'    training adv accuracy {adv_acc:.4}%')
        if ii != 0:
            print(f'    {num_output_steps * batch_size / training_time} examples per second')
            training_time = 0.0

        writer.add_scalar('accuracy adv train', adv_acc, ii)
        writer.add_scalar('accuracy adv', adv_acc, ii)
        writer.add_scalar('cross-entropy train', adv_loss / batch_size, ii)
        writer.add_scalar('cross-entropy', adv_loss / batch_size, ii)

    # save model checkpoint
    if ii % num_checkpoint_steps == 0:
        torch.save({
            'iteration': ii,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_schedule.state_dict(),
            'loss': loss
            }, model_dir + f'model_{ii}.pt')
    lr_schedule.step()
writer.close()
