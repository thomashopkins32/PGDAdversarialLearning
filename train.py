import json
import os
from timeit import default_timer as timer
from datetime import datetime
import shutil

import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.tensorboard as tb
from torchmetrics.functional import accuracy
import torchmetrics as tm
import torchvision
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import ResNet18, LeNet5
from pgd_attack import LinfPGDAttack


def train_one_epoch(model, loss_func, optimizer, dataloader, attack=None, device='cpu'):
    ''' Trains the model for one epoch '''
    model.train()
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        # compute adversarial perturbations
        if attack:
            x_batch = attack.perturb(x_batch, y_batch)
        # training step
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = loss_func(outputs, y_batch)
        loss.backward()
        optimizer.step()


def evaluate(model, loss_func, dataloader, attack, device='cpu'):
    ''' Evaluates the model on full test set '''
    model.eval()
    total_points = 0
    total_nat_loss = 0.0
    total_adv_loss = 0.0
    nat_acc = tm.Accuracy()
    adv_acc = tm.Accuracy()
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if attack:
            x_batch_adv = attack.perturb(x_batch, y_batch)
            adv_outputs = model(x_batch_adv)
            adv_loss = loss_func(adv_outputs, y_batch)
            total_adv_loss += adv_loss.item()
            adv_acc.update(adv_outputs, y_batch)
        nat_outputs = model(x_batch)
        nat_loss = loss_func(nat_outputs, y_batch)
        total_nat_loss += nat_loss.item()
        nat_acc.update(nat_outputs, y_batch)
        total_points += x_batch.shape[0]
    nat_ret = (total_nat_loss / total_points, nat_acc.compute().item() * 100)
    adv_ret = (total_adv_loss / total_points, adv_acc.compute().item() * 100)
    return nat_ret, adv_ret


if __name__=='__main__':

    # get configuration for attack and learning
    with open('config.json') as config_file:
        config = json.load(config_file)

    # randomness control
    torch.manual_seed(config['torch_random_seed'])
    np.random.seed(config['np_random_seed'])

    # GPU check
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # training parameters
    max_num_epochs = config['max_num_epochs']
    step_size_start = config['step_size_start']
    step_size_schedule = config['step_size_schedule']
    step_size_decay = config['step_size_decay']
    weight_decay = config['weight_decay']
    data_path = config['data_path']
    momentum = config['momentum']
    batch_size = config['training_batch_size']
    dataset = config['dataset']
    adv_training = config['train_with_adversary']

    # set up data and model
    to_tensor = torchvision.transforms.PILToTensor()
    to_double = torchvision.transforms.ConvertImageDtype(torch.double)
    cifar_transform = torchvision.transforms.Compose([to_tensor,
                                                      to_double])
    if dataset == 'cifar-100':
        num_classes = 100
        cifar_train_data = torchvision.datasets.CIFAR100(data_path,
                                                         transform=cifar_transform,
                                                         train=True,
                                                         download=True)
        cifar_test_data = torchvision.datasets.CIFAR100(data_path,
                                                        transform=cifar_transform,
                                                        train=False,
                                                        download=True)
    elif dataset == 'cifar-10':
        num_classes = 10
        cifar_train_data = torchvision.datasets.CIFAR10(data_path,
                                                        transform=cifar_transform,
                                                        train=True,
                                                        download=True)
        cifar_test_data = torchvision.datasets.CIFAR10(data_path,
                                                       transform=cifar_transform,
                                                       train=False,
                                                       download=True)
    else:
        print(f'Dataset {dataset} unknown...')
        exit(1)


    train_data = torch.utils.data.DataLoader(cifar_train_data,
                                             batch_size=batch_size,
                                             shuffle=True)
    test_data = torch.utils.data.DataLoader(cifar_test_data,
                                            batch_size=batch_size,
                                            shuffle=False)

    model = LeNet5(num_classes).double().to(device)
    # model = ResNet18(num_classes).double().to(device)
    model.eval()

    # set up optimizer and learning rate schedule
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
                           config['random_start'],
                           device)

    # set up checkpoint output directory
    model_dir = config['model_dir']
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    shutil.copy('config.json', model_dir)
    writer = tb.SummaryWriter(model_dir)

    print(f'Training on {dataset} with {len(train_data)*batch_size} data points for {max_num_epochs} epochs...')

    # training loop
    training_time = 0.0
    eval_time = 0.0
    for ii in tqdm(range(max_num_epochs), total=max_num_epochs):
        start = timer()
        if adv_training:
            train_one_epoch(model, loss_func, optimizer, train_data, attack=attack, device=device)
        else:
            train_one_epoch(model, loss_func, optimizer, train_data, attack=None, device=device)
        end = timer()
        training_time += end - start

        start = timer()
        nat_stats, adv_stats = evaluate(model, loss_func, test_data, attack=attack, device=device)
        end = timer()
        eval_time += end - start

        print(f'Epoch {ii}: ({datetime.now()})')
        print(f'    eval nat loss {nat_stats[0]:.4}')
        print(f'    eval adv loss {adv_stats[0]:.4}')
        print(f'    eval nat accuracy {nat_stats[1]:.4}%')
        print(f'    eval adv accuracy {adv_stats[1]:.4}%')
        print(f'Training time: {training_time:.4}s')
        print(f'Evaluation time: {eval_time:.4}s')

        writer.add_scalar('loss nat', nat_stats[0], ii)
        writer.add_scalar('loss adv', adv_stats[0], ii)
        writer.add_scalar('accuracy nat', nat_stats[1], ii)
        writer.add_scalar('accuracy adv', adv_stats[1], ii)

        # save model checkpoint
        torch.save({
            'epoch': ii,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_schedule.state_dict(),
            }, model_dir + f'_e{ii}.pt')
        lr_schedule.step()
    writer.close()
    if adv_training:
        torch.save(model.state_dict(), model_dir + f'final_{max_num_epochs}_adv.pt')
    else:
        torch.save(model.state_dict(), model_dir + f'final_{max_num_epochs}_nat.pt')
