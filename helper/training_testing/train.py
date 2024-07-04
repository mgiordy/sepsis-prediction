import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from helper.training_testing import test
from helper.training_testing import dataset, metrics


def get_criterion(class_unbalance=1):
    # Using BCE with digits to have better numerical stability
    return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1/class_unbalance]))


def get_optimizer(model, lr, weight_decay):
    # torch.optim.SGD(model.parameters(), lr=lr)#, momentum=0.9)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(lr_schedule, optimizer, n_batches, epoches, stop_lr, lr_epoch_div, lr_mult_factor):
    scheduler = None
    if lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_epoch_div, lr_mult_factor)
    elif lr_schedule == "const":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 1)
    elif lr_schedule == "warmup":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, stop_lr, epochs=epoches, steps_per_epoch=n_batches)
    return scheduler


def train_nn_batch(wandb, dataloader, model, device, dtype, criterion, optimizer, scheduler, pbar, debug_grad):
    train_loss, correct = 0, 0
    total = 0
    model.train()
    for batch, (data, target, ids) in enumerate(dataloader):
        data = data.type(dtype).to(device)
        target = target.type(dtype).to(device)

        data = data.permute((0, 2, 1))

        # forward + backward + optimize
        output = model(data, ids).squeeze()

        sigmoid = nn.Sigmoid()
        output_digit = sigmoid(output)

        loss = criterion(output, target)

        train_loss += loss.item()
        _, correct_batch, total_batch = metrics.accuracy(target, output_digit)
        correct += correct_batch
        total += total_batch

        # wandb.watch(model, log='all', log_freq=10)
        wandb.log({"train/loss": loss.item(),
                    "train/accuracy": correct/total,
                    "train/ones": output_digit.round().sum(),
                    "train/LR": optimizer.param_groups[0]['lr']})

        # zero the parameter gradients and compute backprop
        optimizer.zero_grad()
        loss.backward()

        if debug_grad:
            for name, param in model.named_parameters():
                if param.grad is not None and param.data is not None:
                    wandb.log({"grads/" + name + ".grad": wandb.Histogram(
                                    sequence = param.grad.flatten().cpu(),
                                    num_bins = 64)})
                    wandb.log({"weights/" + name + ".data": wandb.Histogram(
                                    sequence = param.data.flatten().cpu(),
                                    num_bins = 64)})
        
        optimizer.step() # Only for OneCycleLR

        pbar.set_description(f'Batch pbar. Loss: {train_loss/total:.5f}, accuracy: {correct/total:.3f}')
        pbar.update(1)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
    
    if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        scheduler.step()

    pbar.reset()
    return (train_loss/total, correct/total)


def train_nn(wandb, file_path, model, train_generator, test_generator, device, dtype, criterion, optimizer, scheduler, epoches, n_batches, debug_grad, class_unbalance):
    criterion.to(device)
    model.to(device)
    
    pbar = tqdm(total=n_batches, dynamic_ncols=True)
    epoches_pbar = tqdm(range(epoches), dynamic_ncols=True)

    for epoch in epoches_pbar:
        # Train model
        train_loss, accuracy = train_nn_batch(wandb, train_generator, model, device, dtype, criterion, optimizer, scheduler, pbar, debug_grad)
        # Run evaluation
        test.test_nn(test_generator, model, criterion, file_path, device, dtype, wandb, class_unbalance)

        torch.save(model.state_dict(), os.path.join(file_path, f'model_epoch_{epoch}.pth'))
        epoches_pbar.set_description(f'Epochs pbar. Train loss: {train_loss:.5f}, accuracy: {accuracy:.3f}, Val loss: {train_loss:.5f}, accuracy: {accuracy:.3f},')
    pbar.close()
    epoches_pbar.close()