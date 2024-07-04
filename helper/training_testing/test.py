import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from helper.training_testing import metrics

import matplotlib.pyplot as plt
import os

from helper.training_testing import dataset

# Making the forward pass for testing 
# a generator because used in multple metrics
def get_predictions_nn(dataloader, model, device, dtype):
    model.eval()
    with torch.no_grad():
        for data, target, ids in tqdm(dataloader, desc='Test iterator', leave=False, dynamic_ncols=True):
            data = data.type(dtype).to(device)
            target = target.type(dtype).to(device)

            data = data.permute((0, 2, 1))
            output = model(data, ids).squeeze()

            yield (target, output)


def test_nn(test_generator, model, criterion, file_path, device, dtype, wandb, class_unbalance):
    total_loss = 0
    pred_list = []
    target_list = []

    for target, output in get_predictions_nn(test_generator, model, device, dtype):
        total_loss += criterion(output, target).item()

        sigmoid = nn.Sigmoid()
        output = sigmoid(output)

        pred_list.append(output)
        target_list.append(target)

    target_list = torch.cat(target_list).cpu().numpy()
    pred_list = torch.cat(pred_list).cpu().numpy()

    test_acc, _, total = metrics.accuracy(target_list, pred_list)

    total_loss = total_loss/total

    fpr, tpr, roc_auc, tresholds, roc_display = metrics.roc_curves(target_list, pred_list)
    precision, recall,pr_auc, thresholds,pr_display = metrics.pr_curves(target_list, pred_list)

    conf_mat = metrics.conf_matrix(target_list, pred_list)
    f1, weighted_f1 = metrics.f1(target_list, pred_list, class_unbalance)

    # #PR Curve
    # wandb.log({"eval/PR-Curve": pr_display})

    # #ROC Curve
    # wandb.log({"eval/ROC-Curve": roc_display})

    #Confusion Matrix
    wandb.log({"eval/conf_mat" : wandb.plot.confusion_matrix(preds=pred_list.round(), y_true=target_list)})

    ################LOG SCALAR VALUES################
    wandb.log({"eval/accuracy": test_acc,
                    "eval/loss": total_loss,
                    "eval/F1": f1,
                    "eval/weighted_F1": weighted_f1,
                    "eval/ROC-AUC": roc_auc,
                    "eval/PR-AUC": pr_auc})


    return (test_acc, total_loss, roc_auc, pr_auc, conf_mat, f1, weighted_f1)