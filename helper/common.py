import random
import numpy as np
import torch
import tensorflow as tf
import argparse

def set_seed(seed, logger=None):
    logger.info(f"Random seed set to {seed}") if logger else None
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    # # For reproducibility on CUDA with pytorch
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def parse_cmd_args():
    # Setting up arg parse
    argParser = argparse.ArgumentParser()

    argParser.add_argument("--model", help="Choose the model to train")
    argParser.add_argument("--batch_size", help="Choose the batch size to train the network", type=int)
    argParser.add_argument("--epochs", help="Choose the number of epochs to train the network", type=int)
    argParser.add_argument("--weight_decay", help="Choose the network weight decay", type=float)
    argParser.add_argument("--lr_scheduler", help="Choose the lr scheduler", type=str)
    argParser.add_argument("--lr", help="Choose the learning rate", type=float)
    argParser.add_argument("--step_lr_epoch_div", help="Choose every how many epochs to divide the lr", type=float)
    argParser.add_argument("--step_lr_div_factor", help="Choose by how much to divide the lr", type=float)
    argParser.add_argument("--num_channels", help="Choose the number of conv channels", type=list)
    argParser.add_argument("--dense_layers", help="Choose the number of dense layers", type=list)
    argParser.add_argument("--k", help="Choose dataset shard for testing", type=int)
    argParser.add_argument("--data_freq_min", help="Choose the sampling frequency of the data", type=int)
    argParser.add_argument("--data_minutes", help="Choose the data window lenght in minutes", type=int)
    argParser.add_argument("--prediction_minutes", help="Choose the predicion window lenght in minutes", type=int)
    argParser.add_argument("--online_threshold", help="Choose the predicion window lenght in minutes", type=int)
    

    args = argParser.parse_args()
    return args