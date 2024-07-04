# %%
import torch
import pandas as pd
import os
import sys
import yaml
from yaml.loader import SafeLoader
import wandb
import argparse

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from helper import logging_conf
from helper.common import set_seed
from helper.dataset_generation.common import generate_stats
from helper.dataset_generation.hirid_specific import hirid
from helper.dataset_generation.mimic_specific import mimic
from helper.training_testing.dataset import get_ids

logger = logging_conf.get_logger("main.py")


num_workers = int(os.cpu_count()//2)
logger.info(f"CPU count: {num_workers}")


if __name__ == '__main__':
    # Setting up arg parse
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--dataset", help="Choose the dataset to retrieve from PostgreSQL")
    args = argParser.parse_args()

    # Read config and init Wandb
    config={}
    with open('dataset_config.yaml') as f:
        config = yaml.load(f, Loader=SafeLoader)

    # Setting wandb mode
    if config["wandb"]:
        os.environ["WANDB_MODE"] = "online"
    else:
        os.environ["WANDB_MODE"] = "disabled"

    # Overwrite config with args
    for key, value in vars(args).items():
        if value is not None:
            logger.info(f"Overwriting config {key} with {value}")
            config[key] = value

    wandb.init(project=config["project_name"], tags=["dataset_gen"], entity=config["entity"], config=config, allow_val_change=True)

    # Setting seed for reproducibility
    set_seed(wandb.config.seed, logger)

    dtype = torch.float
    device = "cpu"

    # Get the directory of the script, which is the upmost one
    current_script_folder = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(current_script_folder, "dataset_gen_runs", wandb.config.dataset + "_" + wandb.run.id)
    # Check if the directory exists, even if should always be unique due to wandb.run.id
    if not os.path.exists(data_path):
        # Create the directory
        os.makedirs(data_path)
        logger.info(f"Directory '{data_path}' created successfully.")
    else:
        logger.error(f"Directory '{data_path}' already exists!")
        sys.exit(1)

    if not wandb.config.wandb:
        logger.warning("Running wandb in offline mode!!")

    logger.info(f"Running on device: {device}")
    
    if wandb.config.dataset is not None:
        if wandb.config.dataset == "hirid":
            hirid.generate_dataset(data_path, num_workers)
        elif wandb.config.dataset == "mimic":
            mimic.generate_dataset(data_path, num_workers)
        else:
            logger.error(f"ERROR - Dataset '{wandb.config.dataset}' not found!")
            sys.exit(1)
    else:
        logger.error("ERROR - No dataset specified!")
        sys.exit(1)

    logger.info("Finished generating dataset")
    logger.info("Generating stats")
    for k in range(wandb.config.k_splits):
        ids = pd.read_csv(os.path.join(data_path, "0labels.txt"))
        train_ids, _ = get_ids(ids, k, wandb.config.k_splits, 0) # Agnostic of time to sepsis cutoff
        train_ids = train_ids.iloc[:1000] # 1000 shuffled patients -> RAM constraints
        q_low, q_hi, mu, std = generate_stats(train_ids, data_path)
        stats = pd.DataFrame({"k": k, "mu": mu, "std": std, "q_low": q_low, "q_hi": q_hi}).reset_index(names="vitalsign")
        stats.to_csv(os.path.join(data_path, "0stats.txt"), mode='a', index=False, header=False if k > 0 else True)
    logger.info("Finished generating stats")