import wandb
import yaml
from yaml.loader import SafeLoader
import sys
import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ["OMP_NUM_THREADS"] = "10"

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1) 
tf.config.threading.set_intra_op_parallelism_threads(1)

from helper import logging_conf
from helper.common import set_seed, parse_cmd_args
from helper.tf_helper import dataset as dataset_tf
from helper.tf_helper import train as train_tf
from helper.tf_helper.test import test_float_model, evaluate_quantized_model
from helper.tf_helper.quantise import convert

from models import tf_tcn

logger = logging_conf.get_logger("main TF.py")

num_workers = 8# int(os.cpu_count()//2)
logger.info(f"CPU count: {num_workers}")


if __name__ == '__main__':
    args = parse_cmd_args()

    # Read config and init Wandb
    config={}
    with open('network_config.yaml') as f:
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

    wandb.init(project=config["project_name"], tags=["train_tf", config["model"]], entity=config["entity"], config=config, allow_val_change=True)

    if len(wandb.config.datasets) == 0:
        logger.error("No dataset specified! Please specify at least one dataset to train on.")
        sys.exit(1)

    if not wandb.config.wandb:
        logger.warning("Running wandb in offline mode!!")

    set_seed(wandb.config.seed, logger)

    # Retrieving the right folder
    # Get the directory of the script, which is the upmost one
    current_script_folder = os.path.dirname(os.path.abspath(__file__))
    dataset_folders = {}
    for d in wandb.config.datasets:
        data_path = os.path.join(current_script_folder, "generated_datasets", d)
        if not os.path.exists(data_path):
            logger.error(f"Directory '{data_path}' does not exists! Generate dataset before training.")
            sys.exit(1)
        dataset_folders[d] = data_path

    datagen_params = {'batch_size': wandb.config.batch_size,
        'shuffle': True,
        'num_workers': num_workers}

    # Get datasets
    train_ds, test_ds, rep_ds, test_ids = dataset_tf.get_datasets(dataset_folders,
                                os.path.join(current_script_folder, "generated_datasets"),
                                wandb.config.k,
                                wandb.config.k_splits,
                                datagen_params,
                                custom_sampler=wandb.config.custom_sampler,
                                frequency=wandb.config.data_freq_min,
                                data_minutes=wandb.config.data_minutes,
                                prediction_minutes=wandb.config.prediction_minutes,
                                online_train=wandb.config.online_train,
                                online_test=wandb.config.online_test,
                                online_stride=wandb.config.online_stride,
                                time_to_sepsis_cutoff=wandb.config.time_to_sepsis_cutoff,
                                control_window_cutoff=wandb.config.control_window_cutoff,
                                col_to_keep=wandb.config.col_to_keep)

    model_params = {
        "vital_signs": 6 if wandb.config.col_to_keep is None else len(wandb.config.col_to_keep),
        "time_samples": wandb.config.data_minutes // wandb.config.data_freq_min,
        "num_channels": wandb.config.num_channels,
        "kernel_size": wandb.config.kernel_size,
        "dense_layers": wandb.config.dense_layers,
        "output_size": 1,
        "max_pool": wandb.config.max_pooling,
    }
    #get model with specified parameters
    model = tf_tcn.get_TCN(**model_params)
    total_params = model.count_params()
    wandb.log({"total_params": total_params})
    logger.info(f"Total number of parameters: {total_params}")

    logger.info(model.summary())
    
    #train model
    model = train_tf.train_model(model, train_ds, wandb.config.epochs, wandb.config.lr, wandb.config.step_lr_epoch_div, wandb.config.step_lr_div_factor, wandb.config.weight_decay)
    model.evaluate(test_ds)

    model_save_path = os.path.join(current_script_folder, "trained_models", wandb.config.model + "_" + wandb.run.id)
    os.makedirs(model_save_path)
    model.save_weights(os.path.join(model_save_path, "model_float.h5"))

    #Evaluate float model
    test_float_model(model, test_ds, model_save_path, wandb.config.online_test, test_ids, wandb.config.online_threshold, wandb.config.online_stride)


    # Quantize model to int8
    quantized_model = convert(model, rep_ds, model_save_path)

    # Evaluate quantized model
    evaluate_quantized_model(quantized_model, test_ds, model_save_path, wandb.config.online_test, test_ids, wandb.config.online_threshold, wandb.config.online_stride)

    # Save the file as a C source file
    bash_command = f"xxd -i {os.path.join(model_save_path, 'quant_model.tflite')} > {os.path.join(model_save_path, 'quant_model.cc')}"
    result = subprocess.run(bash_command, shell=True, capture_output=True, text=True)
    print(result)