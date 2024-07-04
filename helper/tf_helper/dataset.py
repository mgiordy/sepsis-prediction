import tensorflow as tf
import numpy as np
from tqdm import tqdm

from helper import logging_conf
from helper.training_testing.dataset import get_dataloaders

logger = logging_conf.get_logger("tf_dataset.py")


def get_datasets(dataset_paths, folder, k, k_splits, datagen_params, custom_sampler, frequency, data_minutes, prediction_minutes, online_train, online_test, online_stride, time_to_sepsis_cutoff, control_window_cutoff, col_to_keep):
    train_generator, test_generator = get_dataloaders(dataset_paths,
                                                    folder,
                                                    k,
                                                    k_splits,
                                                    datagen_params,
                                                    custom_sampler=custom_sampler,
                                                    frequency=frequency,
                                                    data_minutes=data_minutes,
                                                    prediction_minutes=prediction_minutes,
                                                    online_train=online_train,
                                                    online_test=online_test,
                                                    online_stride=online_stride,
                                                    time_to_sepsis_cutoff=time_to_sepsis_cutoff,
                                                    control_window_cutoff=control_window_cutoff,
                                                    col_to_keep=col_to_keep)

    train_data_batch = []
    train_label_batch = []
    for data, label, _ in tqdm(train_generator, dynamic_ncols=True, desc="Loading training data"):
        train_data_batch.append(data.numpy().astype("float32"))
        train_label_batch.append(label.numpy().astype("float32"))

    test_data_batch = []
    test_label_batch = []
    test_ids = []
    for data, label, id in tqdm(test_generator, dynamic_ncols=True, desc="Loading test data"):
        test_data_batch.append(data.numpy().astype("float32"))
        test_label_batch.append(label.numpy().astype("float32"))
        test_ids.append(id.numpy().astype("float32"))

    # Last batch might have different number of samples
    x_train = tf.convert_to_tensor(np.concatenate(train_data_batch))
    y_train = tf.convert_to_tensor(np.concatenate(train_label_batch))
    x_test = tf.convert_to_tensor(np.concatenate(test_data_batch))
    y_test = tf.convert_to_tensor(np.concatenate(test_label_batch))
    test_ids = tf.convert_to_tensor(np.concatenate(test_ids))

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=len(x_train), reshuffle_each_iteration=True).batch(datagen_params["batch_size"])
    rep_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(1)
    # Test dataset is not shuffled, so ids can be matched with predictions
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(datagen_params["batch_size"])

    return train_ds, test_ds, rep_ds, test_ids