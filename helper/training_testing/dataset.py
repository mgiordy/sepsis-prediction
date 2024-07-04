import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler
import os
import sys

from helper import logging_conf
logger = logging_conf.get_logger("dataset.py")

from helper.collate import collate_nn
from helper.dataset_generation.common import resample_data, get_window

# %%
class Dataset(torch.utils.data.Dataset):
  # Torch data loader
  def __init__(self, ids, folder, stats, k, frequency, data_minutes, prediction_minutes, online, online_stride, control_window_cutoff, col_to_keep=None):
    'Initialization'
    self.ids = ids
    self.folder = folder

    self.stats = stats
    self.k = k

    self.frequency = frequency
    self.data_minutes = data_minutes
    self.prediction_minutes = prediction_minutes

    self.online = online
    self.stride = online_stride // frequency # From minutes to number of samples
    self.control_window_cutoff = control_window_cutoff

    self.col_to_keep = col_to_keep

  def __len__(self):
    'Denotes the total number of samples'
    return self.ids.shape[0]

  def __getitem__(self, index):
    'Generates one sample of data'
    id = self.ids.iloc[index]
    data = pd.read_csv(os.path.join(self.folder, id["path"] + ".csv"))

    data['time'] = pd.to_datetime(data['time'])

    sepsis = pd.to_datetime(id["sepsis_time"]) if id["sepsis"] else -1
    vital_cols = ["heartrate","spo2","respiration","temperature","sbp","dbp","mbp"]

    for col in vital_cols:
        dataset = id["path"].split("/")[0]
        stat = self.stats[dataset]
        stat = stat[stat["k"] == self.k]
        q_low = stat[stat["vitalsign"] == col]["q_low"].item()
        q_hi = stat[stat["vitalsign"] == col]["q_hi"].item()
        mu = stat[stat["vitalsign"] == col]["mu"].item()
        std = stat[stat["vitalsign"] == col]["std"].item()
        # Replace zeros with dataset mean
        data[col] = data[col].replace(0, mu)
        # If the value is outside the quantiles, replace it with the mean of the dataset
        data[col] = data[col].apply(lambda x: x if q_low <= x <= q_hi else mu)
        # Normalise the data
        data[col] = (data[col] - mu)/std

    if self.col_to_keep is None:
        self.col_to_keep = ["heartrate","sbp","dbp","spo2","respiration","temperature"]

    data = resample_data(data, str(self.frequency)+"Min", "time", 'mean', "ffill")

    if self.online:
        # Cropping until sepsis. If not recognise before, would be false negative anyway
        if sepsis != -1:
            data = data[data["time"] <= sepsis]
        else:
            # Cutting the data of controls for balancing the windows in training
            # Not cutting during testing to have realistic scenario
            if self.control_window_cutoff > 0:
                data = data[data["time"] <= (data["time"].min() + pd.Timedelta(minutes=self.control_window_cutoff))]
        data = data.loc[:, self.col_to_keep]
        # Aligning the data to integer number of windows
        samples_per_windows = self.data_minutes//self.frequency
        data_lenght = data.shape[0]
        n_windows = ((data_lenght - samples_per_windows) // self.stride) + 1
        data = data.iloc[data_lenght - (samples_per_windows + (n_windows-1)*self.stride):]
        window_tensor = torch.tensor(np.stack([data.iloc[k*self.stride:k*self.stride+samples_per_windows] for k in range(n_windows)]))
        onset_tensor = torch.tensor([0 if sepsis==-1 else 1 for _ in range(n_windows)])
        id_tensor = torch.tensor([id['id'] for _ in range(n_windows)])
    else:
        data = get_window(data, "time", self.data_minutes, self.prediction_minutes, sepsis, self.frequency, padding=False)
        # Col to pass to the neural network
        window_tensor = torch.tensor(data.loc[:, self.col_to_keep].values).unsqueeze_(dim=0)
        onset_tensor = torch.tensor([0 if sepsis==-1 else 1]).unsqueeze_(dim=0)
        id_tensor = torch.tensor([id['id']]).unsqueeze_(dim=0)

    return (window_tensor, onset_tensor, id_tensor)


# This sampler provides the data indices already divided in batched
# Which will be fed to the dataloader
class CustomSampler(Sampler):
    def __init__(self, train_data, batch_size, sampling):
        self.batch_size = batch_size

        # Get the indices of positive and negative samples
        self.ids = train_data.reset_index()
        self.pos_ids = self.ids[self.ids["sepsis"]==True].index.to_numpy()
        self.neg_ids = self.ids[self.ids["sepsis"]==False].index.to_numpy()

        if sampling == "under" or sampling == "undersampling":
            self.steps = int(2*np.floor(self.pos_ids.shape[0]/self.batch_size))
        elif sampling == "over" or sampling == "oversampling":
            self.pos_ids = np.repeat(self.pos_ids, len(self.neg_ids)//len(self.pos_ids), axis=0)
            self.steps = int(2*np.floor(self.pos_ids.shape[0]/self.batch_size))
        else:
            ValueError("Must select either \"under\" or \"over\"")


    def __iter__(self):
        # Yield the indices of each batch
        np.random.shuffle(self.pos_ids)
        np.random.shuffle(self.neg_ids)

        for s in range(self.steps):
            p_id = self.pos_ids[s*int(self.batch_size/2):(s+1)*int(self.batch_size/2)]
            n_id = self.neg_ids[s*int(self.batch_size/2):(s+1)*int(self.batch_size/2)]
            
            y = np.concatenate((p_id, n_id))
            np.random.shuffle(y)

            yield y

    def __len__(self):
        # Return the number of batches
        return self.steps


def get_ids(total_ids, k, k_splits, time_to_sepsis_cutoff):
    # Removing patients that do not have enough data
    total_ids_pos = total_ids[pd.to_timedelta(total_ids["time_to_sepsis"]) >= pd.Timedelta(time_to_sepsis_cutoff, 'minutes')]
    total_ids_neg = total_ids[total_ids["sepsis"] == False]
    total_ids = pd.concat([total_ids_pos, total_ids_neg], axis=0)
    # Shuffling the ids
    total_ids = total_ids.sample(frac = 1)
    # Splitting the ids in k folds
    test_index = total_ids.shape[0]//k_splits
    test_ids = total_ids.iloc[k*test_index:(k+1)*test_index]
    train_ids = total_ids.drop(test_ids.index)

    # Check for train test spillage
    if test_ids['id'].isin(train_ids['id']).any():
        logger.error("Train test spillage detected! Please check your dataset.")
        sys.exit(1)

    return (train_ids, test_ids)


def get_dataloaders(dataset_paths,
                    folder,
                    k,
                    k_splits,
                    datagen_params,
                    custom_sampler,
                    frequency,
                    data_minutes,
                    prediction_minutes,
                    online_train,
                    online_test,
                    online_stride,
                    time_to_sepsis_cutoff,
                    control_window_cutoff,
                    col_to_keep):

    datasets_ids = []
    stats = {}
    # Extracting the subject ids
    for d, d_path in dataset_paths.items():
        ids = pd.read_csv(os.path.join(d_path, "0labels.txt"))
        # Adding folder path to dataframe
        ids["path"] = d + "/" + ids["id"].astype(str)
        datasets_ids.append(ids)
        # Retrieve stats

        stats[d] = pd.read_csv(os.path.join(d_path, "0stats.txt"))

    total_ids = pd.concat(datasets_ids, axis=0)
    # Checking for duplicates
    if total_ids["id"].duplicated().any():
        logger.error("Duplicate ids detected! Please check your dataset.")
        sys.exit(1)

    train_ids, test_ids = get_ids(total_ids, k, k_splits, time_to_sepsis_cutoff)

    if custom_sampler != "":
        train_sampler = CustomSampler(train_ids, datagen_params["batch_size"], custom_sampler)
        train_set = Dataset(train_ids, folder, stats, k, frequency, data_minutes, prediction_minutes, online_train, online_stride, control_window_cutoff, col_to_keep)
        train_generator = torch.utils.data.DataLoader(train_set, num_workers=datagen_params['num_workers'], collate_fn=collate_nn, batch_sampler=train_sampler)
    else:
        train_set = Dataset(train_ids, folder, stats, k, frequency, data_minutes, prediction_minutes, online_train, online_stride, control_window_cutoff, col_to_keep)
        train_generator = torch.utils.data.DataLoader(train_set, **datagen_params, collate_fn=collate_nn)
    
    # Testing always on the actual data
    test_set = Dataset(test_ids, folder, stats, k, frequency, data_minutes, prediction_minutes, online_test, online_stride, control_window_cutoff=-1, col_to_keep=col_to_keep)
    test_generator = torch.utils.data.DataLoader(test_set, **datagen_params, collate_fn=collate_nn)

    return (train_generator, test_generator)