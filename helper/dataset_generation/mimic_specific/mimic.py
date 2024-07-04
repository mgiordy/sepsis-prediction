import numpy as np
import pandas as pd

import torch
import os
from tqdm import tqdm
import wandb

from helper.collate import collate_dataset_gen
from helper.dataset_generation.common import in_sql_gen, get_sql_data, resample_data, get_window

from helper import logging_conf

logger = logging_conf.get_logger("MIMIC dataset generation")

# %%
def get_all_patient_ids(conn_params):
    ###
    # Return list of all patients from SQL database
    ###

    # Getting subject ids for static data and stay ids for dynamic data
    # Getting from vitalsign to exlude patients without vitals
    stay_ids = get_sql_data(conn_params, "SELECT DISTINCT subject_id, stay_id FROM mimiciv_derived.vitalsign")
    # Check to have retrieved all stays:
    number_all_stays = get_sql_data(conn_params, "SELECT COUNT(DISTINCT stay_id) FROM mimiciv_derived.vitalsign")
    assert len(stay_ids["stay_id"]) == number_all_stays["count"][0], logger.error("Mismatch in number of stays!")

    return stay_ids


def get_age(conn_params, subject_id):
    age = get_sql_data(conn_params, "SELECT anchor_age FROM mimiciv_hosp.patients WHERE subject_id in" + in_sql_gen([subject_id]))
    return age.iloc[0, 0] if age.shape[0] != 0 else -1


def get_gender(conn_params, subject_id):
    gender = get_sql_data(conn_params, "SELECT gender FROM mimiciv_hosp.patients WHERE subject_id in" + in_sql_gen([subject_id]))
    return gender.iloc[0, 0] if gender.shape[0] != 0 else -1

def get_ethnicity(conn_params, subject_id):
    ethnicity = get_sql_data(conn_params, "SELECT race FROM mimiciv_hosp.admissions WHERE subject_id in" + in_sql_gen([subject_id]))
    return ethnicity.iloc[0, 0] if ethnicity.shape[0] != 0 else -1


def get_weight(conn_params, stay_id):
    weight = get_sql_data(conn_params, "SELECT weight FROM mimiciv_derived.weight_durations WHERE stay_id in" + in_sql_gen([stay_id]))
    return weight.iloc[0, 0] if weight.shape[0] != 0 else -1


def get_height(conn_params, stay_id):
    height = get_sql_data(conn_params, "SELECT height FROM mimiciv_derived.height WHERE stay_id in" + in_sql_gen([stay_id]))
    return height.iloc[0, 0] if height.shape[0] != 0 else -1


def get_vitals(conn_params, stay_id):
    vitals = get_sql_data(conn_params, "SELECT * FROM mimiciv_derived.vitalsign WHERE stay_id in" + in_sql_gen([stay_id]))
    return vitals


def get_sepsis3(conn_params, stay_id):
    sepsis3 = get_sql_data(conn_params, "SELECT * FROM mimiciv_derived.sepsis3 WHERE sepsis3 is True and stay_id in" + in_sql_gen([stay_id]))
    return sepsis3


def get_antibiotics(conn_params, stay_id):
    antibiotics = get_sql_data(conn_params, "SELECT stay_id, starttime FROM mimiciv_derived.antibiotic WHERE stay_id in" + in_sql_gen([stay_id]))
    return antibiotics


def inclusion_criteria(data, pharma, age_cut, dur_stay, antibiotics_before):
    # Duration of stay > 24h
    lenght_of_stay = data["charttime"].max() - data["charttime"].min()
    if lenght_of_stay < dur_stay:
        return 1

    # Check if a column has all missing values
    # if data[inter_columns].eq(0).all().any():
    #     return 1

    # Check whether antibiotic given before time
    if pharma.shape[0] != 0:
        if pharma["starttime"].min() < (data["charttime"].min() + antibiotics_before):
            return 1
    
    # Sample passed
    return 0


def generate_dataset(data_path, cpu):
    db_params = {
            "host"      : "",
            "database"  : "mimic",
            "user"      : "",
            "password"  : "",
            'port'      : '5432',
        }

    # batch=1 inside the dataloader since we're getting them from the database
    dataloader_params = {'batch_size': 1,
        'shuffle': False,
        'num_workers': cpu}

    # Getting the all the patient ids
    subject_ids = get_all_patient_ids(db_params)

    dataset_mimic = Dataset(subject_ids["subject_id"], subject_ids["stay_id"], db_params)
    pbar = tqdm(total=dataset_mimic.__len__(), dynamic_ncols=True)
    dataset_generator = torch.utils.data.DataLoader(dataset_mimic, collate_fn=collate_dataset_gen, **dataloader_params)

    with open(os.path.join(data_path, "0labels.txt"), 'w') as f:
        f.write("id,sepsis,sepsis_time,time_to_sepsis,weight,height,age,gender,ethnicity\n")
    for batch in dataset_generator:
        pbar.update(dataloader_params["batch_size"])
        wandb.log({"Generated patients": pbar.n})
        for stay_id, wind, ons, time_to_sepsis, inc, weight, height, age, gender, ethnicity in batch:
            if inc == 0:
                wind.to_csv(os.path.join(data_path, str(stay_id)+".csv"))
                with open(os.path.join(data_path, "0labels.txt"), 'a') as f:
                    f.write(f"{stay_id},{ons != -1},{ons},{time_to_sepsis},{weight},{height},{age},{gender},{ethnicity}\n")


# %%
class Dataset(torch.utils.data.Dataset):
  # Torch data loader, used to easily parallelize the data generation
  def __init__(self, subject_id, stay_id, conn_params):
    self.subject_id = subject_id
    self.stay_id = stay_id
    self.conn_params = conn_params

  def __len__(self):
    # Returns the number of patients
    return self.subject_id.shape[0]

  def __getitem__(self, index):
    # Generates one patient at a time and saves it to disk

    subject_id = self.subject_id.iloc[index]
    stay_id = self.stay_id.iloc[index]

    # Getting the data for the patient. It is not guaranteed to have data
    age = get_age(self.conn_params, subject_id)
    weight = get_weight(self.conn_params, stay_id)
    height = get_height(self.conn_params, stay_id)
    gender = get_gender(self.conn_params, subject_id)
    ethnicity = get_ethnicity(self.conn_params, subject_id)

    # Get vitalsigns
    # Getting hour of stay from the vitalsign table
    vitals = get_vitals(self.conn_params, stay_id)
    vitals = vitals.drop(columns=["temperature_site"]) # Is string, not needed
    sepsis3 = get_sepsis3(self.conn_params, stay_id)
    sepsis3 = sepsis3["sofa_time"][0] if sepsis3.shape[0] != 0 else -1
    antibiotics = get_antibiotics(self.conn_params, stay_id)


    inclusion_criteria_param = {
        "data": vitals,
        "pharma": antibiotics,
        "age_cut": 18, # Every patient is older than 18
        "dur_stay": pd.Timedelta(24, unit='h'),
        "antibiotics_before": pd.Timedelta(7, unit='h')
    }
    inclusion = inclusion_criteria(**inclusion_criteria_param)

    col_keep = ['charttime', 'stay_id', 'heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'temperature', 'spo2']
    vitals = vitals[col_keep]

    # Rename columns for consisntency with other datasets
    vitals = vitals.rename(columns={"stay_id": "id", "charttime": "time", "heart_rate": "heartrate", "resp_rate": "respiration", "core_body_temperature": "temperature"})
    vitals["id"] = vitals["id"].astype(int)
    
    vitals = vitals.sort_values("time")

    # Saving time to sepsis for sepsis positive patients for inclusion in training
    time_to_sepsis = -1
    if sepsis3 != -1:
        time_to_sepsis = sepsis3 - vitals["time"].min()

    return (stay_id, vitals, sepsis3, time_to_sepsis, inclusion, weight, height, age, gender, ethnicity)