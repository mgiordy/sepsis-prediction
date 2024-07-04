import numpy as np
import pandas as pd

import torch
import os
from tqdm import tqdm
import wandb

import helper.dataset_generation.hirid_specific.sofa as sofa
from helper.collate import collate_dataset_gen
from helper.dataset_generation.common import in_sql_gen, get_sql_data, resample_data

from helper import logging_conf

logger = logging_conf.get_logger("Hirid dataset generation")

# %%
def get_all_patient_ids(conn_params):
    ###
    # Return list of all patients from SQL database
    ###
    subject_ids = get_sql_data(conn_params, "SELECT * FROM patientid_table")
    # Check to have retrieved all patients:
    number_all_patient = get_sql_data(conn_params, "SELECT COUNT(DISTINCT patientid) FROM general_data")
    assert len(subject_ids["patientid"]) == number_all_patient["count"][0], logger.error("Mismatch in number of patients!")
    # Some subjects have no data
    drop_list = [26045]
    subject_ids = subject_ids[~subject_ids["patientid"].isin(drop_list)]
    return subject_ids


def get_age(conn_params, subject_id):
    age = get_sql_data(conn_params, "SELECT age FROM general_data WHERE patientid in " + in_sql_gen(subject_id))
    return age.loc[0, "age"]


def get_gender(conn_params, subject_id):
    gender = get_sql_data(conn_params, "SELECT sex FROM general_data WHERE patientid in " + in_sql_gen(subject_id))
    return gender.loc[0, "sex"]


def sepsis_labeling(sofa, pharma_table, antibiotics_criteria):
    sofa_diff = sofa.iloc[:,-1] - sofa.iloc[:,-1].shift(1)
    onset = sofa_diff[sofa_diff >= 2].dropna().reset_index()
    if onset.shape[0] == 0:
        return -1
    else:
        # Return the first occurance where SOFA greater than treshold
        onset_index = onset["index"][0]
        sepsis_onset_time = sofa.loc[onset_index,"date_time"]

        suspicion_infection = pharma_table[antibiotics_criteria].any(axis=1)
        suspicion_infection_index = suspicion_infection[suspicion_infection == True].index

        if suspicion_infection_index.shape[0] != 0:
            suspicion_infection_index = suspicion_infection_index[0] # Taking first antibiotic
            si_time = pharma_table.loc[suspicion_infection_index, "givenat"]
            if (sepsis_onset_time > si_time - pd.Timedelta(hours=48)) and (sepsis_onset_time < si_time + pd.Timedelta(hours=24)):
                return sepsis_onset_time
            else:
                return -1
        else:
            return -1


def clean_window(window, data_time, freq, col_interest):
    window["sbp"] = window[["invasive_systolic_arterial_pressure", "non_invasive_systolic_arterial_pressure"]].apply(lambda row: np.mean(row[row.notnull()]), axis=1)
    window["dbp"] = window[["invasive_diastolic_arterial_pressure", "non_invasive_diastolic_arterial_pressure"]].apply(lambda row: np.mean(row[row.notnull()]), axis=1)
    window["mbp"] = window[["invasive_mean_arterial_pressure", "non_invasive_mean_arterial_pressure"]].apply(lambda row: np.mean(row[row.notnull()]), axis=1)
    window["spo2"] = window[["peripheral_oxygen_saturation", "peripheral_oxygen_saturation_0"]].apply(lambda row: np.mean(row[row.notnull()]), axis=1)
    window["glucose"] = window[["glucose_molesvolume_in_serum_or_plasma", "glucose_molesvolume_in_serum_or_plasma_0", "glucose_molesvolume_in_serum_or_plasma_1"]].apply(lambda row: np.mean(row[row.notnull()]), axis=1)
    
    # resample data
    window = window[col_interest]
    window = window.resample(str(freq)+"Min", on="date_time", closed='left').mean().reset_index()
    # Making sure the time dimension is right
    window = window.iloc[0:int(data_time/freq),:]

    return window


def inclusion_criteria(data, pharma, dur_stay, antibiotics, antibiotics_before):
    # Duration of stay > 24h
    length_of_stay = data["date_time"].max() - data["date_time"].min()
    if length_of_stay < dur_stay:
        return 1

    # Check if a column has all missing values
    # if data[inter_columns].eq(0).all().any():
    #     return 1

    # Check whether antibiotic given before time
    if pharma.shape[0] != 0:
        pharma_first_n_hour = pharma[pharma["givenat"] < (data["date_time"].min() + antibiotics_before)]
        if pharma_first_n_hour[antibiotics].ne(0).any().any():
            return 1

    # Sample passed
    return 0


def generate_dataset(data_path, cpu):
    db_params = {
            "host"      : "",
            "database"  : "hirid",
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

    dataset_hirid = Dataset(subject_ids, db_params)
    pbar = tqdm(total=dataset_hirid.__len__(), dynamic_ncols=True)
    dataset_generator = torch.utils.data.DataLoader(dataset_hirid, collate_fn=collate_dataset_gen, **dataloader_params)

    with open(os.path.join(data_path, "0labels.txt"), 'w') as f:
        f.write("id,sepsis,sepsis_time,time_to_sepsis,weight,height,age,gender,ethnicity\n")
    for batch in dataset_generator:
        pbar.update(dataloader_params["batch_size"])
        wandb.log({"Generated patients": pbar.n})
        for patid, wind, ons, time_to_sepsis, inc, age, gender in batch:
            if inc == 0:
                wind.to_csv(os.path.join(data_path, str(patid)+".csv"))
                with open(os.path.join(data_path, "0labels.txt"), 'a') as f:
                    f.write(f"{patid},{ons != -1},{ons},{time_to_sepsis},-1,-1,{age},{gender},-1\n")


# %%
class Dataset(torch.utils.data.Dataset):
  # Torch data loader, used to easily parallelize the data generation
  def __init__(self, patientid, conn_params):
    self.patientid = patientid
    self.conn_params = conn_params

  def __len__(self):
    # Returns the number of patients
    return self.patientid.shape[0]

  def __getitem__(self, index):
    # Generates one patient at a time and saves it to disk

    patientid = self.patientid.iloc[index]

    age = get_age(self.conn_params, patientid)
    gender = get_gender(self.conn_params, patientid)

    load_obs = "SELECT * FROM combined_table WHERE patientid in" + in_sql_gen(patientid)
    load_pharma = "SELECT * FROM combined_pharma_table WHERE patientid in" + in_sql_gen(patientid)

    observation_table = get_sql_data(self.conn_params, load_obs)
    pharma_table = get_sql_data(self.conn_params, load_pharma)

    # Resampling at 1 sample per hour and summing the total amount of given drug
    if(pharma_table.shape[0] > 0):
        pharma_table = pharma_table.fillna(0)
        # NOTE: patientid wrong due to sum
        pharma_table = resample_data(pharma_table, "1H", "givenat", np.sum)

    # Resampling the observations at 1 sample per hour and taking the mean
    observation_table_sofa = resample_data(observation_table, "1H", "date_time", np.nanmean, "ffill")
    sofa_table = sofa.get_sofa(observation_table_sofa, pharma_table)

    antibiotics_criteria = ["penicillin_50_000_uml","clamoxyl_inj_lsg","clamoxyl_inj_lsg_2g","augmentin_tabl_625_mg","co_amoxi_tbl_625_mg","co_amoxi_tbl_1g","co_amoxi_12_g_inj_lsg","co_amoxi_22g_inf_lsg","augmentin_12_inj_lsg","augmentin_inj_22g","augmentin_22_inf_lsg","augmentin_ad_tbl_1_g","penicillin_g_1_mio","kefzol_inj_lsg","kefzol_stechamp_2g","cepimex","cefepime_2g_amp","cepimex_amp_1g","fortam_1_g_inj_lsg","fortam_2g_inj_lsg","fortam_stechamp_2g","rocephin_2g","rocephin_2_g_inf_lsg","rocephin_1_g_inf_lsg","zinacef_amp_15_g","zinat_tabl_500_mg","zinacef_inj_100_mgml","ciproxin_tbl_250_mg","ciproxin_tbl_500_mg","ciproxin_200_mg100ml","ciproxin_infusion_400_mg","klacid_tbl_500_mg","klacid_amp_500_mg","dalacin_c_600_phosphat_amp","dalacin_c_kps_300_mg","dalacin_c_phosphat_inj_lsg_300_mg","dalacin_phosphat_inj_lsg_600_mg","clindamycin_kps_300_mg","clindamycin_posphat_600","clindamycin_posphat_300","doxyclin_tbl_100_mg","vibravenös_inj_lsg_100_mg_5_ml","erythrocin_inf_lsg","floxapen_inj_lsg","floxapen_inj_lsg_2g","garamycin","sdd_gentamycinpolymyxin_kps","tienam_500_mg","tavanic_tbl_500_mg","tavanic_inf_lsg_500_mg_100_ml","meropenem_500_mg","meropenem_1g","meronem_1g","meronem_500_mg","flagyl_tbl_500_mg","metronidazole_tabl_200_mg","metronidazole_inf_500_mg100ml","avalox_filmtbl_400_mg","avalox_inf_lsg_400_mg","norfloxacin_filmtbl_400_mg","noroxin_tabl_400_mg","tazobac_inf_4g","tazobac_2_g_inf","piperacillin_tazobactam_225_inj_lsg","rifampicin_filmtbl_600_mg","rifampicin_inf_lsg","rimactan_inf_300_mg","rimactan_kps_300_mg","rimactan_kps_600_mg","colidimin_tbl_200_mg","xifaxan_tabl_550_mg","bactrim_amp_40080_mg_inf_lsg","bactrim_forte_lacktbl","bactrim_inf_lsg","obracin_80_mg","vancocin_oral_kps_250_mg","vancocin__amp_500_mg"]
    sepsis = sepsis_labeling(sofa_table, pharma_table, antibiotics_criteria)

    inclusion_criteria_param = {
        "data": observation_table,
        "pharma": pharma_table,
        "dur_stay": pd.Timedelta(24, unit='h'),
        "antibiotics": antibiotics_criteria,
        "antibiotics_before": pd.Timedelta(7, unit='h')
    }
    inclusion = inclusion_criteria(**inclusion_criteria_param)

    col_keep = ["patientid", "date_time", "heart_rate", "invasive_systolic_arterial_pressure", "invasive_diastolic_arterial_pressure", "invasive_mean_arterial_pressure", "peripheral_oxygen_saturation", "respiratory_rate", "core_body_temperature"]

    # Keeping only the columns of interest and resetting the index
    observation_table = observation_table[col_keep]

    # Rename the columns for consistency with the other datasets
    observation_table = observation_table.rename(columns={"patientid": "id", "date_time": "time", "heart_rate": "heartrate", "peripheral_oxygen_saturation": "spo2", "invasive_systolic_arterial_pressure": "sbp", "invasive_diastolic_arterial_pressure": "dbp", "invasive_mean_arterial_pressure": "mbp", "respiratory_rate": "respiration", "core_body_temperature": "temperature"})
    observation_table["id"] = observation_table["id"].astype(int)

    observation_table = observation_table.sort_values("time")

    # Saving time to sepsis for sepsis positive patients for inclusion in training
    time_to_sepsis = -1
    if sepsis != -1:
        time_to_sepsis = sepsis - observation_table["time"].min()

    return (patientid.values[0], observation_table, sepsis, time_to_sepsis, inclusion, age, gender)