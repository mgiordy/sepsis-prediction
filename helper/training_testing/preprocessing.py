import pandas as pd
import numpy as np
from datetime import datetime

import os

def peak_remover(df):
    df_replaced = df.copy()
    mask = (df_replaced / df_replaced.shift(1)) > 1.5
    df_replaced[mask] = df_replaced.shift(1)[mask]
    return df_replaced

def get_patients_fs(folder, train_ids):
    subject_ids = pd.read_csv(os.path.join(folder, "0labels.txt"))
    subject_ids["patientid"] = subject_ids["patientid"].astype(str)
    ids = subject_ids[subject_ids["patientid"].isin(train_ids)]
    return ids, subject_ids


def get_time_from_admission_to_sepsis(ids, folder):
    times_before_sepsis = []
    for _, patient in ids[ids["sepsis"]==True].iterrows():
        data = pd.read_csv(os.path.join(folder, str(patient["patientid"]) + ".csv"))
        time_admission = datetime.strptime(data["date_time"].min(), "%Y-%m-%d %H:%M:%S")
        time_sepsis = datetime.strptime(patient["sepsis_time"], "%Y-%m-%d %H:%M:%S")
        time_before_sepsis = time_sepsis - time_admission
        times_before_sepsis.append(time_before_sepsis)
    times_before_sepsis = np.array(times_before_sepsis)
    return times_before_sepsis