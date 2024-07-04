from sqlalchemy import create_engine, exc, text as sql_text
import sys
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from helper import logging_conf


logger = logging_conf.get_logger("Dataset common")


# %%
def get_sql_engine(db_params):
    try:
        # connect to the PostgreSQL server
        engine = create_engine(
        f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}')
    except exc.SQLAlchemyError as e:
        logger.error("Connection error")
        logger.error(e)
        sys.exit(1) 
    return engine


def get_sql_data(conn_params, query):
    engine = get_sql_engine(conn_params)
    try:
        with engine.connect() as conn:
            # Have to wrap the query in text from SQL-Alchemy >2.0
            data = pd.read_sql_query(sql_text(query), con=conn)
        engine.dispose()
    except exc.SQLAlchemyError as e:
        logger.error("Could not retrieve data from database")
        logger.error(e)
        sys.exit(1)
    return data


# %%
def in_sql_gen(data, begin=None, end=None):
    load = "("
    for i in data[begin:end]:
        load += str(i) + ", "
    load = load[:-2] + ")"
    return load


def data_imputation(data, method):
    if method == "ffill":
        # Filling the NaN with the previous value, next value, and if any left with the zero
        data = data.ffill().bfill().fillna(0)
    elif method == "zero":
        data = data.fillna(0)
    return data


def resample_data(data, freq, time_col, function, imputation=None):
    # Saving column names, removing columns with all nan to avoid warnings
    columns = data.columns
    data = data.dropna(axis=1, how='all')
    # Resampling the data and resetting index from time to integer
    data_res = data.resample(freq, on=time_col).agg(function).reset_index()

    # Restoring columns
    # Empty dataframe when concat treated as NaNs
    df_nan = pd.DataFrame(columns=[c for c in columns if c not in data_res.columns])
    data_res = pd.concat([data_res, df_nan])
    # data_res[[col for col in columns if col not in data_res.columns]] = np.NaN

    # Eventually Imputing missing data
    if imputation is not None:
        data_res = data_imputation(data_res, imputation)
    return data_res


def pad_hours_before(data, freq, start_time, padding):
    # If onset happens before the lenght of data window + prediction time
    end = data["date_time"].min()
    time_pad = pd.date_range(start=start_time, end=end, freq=str(freq)+"Min", inclusive='neither')
    # Adding the new created time, patientid and NaN for the rest
    time_pad = pd.DataFrame([[t_p if c=="date_time" else data["patientid"][0] if c=="patientid" else np.NaN for c in data.columns] for t_p in time_pad.values], columns=data.columns)
    
    data = pd.concat([data, time_pad], ignore_index=True).sort_values("date_time").reset_index(drop=True)
    if padding != -1:
        for col in data.columns:
            data[col] = data[col].replace('nan', np.nan).fillna(padding)
    else:
        data = data.bfill()
    return data


def get_window(data, time_label, data_minutes, prediction_minutes, onset, freq, padding=False):
    # Getting the data_minutes amount of data before prediction_minutes from sepsis onset
    start_data = data[time_label].min()
    end_data = data[time_label].max()
    available_data = onset - start_data if onset != -1 else end_data - start_data
    data_minutes_timedelta = pd.Timedelta(minutes=data_minutes)
    prediction_minutes_timedelta = pd.Timedelta(minutes=prediction_minutes)
    needed_data = pd.Timedelta(minutes=data_minutes+prediction_minutes)


    # If no sepsis the first values from the patient are reported
    if available_data >= needed_data:
        # Data are long enough, we can take the first
        if onset == -1:
            # If no sepsis onset is given, the first data_minutes are taken
            start = start_data
            end = start_data + data_minutes_timedelta
        else:
            start = onset - needed_data
            end = onset - prediction_minutes_timedelta
    else:
        if padding:
            # Data are not enough, will be padded before
            logger.warning(f"Patient {data['id'][0]} has not enough data, padding with {padding}")
            start = end_data - needed_data ####
            end = end_data
            data = pad_hours_before(data, freq, start, -1)
        else:
            # Discarding the patient because not enough data
            logger.error(f"Patient {data['id'][0]} has not enough data, stopping")
            sys.exit(1)
    
    temp = data[data[time_label].between(start, end, inclusive='right')]
    
    return temp


def generate_stats(ids, data_path):
    cols = ["heartrate", "sbp", "dbp", "mbp", "spo2", "respiration", "temperature"]

    datas = []

    for id in tqdm(ids["id"]):
        data = pd.read_csv(os.path.join(data_path, str(id)+".csv"))
        datas.append(data[cols])
    datas = pd.concat(datas, copy=False)

    datas[datas < 0] = 0 # 0 is the lowest value for vital signs
    q_low = datas.quantile(0.01, axis=0, numeric_only=True)
    q_hi = datas.quantile(0.99, axis=0, numeric_only=True)

    for col in cols:
        datas[datas[col] < q_low[col]] = q_low[col]
        datas[datas[col] > q_hi[col]] = q_hi[col]

    mu = datas.mean()
    std = datas.std()

    return q_low, q_hi, mu, std