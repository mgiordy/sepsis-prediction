import numpy as np

def sofa_respiration(row):
    sofa = 0
    row = row["oxygen_partial_pressure_in_arterial_blood"]
    if row == 0:
        # Measure not given
        sofa = 0
    elif row >= 400:
        sofa = 0
    elif row < 400 and row >= 300:
        sofa = 1
    elif row < 300 and row >= 200:
        sofa = 2
    elif row < 200 and row >= 100:
        sofa = 3
    elif row < 100:
        sofa = 4
    return sofa

def sofa_coagulation(row):
    sofa = 0
    row = row["platelets_volume_in_blood"]
    if row == 0:
        # Measure not given
        sofa = 0
    elif row >= 150:
        sofa = 0
    elif row < 150 and row >= 100:
        sofa = 1
    elif row < 100 and row >= 50:
        sofa = 2
    elif row < 50 and row >= 20:
        sofa = 3
    elif row < 20:
        sofa = 4
    return sofa

def sofa_liver(row):
    sofa = 0
    row = row["bilirubintotal_molesvolume_in_serum_or_plasma"]
    if row == 0:
        # Measure not given
        sofa = 0
    elif row <= 20:
        sofa = 0
    elif row > 20 and row <= 32:
        sofa = 1
    elif row > 32 and row <= 101:
        sofa = 2
    elif row > 101 and row <= 204:
        sofa = 3
    elif row > 204:
        sofa = 4
    return sofa

def sofa_cardiovascular(cardiovascular_data):
    sofa = 0
    if cardiovascular_data["map"] == 0:
        # Measure not given
        sofa = 0
    elif cardiovascular_data["map"] >= 70:
        sofa = 0
    elif cardiovascular_data["map"] < 70:
        sofa = 1

    if all(drug in cardiovascular_data.index for drug in ["dobutamine", "epinephrine", "norepinephrine"]):
        if cardiovascular_data["dobutamine"] != 0:
            sofa = 2
        if (cardiovascular_data["epinephrine"] > 0 and cardiovascular_data["epinephrine"] <= 0.1) or (cardiovascular_data["norepinephrine"] > 0 and cardiovascular_data["norepinephrine"] <= 0.1):
            sofa = 3
        if cardiovascular_data["epinephrine"] > 0.1 or cardiovascular_data["norepinephrine"] > 0.1:
            sofa = 4

    return sofa

def sofa_cns(row):
    sofa = 0
    row = row["glagsow_coma_score"]
    if row == 0:
        # Measure not given
        sofa = 0
    elif row >= 15:
        sofa = 0
    elif row < 15 and row >= 13:
        sofa = 1
    elif row < 13 and row >= 10:
        sofa = 2
    elif row < 10 and row >= 6:
        sofa = 3
    elif row < 6 and row > 0:
        sofa = 4
    return sofa

def sofa_renal(renal_data):
    sofa = 0
    creatinine = renal_data["creatinine"]
    hourly_urine_volume = renal_data["hourly_urine_volume"]
    if creatinine == 0:
        # Measure not given
        sofa = 0
    elif creatinine <= 110:
        sofa = 0
    elif creatinine > 110 and creatinine <= 170:
        sofa = 1
    elif creatinine > 170 and creatinine <= 299:
        sofa = 2
    elif creatinine > 299 and creatinine <= 440:
        sofa = 3
    elif creatinine > 440:
        sofa = 4 
    
    if hourly_urine_volume != 0:
        if hourly_urine_volume < 500:
            sofa = 3
        if hourly_urine_volume < 200:
            sofa = 4
    return sofa


def get_sofa(data, pharma):
    # Generating sub-dataframes for each organ system
    if pharma.shape[0] > 0:
        # Renaming column for merging later
        pharma = pharma.rename(columns={"givenat": "date_time"})

    fraction_inspired_oxygen = data["inspired_oxygen_concentration"]/100 # Expressed as %
    fraction_inspired_oxygen = fraction_inspired_oxygen.replace(0, 0.21) # 0.21 at room air if not given
    respiration = data["oxygen_partial_pressure_in_arterial_blood"]/fraction_inspired_oxygen # mmHg
    respiration = respiration.to_frame('oxygen_partial_pressure_in_arterial_blood')

    coagulation = data["platelets_volume_in_blood"].to_frame() # G/l -> 10^9*L -> 10^3/ul

    liver = data["bilirubintotal_molesvolume_in_serum_or_plasma"].to_frame() # umol/l
    
    df_cardiovascular = data["date_time"].to_frame()
    df_cardiovascular["map"] = data[["invasive_mean_arterial_pressure", "non_invasive_mean_arterial_pressure"]].apply(lambda row: np.mean(row[row.notnull()]), axis=1) # mmHg
    if pharma.shape[0] > 0:
        df_cardiovascular_pharma = pharma["date_time"].to_frame()
        # No dopamine?
        df_cardiovascular_pharma["dobutamine"] = pharma["dobutrex_250_mg20ml"] # mg
        df_cardiovascular_pharma["epinephrine"] = pharma[["adrenalin__1mgml", "adrenalin_100_µgml_bolus", "adrenalin_20_µgml_perfusor", "adrenalin_100_µgml_perfusor", "adrenalin_10_µgml_bolus"]].sum(axis=1) # ug
        df_cardiovascular_pharma["norepinephrine"] = pharma[["noradrenalin_1mgml", "noradrenalin_100_µgml_perfusor", "noradrenalin_20_µgml_perfusor", "noradrenalin_10_µgml_bolus"]].sum(axis=1) # ug
        # Sofa requires dose in ug/kg/min. Not clear unit of Hirid, assuming ug
        # Weight not given, assuming 70kg
        weight = 70
        df_cardiovascular_pharma["epinephrine"] = df_cardiovascular_pharma["epinephrine"]/weight/60
        df_cardiovascular_pharma["norepinephrine"] = df_cardiovascular_pharma["norepinephrine"]/weight/60
        df_cardiovascular = df_cardiovascular.merge(df_cardiovascular_pharma, on=["date_time"], how='left').fillna(0)

    df_cns = data["date_time"].to_frame()
    df_cns["glagsow_coma_score"] = data[["glasgow_coma_score_verbal_response_subscore", "glasgow_coma_score_motor_response_subscore", "glasgow_coma_score_eye_opening_subscore"]].sum(axis=1) # Absolute number

    df_renal = data["date_time"].to_frame()
    df_renal["creatinine"] = data["creatinine_molesvolume_in_blood"].to_frame() # umol/l
    # Must resample to day summing the output
    urine_temp = data[["date_time", "hourly_urine_volume"]].resample("1D", on="date_time").sum().resample("1H").nearest().reset_index()
    df_renal = df_renal.merge(urine_temp, on=["date_time"], how='left').fillna(0)

    # Computing the SOFA score per subsytem
    sofa_score = data["date_time"].to_frame()
    data_list = [respiration, coagulation, liver, df_cardiovascular, df_cns, df_renal]
    function_list = [sofa_respiration, sofa_coagulation, sofa_liver, sofa_cardiovascular, sofa_cns, sofa_renal]
    for data_to_process, function in zip(data_list, function_list):
        temp = 0
        temp = data_to_process.apply(function, axis=1, result_type='broadcast')
        sofa_score = sofa_score.join(temp.iloc[:,-1].astype(float))

    # Summing the scores
    sofa_scor_value = sofa_score.sum(axis=1, numeric_only = True).to_frame()
    sofa = data["date_time"].to_frame().reset_index()
    sofa = sofa.join(sofa_scor_value)
    return sofa