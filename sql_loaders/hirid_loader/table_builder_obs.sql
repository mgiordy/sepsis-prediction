DROP TABLE IF EXISTS combined_table;
CREATE TABLE combined_table(
	patientid integer,
	date_time timestamp without time zone,
	Heart_rate real,
	Core_body_temperature real,
	Rectal_temperature real,
	Axillary_temperature real,
	Invasive_systolic_arterial_pressure real,
	Invasive_diastolic_arterial_pressure real,
	Invasive_mean_arterial_pressure real,
	Non_invasive_systolic_arterial_pressure real,
	Non_invasive_diastolic_arterial_pressure real,
	Non_invasive_mean_arterial_pressure real,
	Pulmonary_artery_mean_pressure real,
	Pulmonary_artery_systolic_pressure real,
	Pulmonary_artery_diastolic_pressure real,
	Pulmonary_artery_wedge_pressure real,
	Cardiac_output real,
	Mixed_venous_oxygen_saturation real,
	Central_venous_pressure real,
	Central_venous_pressure_0 real,
	Central_venous_pressure_1 real,
	Peripheral_oxygen_saturation real,
	Peripheral_oxygen_saturation_0 real,
	End_tidal_carbon_dioxide_concentration real,
	End_tidal_carbon_dioxide_concentration_0 real,
	End_tidal_carbon_dioxide_concentration_1 real,
	Respiratory_rate real,
	Respiratory_rate_0 real,
	Respiratory_rate_1 real,
	Oxygen_administration_by_mask real,
	Oxygen_administration_by_nasal_cannula real,
	Inspired_oxygen_concentration real,
	Positive_end_expiratory_pressure_setting real,
	Positive_end_expiratory_pressure real,
	Ventilator_mode real,
	Expiratory_tidal_volume real,
	Tidal_volume_setting real,
	Peak_inspiratory_pressure real,
	Plateau_pressure real,
	Mean_inspiratory_airway_pressure real,
	Ventilator_rate real,
	Ventilator_Airway_Code real,
	Hourly_urine_volume real,
	Glasgow_Coma_Score_verbal_response_subscore real,
	Glasgow_Coma_Score_motor_response_subscore real,
	Glasgow_Coma_Score_eye_opening_subscore real,
	Measurement_of_output_from_drain real,
	Measurement_of_output_from_drain_0 real,
	Measurement_of_output_from_drain_1 real,
	Measurement_of_output_from_drain_2 real,
	Fluid_balance real,
	Fluid_balance_0 real,
	Infusion_of_saline_solution real,
	Intravenous_fluid_colloid_administration real,
	Body_weight real,
	Body_height_measure real,
	APACHE_II_Group real,
	APACHE_IV_Group real,
	CarboxyhemoglobinHemoglobintotal_in_Arterial_blood real,
	Hemoglobin_Massvolume_in_Arterial_blood real,
	Bicarbonate_Molesvolume_in_Arterial_blood real,
	Lactate_Massvolume_in_Arterial_blood real,
	MethemoglobinHemoglobintotal_in_Arterial_blood real,
	pH_of_Arterial_blood real,
	Carbon_dioxide_Partial_pressure_in_Arterial_blood real,
	Oxygen_Partial_pressure_in_Arterial_blood real,
	Oxygen_saturation_in_Arterial_blood real,
	Central_venous_oxygenation_saturation real,
	Central_venous_oxygenation_saturation_0 real,
	Troponin_Tcardiac_Massvolume_in_Serum_or_Plasma real,
	Troponin_Tcardiac_Massvolume_in_Serum_or_Plasma_by_High_se real,
	Creatine_kinase_panel___Serum_or_Plasma real,
	Creatine_kinaseMB_Massvolume_in_Serum_or_Plasma real,
	Lactate_Molesvolume_in_Venous_blood real,
	Lactate_Molesvolume_in_Venous_blood_0 real,
	Natriuretic_peptideB_prohormone_N_Terminal_Massvolume_in_S real,
	Potassium_Molesvolume_in_Blood real,
	Potassium_Molesvolume_in_Blood_0 real,
	Potassium_Molesvolume_in_Blood_1 real,
	Potassium_Molesvolume_in_Blood_2 real,
	Sodium_Molesvolume_in_Blood real,
	Sodium_Molesvolume_in_Blood_0 real,
	Sodium_Molesvolume_in_Blood_1 real,
	Sodium_Molesvolume_in_Blood_2 real,
	Sodium_Molesvolume_in_Blood_3 real,
	Chloride_Molesvolume_in_Blood real,
	Chloride_Molesvolume_in_Blood_0 real,
	Calciumionized_Molesvolume_in_Blood real,
	Calcium_Molesvolume_in_Blood real,
	Phosphate_Molesvolume_in_Blood real,
	Magnesium_Molesvolume_in_Blood real,
	Urea_Molesvolume_in_Venous_blood real,
	Creatinine_Molesvolume_in_Blood real,
	Creatinine_Molesvolume_in_Urine real,
	Creatinine_Molesvolume_in_Urine_0 real,
	Sodium_Molesvolume_in_Urine real,
	Urea_Molesvolume_in_Urine real,
	Aspartate_aminotransferase_Enzymatic_activityvolume_in_Seru real,
	Alanine_aminotransferase_Enzymatic_activityvolume_in_Serum_ real,
	Bilirubintotal_Molesvolume_in_Serum_or_Plasma real,
	Bilirubindirect_Massvolume_in_Serum_or_Plasma real,
	Alkaline_phosphatase_Enzymatic_activityvolume_in_Blood real,
	Gamma_glutamyl_transferase_Enzymatic_activityvolume_in_Seru real,
	aPTT_in_Blood_by_Coagulation_assay real,
	Fibrinogen_Massvolume_in_Platelet_poor_plasma_by_Coagulatio real,
	Prothrombin_activity_actualnormal_in_Platelet_poor_plasma_by_ real,
	Coagulation_factor_V_activity_actualnormal_in_Platelet_poor_p real,
	Coagulation_factor_VII_activity_actualnormal_in_Platelet_poor real,
	Coagulation_factor_X_activity_actualnormal_in_Platelet_poor_p real,
	INR_in_Blood_by_Coagulation_assay real,
	Albumin_Massvolume_in_Serum_or_Plasma real,
	Glucose_Molesvolume_in_Serum_or_Plasma real,
	Glucose_Molesvolume_in_Serum_or_Plasma_0 real,
	Glucose_Molesvolume_in_Serum_or_Plasma_1 real,
	C_reactive_protein_Massvolume_in_Serum_or_Plasma real,
	Procalcitonin_Massvolume_in_Serum_or_Plasma real,
	Lymphocytes_volume_in_Blood real,
	Neutrophils100_leukocytes_in_Blood real,
	Segmented_neutrophils100_leukocytes_in_Blood real,
	Band_form_neutrophils100_leukocytes_in_Blood real,
	Erythrocyte_sedimentation_rate real,
	Hemoglobin_Massvolume_in_Blood real,
	Hemoglobin_Massvolume_in_Blood_0 real,
	Leukocytes_volume_in_Blood real,
	Platelets_volume_in_Blood real,
	MCH_Entitic_mass real,
	MCHC_Massvolume_in_Cord_blood real,
	MCV_Entitic_volume real,
	Ferritin_Massvolume_in_Blood real,
	Thyrotropin_Unitsvolume_in_Serum_or_Plasma real,
	Amylase_Enzymatic_activityvolume_in_Serum_or_Plasma real,
	Lipase_Enzymatic_activityvolume_in_Serum_or_Plasma real,
	Cortisol_Molesvolume_in_Serum_or_Plasma real,
	pH_of_Cerebral_spinal_fluid real,
	Lactate_Molesvolume_in_Cerebral_spinal_fluid real,
	Glucose_Molesvolume_in_Cerebral_spinal_fluid real,
	pH_of_Body_fluid real,
	Amylase_Enzymatic_activityvolume_in_Body_fluid real);


INSERT INTO combined_table (patientid, date_time, Heart_rate) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (200);
INSERT INTO combined_table (patientid, date_time, Core_body_temperature) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (410);
INSERT INTO combined_table (patientid, date_time, Rectal_temperature) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (7100);
INSERT INTO combined_table (patientid, date_time, Axillary_temperature) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (400);
INSERT INTO combined_table (patientid, date_time, Invasive_systolic_arterial_pressure) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (100);
INSERT INTO combined_table (patientid, date_time, Invasive_diastolic_arterial_pressure) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (120);
INSERT INTO combined_table (patientid, date_time, Invasive_mean_arterial_pressure) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (110);
INSERT INTO combined_table (patientid, date_time, Non_invasive_systolic_arterial_pressure) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (600);
INSERT INTO combined_table (patientid, date_time, Non_invasive_diastolic_arterial_pressure) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (620);
INSERT INTO combined_table (patientid, date_time, Non_invasive_mean_arterial_pressure) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (610);
INSERT INTO combined_table (patientid, date_time, Pulmonary_artery_mean_pressure) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (160);
INSERT INTO combined_table (patientid, date_time, Pulmonary_artery_systolic_pressure) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (150);
INSERT INTO combined_table (patientid, date_time, Pulmonary_artery_diastolic_pressure) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (170);
INSERT INTO combined_table (patientid, date_time, Pulmonary_artery_wedge_pressure) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (800);
INSERT INTO combined_table (patientid, date_time, Cardiac_output) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (1000);
INSERT INTO combined_table (patientid, date_time, Mixed_venous_oxygen_saturation) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (4200);
INSERT INTO combined_table (patientid, date_time, Central_venous_pressure) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (700);
INSERT INTO combined_table (patientid, date_time, Central_venous_pressure_0) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (15001441);
INSERT INTO combined_table (patientid, date_time, Central_venous_pressure_1) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (960);
INSERT INTO combined_table (patientid, date_time, Peripheral_oxygen_saturation) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (4000);
INSERT INTO combined_table (patientid, date_time, Peripheral_oxygen_saturation_0) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (8280);
INSERT INTO combined_table (patientid, date_time, End_tidal_carbon_dioxide_concentration) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (2200);
INSERT INTO combined_table (patientid, date_time, End_tidal_carbon_dioxide_concentration_0) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (8290);
INSERT INTO combined_table (patientid, date_time, End_tidal_carbon_dioxide_concentration_1) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (30010009);
INSERT INTO combined_table (patientid, date_time, Respiratory_rate) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (300);
INSERT INTO combined_table (patientid, date_time, Respiratory_rate_0) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (310);
INSERT INTO combined_table (patientid, date_time, Respiratory_rate_1) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (5685);
INSERT INTO combined_table (patientid, date_time, Oxygen_administration_by_mask) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (15001555);
INSERT INTO combined_table (patientid, date_time, Oxygen_administration_by_nasal_cannula) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (15001525);
INSERT INTO combined_table (patientid, date_time, Inspired_oxygen_concentration) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (2010);
INSERT INTO combined_table (patientid, date_time, Positive_end_expiratory_pressure_setting) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (2600);
INSERT INTO combined_table (patientid, date_time, Positive_end_expiratory_pressure) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (2610);
INSERT INTO combined_table (patientid, date_time, Ventilator_mode) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (3845);
INSERT INTO combined_table (patientid, date_time, Expiratory_tidal_volume) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (2410);
INSERT INTO combined_table (patientid, date_time, Tidal_volume_setting) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (2400);
INSERT INTO combined_table (patientid, date_time, Peak_inspiratory_pressure) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (3110);
INSERT INTO combined_table (patientid, date_time, Plateau_pressure) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (3200);
INSERT INTO combined_table (patientid, date_time, Mean_inspiratory_airway_pressure) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (3000);
INSERT INTO combined_table (patientid, date_time, Ventilator_rate) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (320);
INSERT INTO combined_table (patientid, date_time, Ventilator_Airway_Code) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (15001552);
INSERT INTO combined_table (patientid, date_time, Hourly_urine_volume) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (10020000);
INSERT INTO combined_table (patientid, date_time, Glasgow_Coma_Score_verbal_response_subscore) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (10000100);
INSERT INTO combined_table (patientid, date_time, Glasgow_Coma_Score_motor_response_subscore) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (10000200);
INSERT INTO combined_table (patientid, date_time, Glasgow_Coma_Score_eye_opening_subscore) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (10000300);
INSERT INTO combined_table (patientid, date_time, Measurement_of_output_from_drain) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (10010020);
INSERT INTO combined_table (patientid, date_time, Measurement_of_output_from_drain_0) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (10010072);
INSERT INTO combined_table (patientid, date_time, Measurement_of_output_from_drain_1) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (10010070);
INSERT INTO combined_table (patientid, date_time, Measurement_of_output_from_drain_2) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (10010071);
INSERT INTO combined_table (patientid, date_time, Fluid_balance) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (30005010);
INSERT INTO combined_table (patientid, date_time, Fluid_balance_0) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (30005110);
INSERT INTO combined_table (patientid, date_time, Infusion_of_saline_solution) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (30005075);
INSERT INTO combined_table (patientid, date_time, Intravenous_fluid_colloid_administration) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (30005080);
INSERT INTO combined_table (patientid, date_time, Body_weight) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (10000400);
INSERT INTO combined_table (patientid, date_time, Body_height_measure) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (10000450);
INSERT INTO combined_table (patientid, date_time, APACHE_II_Group) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (9990002);
INSERT INTO combined_table (patientid, date_time, APACHE_IV_Group) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (9990004);
INSERT INTO combined_table (patientid, date_time, CarboxyhemoglobinHemoglobintotal_in_Arterial_blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000526);
INSERT INTO combined_table (patientid, date_time, Hemoglobin_Massvolume_in_Arterial_blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000548);
INSERT INTO combined_table (patientid, date_time, Bicarbonate_Molesvolume_in_Arterial_blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20004200);
INSERT INTO combined_table (patientid, date_time, Lactate_Massvolume_in_Arterial_blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000524);
INSERT INTO combined_table (patientid, date_time, MethemoglobinHemoglobintotal_in_Arterial_blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000549);
INSERT INTO combined_table (patientid, date_time, pH_of_Arterial_blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20000300);
INSERT INTO combined_table (patientid, date_time, Carbon_dioxide_Partial_pressure_in_Arterial_blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20001200);
INSERT INTO combined_table (patientid, date_time, Oxygen_Partial_pressure_in_Arterial_blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20000200);
INSERT INTO combined_table (patientid, date_time, Oxygen_saturation_in_Arterial_blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20000800);
INSERT INTO combined_table (patientid, date_time, Central_venous_oxygenation_saturation) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20001000);
INSERT INTO combined_table (patientid, date_time, Central_venous_oxygenation_saturation_0) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000737);
INSERT INTO combined_table (patientid, date_time, Troponin_Tcardiac_Massvolume_in_Serum_or_Plasma) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000538);
INSERT INTO combined_table (patientid, date_time, Troponin_Tcardiac_Massvolume_in_Serum_or_Plasma_by_High_se) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000806);
INSERT INTO combined_table (patientid, date_time, Creatine_kinase_panel___Serum_or_Plasma) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000210);
INSERT INTO combined_table (patientid, date_time, Creatine_kinaseMB_Massvolume_in_Serum_or_Plasma) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000220);
INSERT INTO combined_table (patientid, date_time, Lactate_Molesvolume_in_Venous_blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000732);
INSERT INTO combined_table (patientid, date_time, Lactate_Molesvolume_in_Venous_blood_0) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000485);
INSERT INTO combined_table (patientid, date_time, Natriuretic_peptideB_prohormone_N_Terminal_Massvolume_in_S) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000569);
INSERT INTO combined_table (patientid, date_time, Potassium_Molesvolume_in_Blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20000500);
INSERT INTO combined_table (patientid, date_time, Potassium_Molesvolume_in_Blood_0) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000520);
INSERT INTO combined_table (patientid, date_time, Potassium_Molesvolume_in_Blood_1) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000833);
INSERT INTO combined_table (patientid, date_time, Potassium_Molesvolume_in_Blood_2) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000867);
INSERT INTO combined_table (patientid, date_time, Sodium_Molesvolume_in_Blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20000400);
INSERT INTO combined_table (patientid, date_time, Sodium_Molesvolume_in_Blood_0) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000519);
INSERT INTO combined_table (patientid, date_time, Sodium_Molesvolume_in_Blood_1) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000658);
INSERT INTO combined_table (patientid, date_time, Sodium_Molesvolume_in_Blood_2) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000835);
INSERT INTO combined_table (patientid, date_time, Sodium_Molesvolume_in_Blood_3) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000866);
INSERT INTO combined_table (patientid, date_time, Chloride_Molesvolume_in_Blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000439);
INSERT INTO combined_table (patientid, date_time, Chloride_Molesvolume_in_Blood_0) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000521);
INSERT INTO combined_table (patientid, date_time, Calciumionized_Molesvolume_in_Blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000522);
INSERT INTO combined_table (patientid, date_time, Calcium_Molesvolume_in_Blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20005100);
INSERT INTO combined_table (patientid, date_time, Phosphate_Molesvolume_in_Blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20002500);
INSERT INTO combined_table (patientid, date_time, Magnesium_Molesvolume_in_Blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000230);
INSERT INTO combined_table (patientid, date_time, Urea_Molesvolume_in_Venous_blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20004100);
INSERT INTO combined_table (patientid, date_time, Creatinine_Molesvolume_in_Blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20000600);
INSERT INTO combined_table (patientid, date_time, Creatinine_Molesvolume_in_Urine) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000572);
INSERT INTO combined_table (patientid, date_time, Creatinine_Molesvolume_in_Urine_0) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000573);
INSERT INTO combined_table (patientid, date_time, Sodium_Molesvolume_in_Urine) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20003200);
INSERT INTO combined_table (patientid, date_time, Urea_Molesvolume_in_Urine) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000754);
INSERT INTO combined_table (patientid, date_time, Aspartate_aminotransferase_Enzymatic_activityvolume_in_Seru) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000330);
INSERT INTO combined_table (patientid, date_time, Alanine_aminotransferase_Enzymatic_activityvolume_in_Serum_) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20002600);
INSERT INTO combined_table (patientid, date_time, Bilirubintotal_Molesvolume_in_Serum_or_Plasma) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20004300);
INSERT INTO combined_table (patientid, date_time, Bilirubindirect_Massvolume_in_Serum_or_Plasma) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000560);
INSERT INTO combined_table (patientid, date_time, Alkaline_phosphatase_Enzymatic_activityvolume_in_Blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20002700);
INSERT INTO combined_table (patientid, date_time, Gamma_glutamyl_transferase_Enzymatic_activityvolume_in_Seru) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000370);
INSERT INTO combined_table (patientid, date_time, aPTT_in_Blood_by_Coagulation_assay) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20004410);
INSERT INTO combined_table (patientid, date_time, Fibrinogen_Massvolume_in_Platelet_poor_plasma_by_Coagulatio) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000536);
INSERT INTO combined_table (patientid, date_time, Prothrombin_activity_actualnormal_in_Platelet_poor_plasma_by_) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000666);
INSERT INTO combined_table (patientid, date_time, Coagulation_factor_V_activity_actualnormal_in_Platelet_poor_p) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000637);
INSERT INTO combined_table (patientid, date_time, Coagulation_factor_VII_activity_actualnormal_in_Platelet_poor) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000552);
INSERT INTO combined_table (patientid, date_time, Coagulation_factor_X_activity_actualnormal_in_Platelet_poor_p) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000667);
INSERT INTO combined_table (patientid, date_time, INR_in_Blood_by_Coagulation_assay) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000567);
INSERT INTO combined_table (patientid, date_time, Albumin_Massvolume_in_Serum_or_Plasma) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000605);
INSERT INTO combined_table (patientid, date_time, Glucose_Molesvolume_in_Serum_or_Plasma) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20005110);
INSERT INTO combined_table (patientid, date_time, Glucose_Molesvolume_in_Serum_or_Plasma_0) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000523);
INSERT INTO combined_table (patientid, date_time, Glucose_Molesvolume_in_Serum_or_Plasma_1) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000585);
INSERT INTO combined_table (patientid, date_time, C_reactive_protein_Massvolume_in_Serum_or_Plasma) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20002200);
INSERT INTO combined_table (patientid, date_time, Procalcitonin_Massvolume_in_Serum_or_Plasma) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000570);
INSERT INTO combined_table (patientid, date_time, Lymphocytes_volume_in_Blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000480);
INSERT INTO combined_table (patientid, date_time, Neutrophils100_leukocytes_in_Blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000550);
INSERT INTO combined_table (patientid, date_time, Segmented_neutrophils100_leukocytes_in_Blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000556);
INSERT INTO combined_table (patientid, date_time, Band_form_neutrophils100_leukocytes_in_Blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000557);
INSERT INTO combined_table (patientid, date_time, Erythrocyte_sedimentation_rate) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000668);
INSERT INTO combined_table (patientid, date_time, Hemoglobin_Massvolume_in_Blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20000900);
INSERT INTO combined_table (patientid, date_time, Hemoglobin_Massvolume_in_Blood_0) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000836);
INSERT INTO combined_table (patientid, date_time, Leukocytes_volume_in_Blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20000700);
INSERT INTO combined_table (patientid, date_time, Platelets_volume_in_Blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (20000110);
INSERT INTO combined_table (patientid, date_time, MCH_Entitic_mass) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000160);
INSERT INTO combined_table (patientid, date_time, MCHC_Massvolume_in_Cord_blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000170);
INSERT INTO combined_table (patientid, date_time, MCV_Entitic_volume) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000150);
INSERT INTO combined_table (patientid, date_time, Ferritin_Massvolume_in_Blood) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000678);
INSERT INTO combined_table (patientid, date_time, Thyrotropin_Unitsvolume_in_Serum_or_Plasma) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000664);
INSERT INTO combined_table (patientid, date_time, Amylase_Enzymatic_activityvolume_in_Serum_or_Plasma) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000427);
INSERT INTO combined_table (patientid, date_time, Lipase_Enzymatic_activityvolume_in_Serum_or_Plasma) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000555);
INSERT INTO combined_table (patientid, date_time, Cortisol_Molesvolume_in_Serum_or_Plasma) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000665);
INSERT INTO combined_table (patientid, date_time, pH_of_Cerebral_spinal_fluid) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000608);
INSERT INTO combined_table (patientid, date_time, Lactate_Molesvolume_in_Cerebral_spinal_fluid) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000564);
INSERT INTO combined_table (patientid, date_time, Glucose_Molesvolume_in_Cerebral_spinal_fluid) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000400);
INSERT INTO combined_table (patientid, date_time, pH_of_Body_fluid) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000741);
INSERT INTO combined_table (patientid, date_time, Amylase_Enzymatic_activityvolume_in_Body_fluid) SELECT patientid, date_time, value_obs FROM observation_table where variableid in (24000587);

CREATE INDEX combined_table_index ON combined_table (patientid)

CREATE TABLE combined_table (patientid integer);
INSERT INTO patientid_table (patientid) SELECT DISTINCT patientid from general_data