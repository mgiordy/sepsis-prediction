DROP TABLE IF EXISTS combined_pharma_table;
CREATE TABLE combined_pharma_table(
	patientid integer,
	givenat timestamp without time zone,
	Noradrenalin_1mgml real,
	Noradrenalin_100_µgml_Perfusor real,
	Noradrenalin_20_µgml_Perfusor real,
	Noradrenalin_10_µgml_Bolus real,
	Adrenalin__1mgml real,
	Adrenalin_100_µgml_Bolus real,
	Adrenalin_20_µgml_Perfusor real,
	Adrenalin_100_µgml_Perfusor real,
	Adrenalin_10_µgml_Bolus real,
	Dobutrex_250_mg20ml real,
	Vasopressin_inj_20_Uml real,
	Vasopressin_inf_04_Uml real,
	Lasix_Bolus_20mg__Amp real,
	Lasix_Perfusor_250_mg_Amp real,
	Furosemide_10_mgml_inj real,
	Lasix_40_mg_tabl real,
	Torem_Tbl_200_mg real,
	Torem_Inj_Lsg_10_mg2_ml real,
	Torem_Tbl_10_mg real,
	Torem_Tbl_5_mg real,
	Metolazon_Tbl_5_mg real,
	Haemofiltration real,
	Penicillin_50_000_Uml real,
	Clamoxyl_Inj_Lsg real,
	Clamoxyl_Inj_Lsg_2g real,
	Augmentin_Tabl_625_mg real,
	Co_Amoxi_Tbl_625_mg real,
	Co_Amoxi_Tbl_1g real,
	Co_Amoxi_12_g_Inj_Lsg real,
	Co_Amoxi_22g_Inf_Lsg real,
	Augmentin_12_Inj_Lsg real,
	Augmentin_Inj_22g real,
	Augmentin_22_Inf_Lsg real,
	Augmentin_AD_Tbl_1_g real,
	Penicillin_G_1_Mio real,
	Kefzol_Inj_Lsg real,
	Kefzol_Stechamp_2g real,
	Cepimex real,
	Cefepime_2g_Amp real,
	Cepimex_Amp_1g real,
	Fortam_1_g_Inj_Lsg real,
	Fortam_2g_Inj_Lsg real,
	Fortam_Stechamp_2g real,
	Rocephin_2g real,
	Rocephin_2_g_Inf_Lsg real,
	Rocephin_1_g_Inf_Lsg real,
	Zinacef_Amp_15_g real,
	Zinat_Tabl_500_mg real,
	Zinacef_inj_100_mgml real,
	Ciproxin_Tbl_250_mg real,
	Ciproxin_Tbl_500_mg real,
	Ciproxin_200_mg100ml real,
	Ciproxin_Infusion_400_mg real,
	Klacid_Tbl_500_mg real,
	Klacid_Amp_500_mg real,
	Dalacin_C_600_Phosphat_Amp real,
	Dalacin_C_Kps_300_mg real,
	Dalacin_C_Phosphat_Inj_Lsg_300_mg real,
	Dalacin_Phosphat_Inj_Lsg_600_mg real,
	Clindamycin_Kps_300_mg real,
	Clindamycin_Posphat_600 real,
	Clindamycin_Posphat_300 real,
	Doxyclin_Tbl_100_mg real,
	Vibravenös_Inj_Lsg_100_mg_5_ml real,
	Erythrocin_Inf_Lsg real,
	Floxapen_Inj_Lsg real,
	Floxapen_Inj_Lsg_2g real,
	Garamycin real,
	SDD_GentamycinPolymyxin_Kps real,
	Tienam_500_mg real,
	Tavanic_Tbl_500_mg real,
	Tavanic_Inf_Lsg_500_mg_100_ml real,
	Meropenem_500_mg real,
	Meropenem_1g real,
	Meronem_1g real,
	Meronem_500_mg real,
	Flagyl_Tbl_500_mg real,
	Metronidazole_tabl_200_mg real,
	Metronidazole_inf_500_mg100ml real,
	Avalox_Filmtbl_400_mg real,
	Avalox_Inf_Lsg_400_mg real,
	Norfloxacin_Filmtbl_400_mg real,
	Noroxin_tabl_400_mg real,
	Tazobac_Inf_4g real,
	Tazobac_2_g_inf real,
	Piperacillin_Tazobactam_225_Inj_Lsg real,
	Rifampicin_Filmtbl_600_mg real,
	Rifampicin_Inf_Lsg real,
	Rimactan_inf_300_mg real,
	Rimactan_Kps_300_mg real,
	Rimactan_Kps_600_mg real,
	Colidimin_Tbl_200_mg real,
	Xifaxan_Tabl_550_mg real,
	Bactrim_Amp_40080_mg_Inf_Lsg real,
	Bactrim_forte_Lacktbl real,
	Bactrim_Inf_Lsg real,
	Obracin_80_mg real,
	Vancocin_oral_Kps_250_mg real,
	Vancocin__Amp_500_mg real,
	Zovirax_inf_250_mg real,
	Foscavir_Inf_Lsg_6000mg250ml real,
	Cymevene_500_mg real,
	Tamiflu_Kps_75_mg real,
	Tamiflu_Kps_30_mg real,
	Tamiflu_Suspension real,
	Valtrex_Tbl_500_mg real,
	Valcyte_Filmtabl_450mg real,
	Glucose_20100ml_Pflege real,
	Glucose_20_100ml real,
	Glucose_40 real,
	Glucose_50 real,
	Glucose_30 real,
	Glucose_20_500ml real,
	Glucose_10 real,
	Glucose_10_0 real,
	Glucose_20 real,
	Insulin_Lantus real,
	Insulin_Mixtard_30_HM real,
	Insulin_Insulatard_HM real,
	Insulin_NovoRapid real,
	Insulin_Actrapid_inj_100_Uml real,
	Morphin_HCL_Perfusor real,
	Morphine_Inj_a_10mg real,
	Morphin_HCl_Lsg_a_20mgml real,
	Dafalgan_tabl_500_mg real,
	Dafalgan_Brausetbl_1g real,
	Dafalgan_Brausetbl_500_mg real,
	Dafalgan_Supp_600_mg real,
	Perfalgan_1g real,
	Perfalgan_500_mg real,
	Dafalgan_Tabl_1g real,
	Dafalgan_ODIS_Schmelz_Tabl_500_mg real,
	Brufen_Filmtbl_200_mg real,
	Brufen_Filmtbl_400_mg real,
	Brufen_Filmtbl_600_mg real,
	Aspirin_cardio_Tbl_300_mg real,
	Aspirin_Tbl_100_mg real,
	Aspirin_Tbl_500_mg real,
	Aggrastat_Inf_Lsg real,
	ReoPro real,
	Freka_MIx_2000_ml real,
	SmofKabiven_NL_1500_ml_FE real,
	SmofKabiven_NL_2000_ml_FE real,
	StructoKabiven_NL_1500_ml_F___E real,
	SmofKabiven_NL1970_ml_F_E real,
	Nutriflex_lipid_spez_1250ml real,
	Nutriflex_lipid_spez_1875ml real,
	Nutriflex_spez_1000ml__F___E real,
	Nutriflex_spez_1000_ml_ohne_Fett real,
	Nutriflex_lipid_spezial_oE real,
	Heparin_Bichsel_500_E_5_ml real,
	Glypressin_Inj_Lsg real,
	Haemopressin_TS_Inj_Lsg real,
	Privigen_20g_Inf_Lsg real,
	Privigen_10g_Inf_Lsg real,
	Simulect_Inj_Lsg real,
	ATG_Fresenius_100mg5_ml real,
	Prograf_Kps_1mg real,
	CellCept_Susp real,
	CellCept_Tbl_500_mg real,
	Sandimmun_Neoral_Trinklsg_100_mg real,
	Sandimmun_Neoral_Kps_100_mg real,
	Prograf_Kps_5_mg real,
	Imurek_Tabl_50_mg real,
	Sandimmun_Neoral_Kps_50_mg real,
	Imurek_Stechamp_50_mg real,
	Sandimmun_Neoral_Kps_25_mg real,
	Prograf_Kps_05mg real,
	Grafalon_Inf_Lsg real);


INSERT INTO combined_pharma_table (patientid, givenat, Noradrenalin_1mgml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000462);
INSERT INTO combined_pharma_table (patientid, givenat, Noradrenalin_100_µgml_Perfusor) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000656);
INSERT INTO combined_pharma_table (patientid, givenat, Noradrenalin_20_µgml_Perfusor) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000657);
INSERT INTO combined_pharma_table (patientid, givenat, Noradrenalin_10_µgml_Bolus) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000658);
INSERT INTO combined_pharma_table (patientid, givenat, Adrenalin__1mgml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (71);
INSERT INTO combined_pharma_table (patientid, givenat, Adrenalin_100_µgml_Bolus) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000750);
INSERT INTO combined_pharma_table (patientid, givenat, Adrenalin_20_µgml_Perfusor) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000649);
INSERT INTO combined_pharma_table (patientid, givenat, Adrenalin_100_µgml_Perfusor) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000650);
INSERT INTO combined_pharma_table (patientid, givenat, Adrenalin_10_µgml_Bolus) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000655);
INSERT INTO combined_pharma_table (patientid, givenat, Dobutrex_250_mg20ml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (426);
INSERT INTO combined_pharma_table (patientid, givenat, Vasopressin_inj_20_Uml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (112);
INSERT INTO combined_pharma_table (patientid, givenat, Vasopressin_inf_04_Uml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (113);
INSERT INTO combined_pharma_table (patientid, givenat, Lasix_Bolus_20mg__Amp) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000747);
INSERT INTO combined_pharma_table (patientid, givenat, Lasix_Perfusor_250_mg_Amp) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (482);
INSERT INTO combined_pharma_table (patientid, givenat, Furosemide_10_mgml_inj) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000232);
INSERT INTO combined_pharma_table (patientid, givenat, Lasix_40_mg_tabl) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (4);
INSERT INTO combined_pharma_table (patientid, givenat, Torem_Tbl_200_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000986);
INSERT INTO combined_pharma_table (patientid, givenat, Torem_Inj_Lsg_10_mg2_ml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000520);
INSERT INTO combined_pharma_table (patientid, givenat, Torem_Tbl_10_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000521);
INSERT INTO combined_pharma_table (patientid, givenat, Torem_Tbl_5_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000522);
INSERT INTO combined_pharma_table (patientid, givenat, Metolazon_Tbl_5_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000428);
INSERT INTO combined_pharma_table (patientid, givenat, Haemofiltration) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (10002508);
INSERT INTO combined_pharma_table (patientid, givenat, Penicillin_50_000_Uml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000233);
INSERT INTO combined_pharma_table (patientid, givenat, Clamoxyl_Inj_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000273);
INSERT INTO combined_pharma_table (patientid, givenat, Clamoxyl_Inj_Lsg_2g) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000601);
INSERT INTO combined_pharma_table (patientid, givenat, Augmentin_Tabl_625_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000272);
INSERT INTO combined_pharma_table (patientid, givenat, Co_Amoxi_Tbl_625_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001095);
INSERT INTO combined_pharma_table (patientid, givenat, Co_Amoxi_Tbl_1g) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001096);
INSERT INTO combined_pharma_table (patientid, givenat, Co_Amoxi_12_g_Inj_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001097);
INSERT INTO combined_pharma_table (patientid, givenat, Co_Amoxi_22g_Inf_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001098);
INSERT INTO combined_pharma_table (patientid, givenat, Augmentin_12_Inj_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000274);
INSERT INTO combined_pharma_table (patientid, givenat, Augmentin_Inj_22g) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000549);
INSERT INTO combined_pharma_table (patientid, givenat, Augmentin_22_Inf_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000812);
INSERT INTO combined_pharma_table (patientid, givenat, Augmentin_AD_Tbl_1_g) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000781);
INSERT INTO combined_pharma_table (patientid, givenat, Penicillin_G_1_Mio) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000284);
INSERT INTO combined_pharma_table (patientid, givenat, Kefzol_Inj_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000299);
INSERT INTO combined_pharma_table (patientid, givenat, Kefzol_Stechamp_2g) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000671);
INSERT INTO combined_pharma_table (patientid, givenat, Cepimex) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000300);
INSERT INTO combined_pharma_table (patientid, givenat, Cefepime_2g_Amp) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000825);
INSERT INTO combined_pharma_table (patientid, givenat, Cepimex_Amp_1g) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000666);
INSERT INTO combined_pharma_table (patientid, givenat, Fortam_1_g_Inj_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000302);
INSERT INTO combined_pharma_table (patientid, givenat, Fortam_2g_Inj_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000830);
INSERT INTO combined_pharma_table (patientid, givenat, Fortam_Stechamp_2g) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (163);
INSERT INTO combined_pharma_table (patientid, givenat, Rocephin_2g) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000304);
INSERT INTO combined_pharma_table (patientid, givenat, Rocephin_2_g_Inf_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000838);
INSERT INTO combined_pharma_table (patientid, givenat, Rocephin_1_g_Inf_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000648);
INSERT INTO combined_pharma_table (patientid, givenat, Zinacef_Amp_15_g) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000305);
INSERT INTO combined_pharma_table (patientid, givenat, Zinat_Tabl_500_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000306);
INSERT INTO combined_pharma_table (patientid, givenat, Zinacef_inj_100_mgml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000234);
INSERT INTO combined_pharma_table (patientid, givenat, Ciproxin_Tbl_250_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000315);
INSERT INTO combined_pharma_table (patientid, givenat, Ciproxin_Tbl_500_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000893);
INSERT INTO combined_pharma_table (patientid, givenat, Ciproxin_200_mg100ml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (176);
INSERT INTO combined_pharma_table (patientid, givenat, Ciproxin_Infusion_400_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000894);
INSERT INTO combined_pharma_table (patientid, givenat, Klacid_Tbl_500_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000317);
INSERT INTO combined_pharma_table (patientid, givenat, Klacid_Amp_500_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000318);
INSERT INTO combined_pharma_table (patientid, givenat, Dalacin_C_600_Phosphat_Amp) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000829);
INSERT INTO combined_pharma_table (patientid, givenat, Dalacin_C_Kps_300_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000320);
INSERT INTO combined_pharma_table (patientid, givenat, Dalacin_C_Phosphat_Inj_Lsg_300_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000321);
INSERT INTO combined_pharma_table (patientid, givenat, Dalacin_Phosphat_Inj_Lsg_600_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000322);
INSERT INTO combined_pharma_table (patientid, givenat, Clindamycin_Kps_300_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001170);
INSERT INTO combined_pharma_table (patientid, givenat, Clindamycin_Posphat_600) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001169);
INSERT INTO combined_pharma_table (patientid, givenat, Clindamycin_Posphat_300) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001168);
INSERT INTO combined_pharma_table (patientid, givenat, Doxyclin_Tbl_100_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001075);
INSERT INTO combined_pharma_table (patientid, givenat, Vibravenös_Inj_Lsg_100_mg_5_ml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000335);
INSERT INTO combined_pharma_table (patientid, givenat, Erythrocin_Inf_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000791);
INSERT INTO combined_pharma_table (patientid, givenat, Floxapen_Inj_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000352);
INSERT INTO combined_pharma_table (patientid, givenat, Floxapen_Inj_Lsg_2g) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000670);
INSERT INTO combined_pharma_table (patientid, givenat, Garamycin) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000363);
INSERT INTO combined_pharma_table (patientid, givenat, SDD_GentamycinPolymyxin_Kps) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000365);
INSERT INTO combined_pharma_table (patientid, givenat, Tienam_500_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000518);
INSERT INTO combined_pharma_table (patientid, givenat, Tavanic_Tbl_500_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000407);
INSERT INTO combined_pharma_table (patientid, givenat, Tavanic_Inf_Lsg_500_mg_100_ml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000408);
INSERT INTO combined_pharma_table (patientid, givenat, Meropenem_500_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001084);
INSERT INTO combined_pharma_table (patientid, givenat, Meropenem_1g) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001086);
INSERT INTO combined_pharma_table (patientid, givenat, Meronem_1g) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000424);
INSERT INTO combined_pharma_table (patientid, givenat, Meronem_500_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000425);
INSERT INTO combined_pharma_table (patientid, givenat, Flagyl_Tbl_500_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000437);
INSERT INTO combined_pharma_table (patientid, givenat, Metronidazole_tabl_200_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (300);
INSERT INTO combined_pharma_table (patientid, givenat, Metronidazole_inf_500_mg100ml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (181);
INSERT INTO combined_pharma_table (patientid, givenat, Avalox_Filmtbl_400_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000854);
INSERT INTO combined_pharma_table (patientid, givenat, Avalox_Inf_Lsg_400_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000855);
INSERT INTO combined_pharma_table (patientid, givenat, Norfloxacin_Filmtbl_400_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001171);
INSERT INTO combined_pharma_table (patientid, givenat, Noroxin_tabl_400_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (326);
INSERT INTO combined_pharma_table (patientid, givenat, Tazobac_Inf_4g) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000483);
INSERT INTO combined_pharma_table (patientid, givenat, Tazobac_2_g_inf) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (405);
INSERT INTO combined_pharma_table (patientid, givenat, Piperacillin_Tazobactam_225_Inj_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000837);
INSERT INTO combined_pharma_table (patientid, givenat, Rifampicin_Filmtbl_600_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001068);
INSERT INTO combined_pharma_table (patientid, givenat, Rifampicin_Inf_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001173);
INSERT INTO combined_pharma_table (patientid, givenat, Rimactan_inf_300_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (186);
INSERT INTO combined_pharma_table (patientid, givenat, Rimactan_Kps_300_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000797);
INSERT INTO combined_pharma_table (patientid, givenat, Rimactan_Kps_600_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (351);
INSERT INTO combined_pharma_table (patientid, givenat, Colidimin_Tbl_200_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001005);
INSERT INTO combined_pharma_table (patientid, givenat, Xifaxan_Tabl_550_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001079);
INSERT INTO combined_pharma_table (patientid, givenat, Bactrim_Amp_40080_mg_Inf_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001198);
INSERT INTO combined_pharma_table (patientid, givenat, Bactrim_forte_Lacktbl) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000507);
INSERT INTO combined_pharma_table (patientid, givenat, Bactrim_Inf_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000508);
INSERT INTO combined_pharma_table (patientid, givenat, Obracin_80_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000519);
INSERT INTO combined_pharma_table (patientid, givenat, Vancocin_oral_Kps_250_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (331);
INSERT INTO combined_pharma_table (patientid, givenat, Vancocin__Amp_500_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (189);
INSERT INTO combined_pharma_table (patientid, givenat, Zovirax_inf_250_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (190);
INSERT INTO combined_pharma_table (patientid, givenat, Foscavir_Inf_Lsg_6000mg250ml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001126);
INSERT INTO combined_pharma_table (patientid, givenat, Cymevene_500_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000362);
INSERT INTO combined_pharma_table (patientid, givenat, Tamiflu_Kps_75_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000765);
INSERT INTO combined_pharma_table (patientid, givenat, Tamiflu_Kps_30_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001099);
INSERT INTO combined_pharma_table (patientid, givenat, Tamiflu_Suspension) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001100);
INSERT INTO combined_pharma_table (patientid, givenat, Valtrex_Tbl_500_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000530);
INSERT INTO combined_pharma_table (patientid, givenat, Valcyte_Filmtabl_450mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000979);
INSERT INTO combined_pharma_table (patientid, givenat, Glucose_20100ml_Pflege) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000746);
INSERT INTO combined_pharma_table (patientid, givenat, Glucose_20_100ml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000544);
INSERT INTO combined_pharma_table (patientid, givenat, Glucose_40) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000545);
INSERT INTO combined_pharma_table (patientid, givenat, Glucose_50) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000567);
INSERT INTO combined_pharma_table (patientid, givenat, Glucose_30) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000060);
INSERT INTO combined_pharma_table (patientid, givenat, Glucose_20_500ml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000835);
INSERT INTO combined_pharma_table (patientid, givenat, Glucose_10) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000022);
INSERT INTO combined_pharma_table (patientid, givenat, Glucose_10_0) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000690);
INSERT INTO combined_pharma_table (patientid, givenat, Glucose_20) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000689);
INSERT INTO combined_pharma_table (patientid, givenat, Insulin_Lantus) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000963);
INSERT INTO combined_pharma_table (patientid, givenat, Insulin_Mixtard_30_HM) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000379);
INSERT INTO combined_pharma_table (patientid, givenat, Insulin_Insulatard_HM) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000381);
INSERT INTO combined_pharma_table (patientid, givenat, Insulin_NovoRapid) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000724);
INSERT INTO combined_pharma_table (patientid, givenat, Insulin_Actrapid_inj_100_Uml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (15);
INSERT INTO combined_pharma_table (patientid, givenat, Morphin_HCL_Perfusor) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000659);
INSERT INTO combined_pharma_table (patientid, givenat, Morphine_Inj_a_10mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000862);
INSERT INTO combined_pharma_table (patientid, givenat, Morphin_HCl_Lsg_a_20mgml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000932);
INSERT INTO combined_pharma_table (patientid, givenat, Dafalgan_tabl_500_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (275);
INSERT INTO combined_pharma_table (patientid, givenat, Dafalgan_Brausetbl_1g) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000471);
INSERT INTO combined_pharma_table (patientid, givenat, Dafalgan_Brausetbl_500_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000472);
INSERT INTO combined_pharma_table (patientid, givenat, Dafalgan_Supp_600_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000473);
INSERT INTO combined_pharma_table (patientid, givenat, Perfalgan_1g) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000489);
INSERT INTO combined_pharma_table (patientid, givenat, Perfalgan_500_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000490);
INSERT INTO combined_pharma_table (patientid, givenat, Dafalgan_Tabl_1g) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000683);
INSERT INTO combined_pharma_table (patientid, givenat, Dafalgan_ODIS_Schmelz_Tabl_500_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000900);
INSERT INTO combined_pharma_table (patientid, givenat, Brufen_Filmtbl_200_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000883);
INSERT INTO combined_pharma_table (patientid, givenat, Brufen_Filmtbl_400_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000884);
INSERT INTO combined_pharma_table (patientid, givenat, Brufen_Filmtbl_600_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000885);
INSERT INTO combined_pharma_table (patientid, givenat, Aspirin_cardio_Tbl_300_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000255);
INSERT INTO combined_pharma_table (patientid, givenat, Aspirin_Tbl_100_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000256);
INSERT INTO combined_pharma_table (patientid, givenat, Aspirin_Tbl_500_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000257);
INSERT INTO combined_pharma_table (patientid, givenat, Aggrastat_Inf_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001102);
INSERT INTO combined_pharma_table (patientid, givenat, ReoPro) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000614);
INSERT INTO combined_pharma_table (patientid, givenat, Freka_MIx_2000_ml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000714);
INSERT INTO combined_pharma_table (patientid, givenat, SmofKabiven_NL_1500_ml_FE) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000805);
INSERT INTO combined_pharma_table (patientid, givenat, SmofKabiven_NL_2000_ml_FE) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000806);
INSERT INTO combined_pharma_table (patientid, givenat, StructoKabiven_NL_1500_ml_F___E) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000807);
INSERT INTO combined_pharma_table (patientid, givenat, SmofKabiven_NL1970_ml_F_E) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000852);
INSERT INTO combined_pharma_table (patientid, givenat, Nutriflex_lipid_spez_1250ml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000558);
INSERT INTO combined_pharma_table (patientid, givenat, Nutriflex_lipid_spez_1875ml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000559);
INSERT INTO combined_pharma_table (patientid, givenat, Nutriflex_spez_1000ml__F___E) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000560);
INSERT INTO combined_pharma_table (patientid, givenat, Nutriflex_spez_1000_ml_ohne_Fett) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000561);
INSERT INTO combined_pharma_table (patientid, givenat, Nutriflex_lipid_spezial_oE) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000694);
INSERT INTO combined_pharma_table (patientid, givenat, Heparin_Bichsel_500_E_5_ml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000373);
INSERT INTO combined_pharma_table (patientid, givenat, Glypressin_Inj_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000793);
INSERT INTO combined_pharma_table (patientid, givenat, Haemopressin_TS_Inj_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001061);
INSERT INTO combined_pharma_table (patientid, givenat, Privigen_20g_Inf_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001026);
INSERT INTO combined_pharma_table (patientid, givenat, Privigen_10g_Inf_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001025);
INSERT INTO combined_pharma_table (patientid, givenat, Simulect_Inj_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000757);
INSERT INTO combined_pharma_table (patientid, givenat, ATG_Fresenius_100mg5_ml) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000276);
INSERT INTO combined_pharma_table (patientid, givenat, Prograf_Kps_1mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000510);
INSERT INTO combined_pharma_table (patientid, givenat, CellCept_Susp) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000730);
INSERT INTO combined_pharma_table (patientid, givenat, CellCept_Tbl_500_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000448);
INSERT INTO combined_pharma_table (patientid, givenat, Sandimmun_Neoral_Trinklsg_100_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000314);
INSERT INTO combined_pharma_table (patientid, givenat, Sandimmun_Neoral_Kps_100_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000311);
INSERT INTO combined_pharma_table (patientid, givenat, Prograf_Kps_5_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000511);
INSERT INTO combined_pharma_table (patientid, givenat, Imurek_Tabl_50_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000281);
INSERT INTO combined_pharma_table (patientid, givenat, Sandimmun_Neoral_Kps_50_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000312);
INSERT INTO combined_pharma_table (patientid, givenat, Imurek_Stechamp_50_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000280);
INSERT INTO combined_pharma_table (patientid, givenat, Sandimmun_Neoral_Kps_25_mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000313);
INSERT INTO combined_pharma_table (patientid, givenat, Prograf_Kps_05mg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1000942);
INSERT INTO combined_pharma_table (patientid, givenat, Grafalon_Inf_Lsg) SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (1001070);

CREATE INDEX combined_pharma_table_index ON combined_pharma_table (patientid)