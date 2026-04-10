# Project: Modeling Psychotic Risk from Substance Use in Adolescents
# File title: Perform feature selection by excluding a list of variables
# Author: Silpa Soni Nallacheruvu
# Date: 17/3/2026

# Load the modules 
import pandas as pd
import numpy as np


# Load the latest dataset
catss = pd.read_csv("catss_final_data.csv", sep="\t", na_values=["NA"], keep_default_na=True)

def get_high_missingness_columns(catss):
    # Check if any columns with high missingness
    missing_percentage = catss.isna().mean()
    high_missing_info = missing_percentage[missing_percentage > 0.9]
    print(high_missing_info)
    # save the missing information to a txt file
    high_missing_info.to_csv("high_missingness_columns.txt", sep="\t", header=False)

    # get columns with high missingness
    high_missing_columns = high_missing_info.index.tolist()
    print(f"Number of cols with missingness more than 90 percent : {len(high_missing_columns)}")
    columns_excluded = pd.DataFrame({
        "features": high_missing_columns,
       "reason": "Missingness > 90 percent"
        })

    # Save the columns excluded
    columns_excluded.to_csv("excluded_columns.csv", index=False)
    return high_missing_columns


def drop_columns(catss, high_missing_columns):
    # Drop these columns
    catss = catss.drop(columns=high_missing_columns)
    return catss.copy()

def save_catss(catss):
    catss.to_csv("catss_selected.csv", index=False)


def get_high_correlated_columns(catss, threshold):
    # Check if any columns with high correlation (> 0.8 as default)
    # exclude cmpair and cmtwin ids first
    catss = catss.drop(columns=['cmpair', 'cmtwin'])
    corr_matrix = catss.corr(method='pearson').abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr_pairs = (
        upper.stack()
            .reset_index()
    )
    high_corr_pairs.columns = ["feature_1", "feature_2", "correlation"]
    high_corr_pairs = high_corr_pairs[
        high_corr_pairs["correlation"] > threshold
    ]
    high_corr_pairs = high_corr_pairs.sort_values(
        by="correlation", ascending=False
    )
    high_corr_pairs.to_csv("high_correlation_pairs.csv", index=False)

def regroup_columns(catss, columns, new_col, drop_columns):
    catss[columns] = catss[columns].replace(-1, np.nan)
    catss[new_col] = catss[columns].max(axis=1)
    catss = catss.drop(columns=drop_columns)
    return catss.copy()


def dropna(catss, columns):
    catss = catss.dropna(subset=columns)
    return catss.copy()

def group_columns(catss, category):
    if category == "Drugs15":
        '''
        All different types of drugs except Cannabis and Painkillers were as highly correlated as ~ 95%. 
        They were all grouped into one variable called 'OtherDrugs15' using 'max' operator on the categorical data. 
        Beer and Alcohol were also grouped into Alcohol using 'max' operator due to high correlation.
        '''
        exposure = ["Cigarettes15", "Snuff15", "Beer15", "Alcohol15", "Cannabis15", "Amphetamine_Stimulants15", "Heroin_opioids15", "Morphine_opioids15", "Cocaine_Stimulants15", "LSD_Psychedelics15", "Ecstasy_Stimulants15", "Mushrooms_Psychedelics15", "Sniffed_gas_Inhalants15", "GHB_perf_enhancers15", "Anabolic_steroids_perf_enhancers15", "Sleeping_pills_sedatives15", "Painkillers_opioids15"]
        catss[exposure] = catss[exposure].replace(-1, np.nan)
        catss["Alcohol15"] = catss[["Beer15", "Alcohol15"]].max(axis=1)
        catss = catss.drop(columns=["Beer15"])

        catss["OtherDrugs15"] = catss[["Amphetamine_Stimulants15", "Heroin_opioids15", "Morphine_opioids15", "Cocaine_Stimulants15", "LSD_Psychedelics15", "Ecstasy_Stimulants15", "Mushrooms_Psychedelics15", "Sniffed_gas_Inhalants15", "GHB_perf_enhancers15", "Anabolic_steroids_perf_enhancers15", "Sleeping_pills_sedatives15"]].max(axis=1)
        catss = catss.drop(columns=["Amphetamine_Stimulants15", "Heroin_opioids15", "Morphine_opioids15", "Cocaine_Stimulants15", "LSD_Psychedelics15", "Ecstasy_Stimulants15", "Mushrooms_Psychedelics15", "Sniffed_gas_Inhalants15", "GHB_perf_enhancers15", "Anabolic_steroids_perf_enhancers15", "Sleeping_pills_sedatives15"])
        
        exposure = ["Cigarettes15", "Snuff15", "Alcohol15", "Cannabis15", "OtherDrugs15" , "Painkillers_opioids15"]
        catss = dropna(catss, exposure)

        catss["Alcohol15"] = catss["Alcohol15"].astype(int)
        catss["OtherDrugs15"] = catss["OtherDrugs15"].astype(int)

    elif category == "Symptoms18":
        '''
        The parent-reported symptoms were all highly correlated and had more missingness (~ 30%) compared to self-reported symptoms (~ 15%)
        and therefore had to be dropped from the dataset to ensure power remained in the data.
        '''
        psych_manic_parents = ["read_thoughts_parents18", "Special_messages_parents18", "spied_parents18", "under_control_special_parents18", "read_others_mind_parents18", "special_powers_parents18", "seen_hallucinations_parents18", 
        "hyper_trouble_parents18", "irritable_parents18", "more_confidence_parents18", "not_tired_parents18", "racing_thoughts_parents18", "distracted_parents18", "more_energy_parents18", "unusually_active_parents18", "several_partners_parents18", "unusual_sex_drive_parents18",  "risky_unusual_parents18", "unusual_money_spend_parents18"]
        catss = drop_columns(catss, psych_manic_parents)
        outcome = ["hyper_trouble18", "irritable18", "more_confidence18", "not_tired18", "talking_fast18", "racing_thoughts18", "distracted18", "more_energy18", "unusually_active18", "unusual_social18", "unusual_sex_drive18", "risky_unusual18", "unusual_money_trouble18",
                    "unrealistic_abilities18", "talk_fast18", "sexual_inappropriate18", "hear_voices18", 
                    "poor_appetite18", "depressed18", "felt_effort18", "restless18", "unhappy18", "lonely18", "others_unfriendly18", "not_enjoyed_life18", "sad18", "people_dislike_me18", "could_not_get_going18",
                    "spied18", "read_thoughts18",  "Special_messages18", "special_powers18", "under_control_special18", "read_others_mind18", "seen_hallucinations18"]
        catss = dropna(catss, outcome)

    elif category == "ACE15":
        '''
        High correlation among ("racial_bullying15", "sexual_bullying15", "cyber_bullying15", "money_bullying15", "threaten_bullying15", "physical_bullying15")
        Combined them into one variable - "other_bullying15"
        '''
        bullying = ["racial_bullying15", "sexual_bullying15", "cyber_bullying15", "money_bullying15", "threaten_bullying15", "physical_bullying15"]
        catss = regroup_columns(catss, bullying, "other_bullying15", bullying)
        catss["other_bullying15"] = catss["other_bullying15"].replace(np.nan, -1)
        catss["other_bullying15"] = catss["other_bullying15"].astype(int)

    elif category == "ACE18":
        '''
        High correlation among ("sexual_abuse18", "sexual_assault18", "physical_neglect18", "physical_abuse18", "witness_physical_violence18").
        Combined them into one variable - "other_abuse18"
        '''
        abuse18 = ["sexual_abuse18", "sexual_assault18", "physical_neglect18", "physical_abuse18", "witness_physical_violence18"]
        catss = regroup_columns(catss, abuse18, "other_abuse18", abuse18)
        catss["other_abuse18"] = catss["other_abuse18"].replace(np.nan, -1)
        catss["other_abuse18"] = catss["other_abuse18"].astype(int)
        
    print(f"Grouped and dropped {category}")   
    return catss.copy()



def change_missingness(catss):
    """
    Missing data marked as "Don't know/ Don't want to answer" (998/ 999) has also been marked as missing (-1).
    """
    #catss = catss.replace(998, -1)
    #catss = catss.replace(999, -1)
    catss = catss.replace(-1, np.nan)
    return catss.copy()

def get_missingness(catss):
    # save the missing information to a txt file
    missing_percentage = catss.isna().mean()
    print("missing_percentage after regrouping")
    # save the missing information to a txt file
    missing_percentage.to_csv("missingness_report_after_regrouping.csv", sep="\t", header=False)

def drop_mid_missing_cols(catss):
    # Cols that are too high to be imputed and causing very low complete cases, dropping them as they are not highly correlated to be regrouped 
    catss = drop_columns(catss, ["emotional_abuse_often18", "trauma_hit9", "witness_physical_violence9", "emotional_neglect9", "physical_abuse9", "physical_neglect9", "sexual_touch_trauma9", "witness_crime9", 
    "sexual_abuse9", "hate_crime_parents15", "emotional_abuse_parents15", "sexual_assault_parents15", "sexual_abuse_parents15", "witness_crime_parents15", "Read_others_minds_parents15", "witness_physical_violence_parents15",
    "Seen_hallucinations_parents15", "physical_abuse_parents15", "Others_Read_thoughts_parents15", "physical_neglect_parents15", "spied_parents15", "Special_powers_parents15", "Under_control_special_power_parents15",
    "Unrealistic_abilities_parents15", "Extreme_excitement_parents15", "Special_messages_parents15", "Hear_voices_parents15", "Not_tired_parents15", "Racing_thoughts_parents15", "Sexual_inappropriate_parents15",
    "Too_much_energy_parents15", "Talking_fast_parents15", "Rage_attacks_parents15", "Irritable_parents15", "alcohol_day_count18", "easily_scared_parents15", "unhappy_parents15", "alcohol_lot_often18",
    "headaches_parents15", "worry_parents15", "lose_confidence_parents15"])
    save_catss(catss)

get_missingness(catss)



