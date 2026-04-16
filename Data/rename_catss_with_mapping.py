# Project: Modeling Psychotic Risk from Substance Use in Adolescents
# File title: Create a renamed dataset with available mappings for all the columns
# Author: Silpa Soni Nallacheruvu
# Date: 15/3/2026

import pandas as pd
import json

'''
Create a new dataset from coded columns to readable column names
'''
def rename_cols_with_mapping():
    catss_merged = pd.read_csv("Z:/src/CATSS_Data/catss_merged.csv")
    print(f"The shape of catss data : {catss_merged.shape}")

    with open("catss_column_mapping.json", "r") as f:
        catss_col_mappings = json.load(f)

    catss_merged_renamed = catss_merged.rename(columns=catss_col_mappings)
    print(f"The shape of renamed catss data : {catss_merged_renamed.shape}")
    print(catss_merged_renamed.head())
    catss_merged_renamed.to_csv("catss_merged_renamed.csv", index=False)


'''
Create a new dataset adding a prefix of modality to the column names for easier data processing
'''
def regroup_cols_with_prefix():
    catss = pd.read_csv("catss_final_data.csv", sep="\t", na_values=["NA"], keep_default_na=True)
    print(f"The shape of catss data : {catss.shape}")

    prefix_mapping = {
    "SUD15" : ["Cigarettes15", "Snuff15", "Alcohol15", "Cannabis15", "OtherDrugs15" , "Painkillers_opioids15"],
    "SCZ15" : ["PROD_seen_hallucinations9", "Spied15", "Others_Read_thoughts15", "Special_messages15", "Special_powers15", "Under_control_special_power15", "Read_others_minds15", "Seen_hallucinations15", 
    "Extreme_excitement15", "Irritable15", "Unrealistic_abilities15", "Not_tired15", "Too_much_energy15", "Racing_thoughts15", "Talking_fast15", "Sexual_inappropriate15", "Rage_attacks15", "Hear_voices15",
    "headaches15", "worry15", "unhappy15", "lose_confidence15", "easily_scared15"],
    "ACE15" : ["other_bullying15", "bullied_often15", "tease_bullying15", "emotional_bullying15", "rumours_bullying15", "bullying_by_num15", "bullying_time15" ],
    "ACE18" : ["other_abuse18", "hate_crime18", "emotional_abuse18", "witness_crime18"],
    "SUD18" : ["cigarettes18", "snuff18", "alcohol_often18", "drugs_often18"],
    "SES" : ["education_father", "birth_country_father", "education_mother", "birth_country_mother"],
    "SCZ18" : ["hyper_trouble18", "irritable18", "more_confidence18", "not_tired18", "talking_fast18", "racing_thoughts18", "distracted18", "more_energy18", "unusually_active18", "unusual_social18", "unusual_sex_drive18", "risky_unusual18", "unusual_money_trouble18",
                    "unrealistic_abilities18", "talk_fast18", "sexual_inappropriate18", "hear_voices18", 
                    "poor_appetite18", "depressed18", "felt_effort18", "restless18", "unhappy18", "lonely18", "others_unfriendly18", "not_enjoyed_life18", "sad18", "people_dislike_me18", "could_not_get_going18",
                    "spied18", "read_thoughts18",  "Special_messages18", "special_powers18", "under_control_special18", "read_others_mind18", "seen_hallucinations18"]
    }

    catss = catss.rename(columns={
        col: f"{prefix}_{col}"
        for prefix, cols in prefix_mapping.items()
        for col in cols
        if col in catss.columns
    })

    catss.columns = catss.columns.str.replace(r"^SCORE_", "PRS_", regex=True)
    catss.columns = catss.columns.str.replace(r"^ADHD_", "ADHD9_", regex=True)
    catss.columns = catss.columns.str.replace(r"^ASD_", "ASD9_", regex=True)
    catss = catss.rename(columns={"sex": "SEX"})


    print(f"Columns after adding modality prefix: {catss.columns.tolist()}")
    catss.to_csv("catss_modalities.csv", index=False)
    

'''
Create a new dataset with aggregated positive and negative output columns easier for regression and models comparison
'''
def aggregated_outcome_cols():
    catss = pd.read_csv("catss_modalities.csv")
    print(f"The shape of catss data : {catss.shape}")
    SCZ18_Pos_cols = ["SCZ18_hyper_trouble18", "SCZ18_irritable18", "SCZ18_more_confidence18", "SCZ18_not_tired18", "SCZ18_talking_fast18", "SCZ18_racing_thoughts18", "SCZ18_distracted18", "SCZ18_more_energy18", "SCZ18_unusually_active18", "SCZ18_unusual_social18", "SCZ18_unusual_sex_drive18", "SCZ18_risky_unusual18", "SCZ18_unusual_money_trouble18",
                    "SCZ18_unrealistic_abilities18", "SCZ18_talk_fast18", "SCZ18_sexual_inappropriate18", "SCZ18_hear_voices18",
                    "SCZ18_spied18", "SCZ18_read_thoughts18",  "SCZ18_Special_messages18", "SCZ18_special_powers18", "SCZ18_under_control_special18", "SCZ18_read_others_mind18", "SCZ18_seen_hallucinations18"]
    
    SCZ18_Neg_cols = ["SCZ18_poor_appetite18", "SCZ18_depressed18", "SCZ18_felt_effort18", "SCZ18_restless18", "SCZ18_unhappy18", "SCZ18_lonely18", "SCZ18_others_unfriendly18", "SCZ18_not_enjoyed_life18", "SCZ18_sad18", "SCZ18_people_dislike_me18", "SCZ18_could_not_get_going18"]

    catss["SCZ18_Pos"] = catss[SCZ18_Pos_cols].sum(axis=1)
    catss["SCZ18_Neg"] = catss[SCZ18_Neg_cols].sum(axis=1)

    catss = catss.drop(columns=SCZ18_Pos_cols).copy()
    catss = catss.drop(columns=SCZ18_Neg_cols).copy()

    print(f"The shape of updated catss data : {catss.shape}")
    print(f"Columns after aggregating outputs: {catss.columns.tolist()}")
    catss.to_csv("catss_aggregated_output.csv", index=False)



def normalise_outcome_cols():
    catss = pd.read_csv("catss_final_data.csv")
    print(f"The shape of catss data : {catss.shape}")
    catss["SCZ18_Pos_Norm"] = catss["SCZ18_Pos"]/46.0
    catss["SCZ18_Neg_Norm"] = catss["SCZ18_Neg"]/33.0
    print(f"The shape of updated catss data : {catss.shape}")
    print(f"Columns after normalised outputs: {catss.columns.tolist()}")
    catss.to_csv("catss_normalised_output.csv", index=False)



'''
    Expand batch * PC interactions following the R formula:
        factor(batch) * PC1 + factor(batch) * PC2 + ...
    
    Adds:
      - One-hot encoded batch columns (main effect)
      - Interaction columns: batch_k * PC_j for each combination
    
    The original PC columns stay in the dataframe (main effect of PCs).
'''
def add_batch_pc_interactions(pc_cols=['PC1', 'PC2']):

    catss = pd.read_csv("catss_final_data.csv")
    batch_col='batch'

    if pc_cols is None:
        pc_cols = [c for c in catss.columns if c.startswith('PC')]
    
    # One-hot encode batch (drop_first=True to avoid perfect collinearity,
    # matching R's default treatment contrasts)
    batch_dummies = pd.get_dummies(catss[batch_col], prefix='batch', 
                                    drop_first=True, dtype=float)
    
    print(f"Batch dummies columns: {list(batch_dummies.columns)}")
    
    # Compute interactions: each batch dummy * each PC
    interactions = {}
    for batch_level in batch_dummies.columns:
        for pc in pc_cols:
            interaction_name = f"{batch_level}_x_{pc}"
            interactions[interaction_name] = batch_dummies[batch_level] * catss[pc]

    print(f"Batch and PC Interctions: {list(interactions.keys())}")
    
    interactions_df = pd.DataFrame(interactions, index=catss.index)
    
    # Combine: original catss + batch dummies + interactions
    catss_expanded = pd.concat([catss, batch_dummies, interactions_df], axis=1)

    catss_expanded.to_csv("catss_pc_factorised.csv", index=False)
    

add_batch_pc_interactions()



