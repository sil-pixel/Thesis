# Project: Modeling Psychotic Risk from Substance Use in Adolescents
# File title: Perform LightBoostGBM for feature importance analysis
# Author: Silpa Soni Nallacheruvu
# Date: 20/2/2026

# Load the modules 
import runpy, sys
import pandas as pd
import numpy as np
import lightgbm as lgb 
from lightgbm import LGBMRegressor
from split_training_data import random_split
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import shap 
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
import scipy.spatial.distance as ssd
from scipy import stats


# module load Python/3.11.5-GCCcore-13.2.0
# python3 -m pip install pandas

# Configuration
seeds = [41, 42, 43, 44, 45]  # Use more seeds for better CI estimates
n_seeds = len(seeds)

# Load training and testing dataset for 70:30 data split for X and y 
def catss_train_test_split(outcome_var_list, tag, seed):
    catss_train, catss_test = random_split(0.30, "catss_merged_renamed.csv", seed)

    #catss_train = catss_train.fillna(-1)
    #catss_test = catss_test.fillna(-1)

    catss_train = catss_train.drop(columns=["cmpair", "cmtwin"]).copy()
    catss_test = catss_test.drop(columns=["cmpair", "cmtwin"]).copy()


    catss_train["scz"] = catss_train[outcome_var_list].sum(axis=1, skipna=True)
    catss_test["scz"] = catss_test[outcome_var_list].sum(axis=1, skipna=True)
    catss_train = catss_train.drop(columns=outcome_var_list).copy()
    catss_test = catss_test.drop(columns=outcome_var_list).copy()

    catss_train = catss_train.dropna(subset=["scz"])
    catss_test = catss_test.dropna(subset=["scz"])

    #sns.histplot(catss_train["scz"], bins=30, kde=True)
    #plt.title(f"Distribution of {tag} Outcome in training")
    #plt.xlabel(f"{tag} Outcome")
    #plt.ylabel("Sum of symptoms")
    #plt.savefig(f"{tag}_outcome_train_distribution.png", dpi=300, bbox_inches="tight")
    #plt.close()

    #sns.histplot(catss_test["scz"], bins=30, kde=True)
    #plt.title(f"Distribution of {tag} Outcome in test")
    #plt.xlabel(f"{tag} Outcome")
    #plt.ylabel("Sum of symptoms")
    #plt.savefig(f"{tag}_outcome_test_distribution.png", dpi=300, bbox_inches="tight")
    #plt.close()

    x_train = catss_train.drop(columns=["scz"]).copy()
    y_train = catss_train["scz"].copy()
    x_test = catss_test.drop(columns=["scz"]).copy()
    y_test = catss_test["scz"].copy()
    return x_train, y_train, x_test, y_test



'''
A function to train the LGBMRegressor model.

Parameters of the model:
Tree depth (max_depth): Controls the maximum depth of individual decision trees and should be tuned for best performance. 
Deeper trees can capture more complex relationships but are also prone to overfitting. 

Learning rate (learning_rate): Determines the contribution of each tree to the overall ensemble.
 A smaller learning rate slows down convergence and reduces the risk of overfitting, while a larger value might lead to faster training at the expense of potential overfitting.

Number of trees (n_estimators): Specifies the total number of trees in the ensemble. 
Increasing this parameter can improve performance but also increases the risk of overfitting.

Evaluation Metrics of the model:
RMSE: Square root of MSE. MSE quantifies the difference between predicted values and actual values represented in the dataset for regression problems. 

MAE: Mean Absolute Error (MAE) is a regression metric measuring the average magnitude of errors between predicted and actual values by calculating the absolute difference between them.

R2 Score: It measures how well a regression model predicts outcomes compared to a simple mean. It measures the variation that is explained by a regression model.

Feature Importance metric: 

SHAP Values: Shapley Additive Explanations - It indicates the contribution of each feature to the prediction, compared to the average prediction.
Here, we are using TreeSHAP that explains the predictions after the LightGBM model is trained as the in-built features of importance gain and split are not as informative with correlated variables in the dataset.

Further exploration:
SHAP clustering represent features that influence predictions in similar ways.
'''

def train(outcome_var_list, tag):
    # Checked that these hyperparameters are the most optimal among n_estimators={200, 250, 300}, learning_rate={0.01, 0.05, 0.1}, max_depth={-1, 5}
    
    # Store results
    all_shap_values = []
    all_scores = []
    shap_values_all = []

    # Training over 5 random seeds and taking the average
    for seed in seeds:
        x_train, y_train, x_test, y_test = catss_train_test_split(outcome_var_list, tag, seed)
        model = LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=seed, max_depth=-1)
        model.fit(x_train, y_train)
        training_accuracy = model.score(x_train,y_train)
        testing_accuracy = model.score(x_test,y_test)
        print(f'Training accuracy for seed {seed} =  {training_accuracy}')
        print(f'Testing accuracy for seed {seed} = {testing_accuracy}')
        
        # creating a SHAP explainer
        shap_explainer = shap.TreeExplainer(model)
        # calcuate SHAP values for test set 
        shap_values = shap_explainer.shap_values(x_test)
        shap_mean = np.abs(shap_values).mean(axis=0)
        all_shap_values.append(shap_mean)
        shap_values_all.append(shap_values)

        print("="*60)
        print(f"SHAP VALUES INFO WITH: {tag} at seed : {seed}")
        print("="*60)
        print(f"Shape: {shap_values.shape}")

        y_pred = model.predict(x_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        all_scores.append({'seed': seed, 'training_acc': training_accuracy, 'testing_acc': testing_accuracy, 'rmse': rmse, 'mae': mae, 'r2': r2})

    # Create beeswarm plot
    #plt.figure(figsize=(10, 8))
    #shap.summary_plot(shap_values, x_test, show=False, max_display=20)
    #plt.title(f'{tag} SHAP Beeswarm Plot', fontsize=14, pad=20)
    #plt.tight_layout()
    #plt.savefig(f'{tag}_shap_beeswarm.png', dpi=300, bbox_inches='tight')
    #plt.close()
    #print(f"Beeswarm plot saved: {tag}_shap_beeswarm_class.png")

    #shap_importance = np.abs(shap_values).mean(axis=0)
    #shap_importance_overall = pd.DataFrame({
    #    "feature": x_test.columns,
    #    "importance": shap_importance
    #}).sort_values('importance', ascending=False)

    #print("Top 20 Features Overall:")
    #print(shap_importance_overall.head(20))

    # Save overall importance
    #shap_importance_overall.to_csv(f'{tag}_shap_importance_overall.csv', index=False)
    #print(f"CSV saved: {tag}_shap_importance_overall.csv")
    scores_df = pd.DataFrame(all_scores)
    regression_evaluation_metrics = pd.DataFrame({
        "Metric": ["training_accuracy", "testing_accuracy", "RMSE", "MAE", "R2 score"],
        "Values": [scores_df['training_acc'].mean(), scores_df['testing_acc'].mean(), scores_df['rmse'].mean(), scores_df['mae'].mean(), scores_df['r2'].mean()]
    })
    print(regression_evaluation_metrics)
    regression_evaluation_metrics.to_csv(f"{tag}_regression_evaluation_metrics.csv", index=False)

    all_shap_values = np.array(all_shap_values) # (n_seeds, n_features)
    print(f" all_shap_values Shape: {all_shap_values.shape}")
    # Calculate feature importance with CI
    #feature_importance_by_seed = np.abs(all_shap_values).mean(axis=1)  # (n_seeds, n_features)

    mean_importance = all_shap_values.mean(axis=0)
    std_importance = all_shap_values.std(axis=0, ddof=1)
    se_importance = std_importance / np.sqrt(n_seeds)

    # 95% CI
    t_value = stats.t.ppf(0.975, df=n_seeds-1)
    ci_lower = mean_importance - t_value * se_importance
    ci_upper = mean_importance + t_value * se_importance

    # Create results DataFrame
    results_df = pd.DataFrame({
        'feature': x_test.columns,
        'mean_importance': mean_importance,
        'std': std_importance,
        'se': se_importance,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': ci_upper - ci_lower,
        'cv_percent': (std_importance / mean_importance * 100)
    }).sort_values('mean_importance', ascending=False)

    print("\\n" + "="*80)
    print(f"TOP 20 FEATURES - AVERAGE OVER {n_seeds} SEEDS WITH 95% CI")
    print("="*80)
    print(results_df.head(20).to_string(index=False))

    # Save
    results_df.to_csv(f'{tag}_shap_averaged_{n_seeds}_seeds.csv', index=False)


    # Compute SHAP Clustering
    #shap_matrix = np.array(mean_shap_values)
    #shap_matrix  = np.nan_to_num(shap_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    #feature_variance = shap_matrix.var(axis=0)
    #valid_features = feature_variance > 0
    #shap_matrix_clean = shap_matrix[:, valid_features]  
    #feature_names = x_test.columns[valid_features]

    #invalid_features = x_test.columns[~valid_features].tolist()
    #invalid_columns = pd.DataFrame({"feature": invalid_features})
    #invalid_columns.to_csv(f"{tag}_invalid_features.csv", index=False)

    #distance_matrix = ssd.pdist(shap_matrix_clean.T, metric="correlation")

    #linkage_methods = ['ward', 'complete', 'average', 'single']
    #fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    #axes = axes.flatten()
    #for idx, method in enumerate(linkage_methods):
        #print(f"\nComputing linkage with method: {method}")
        #clustering = linkage(distance_matrix, method=method)
        #dendrogram(
        #    clustering,
        #    ax = axes[idx],
        #    labels=feature_names,
        #    leaf_rotation=90
        #)
        #axes[idx].set_title(f'Linkage Method: {method.capitalize()}', fontsize=14)
        #axes[idx].set_xlabel('Sample Index', fontsize=12)
        #axes[idx].set_ylabel('Distance', fontsize=12)

        #Extract feature clusters
        #clusters = fcluster(clustering, t=5, criterion="maxclust")
        #cluster_map = pd.DataFrame({
        #    "feature": feature_names,
        #    "cluster": clusters
        #})

        # Print top 5 feature groups
        #print("\nTop 5 Feature Groups:\n")
        #for c in sorted(cluster_map.cluster.unique()):
            #print(f"Cluster {c}:")
            #print(cluster_map[cluster_map.cluster == c]["feature"].tolist())
            #print()
        #cluster_map.to_csv(f"{tag}_shap_feature_clusters_{method}.csv", index=False)


    #plt.title(f"{tag} SHAP Feature Clustering Dendrogram")
    #plt.tight_layout()
    #plt.savefig(f"{tag}_shap_feature_dendrogram_correlated_all_linkage.png", dpi=300)
    #plt.close()

    #ward_clustering = linkage(shap_values.T, method="ward")
    #plt.figure(figsize=(12,6))
    #dendrogram(
    #    ward_clustering,
    #    labels=x_test.columns,
    #    leaf_rotation=90
    #)
    #plt.title(f"{tag} SHAP Feature Clustering Dendrogram using ward linkage")
    #plt.tight_layout()
    #plt.savefig(f"{tag}_shap_feature_dendrogram_ward.png", dpi=300)
    #plt.close()

    # save cluster groups to CSV
    #cluster_map.to_csv(f"{tag}_shap_feature_clusters.csv", index=False)


    

# Positive SCZ symptoms - psychotic and manic 
pos_outcome_var_list = ["spied18", "read_thoughts18",  "Special_messages18", "special_powers18", "under_control_special18", "read_others_mind18", "seen_hallucinations18",
                    "read_thoughts_parents18", "Special_messages_parents18", "spied_parents18", "under_control_special_parents18", "read_others_mind_parents18", "special_powers_parents18", "seen_hallucinations_parents18",
                    "hyper_trouble18", "irritable18", "more_confidence18", "not_tired18", "talking_fast18", "racing_thoughts18", "distracted18", "more_energy18", "unusually_active18", "unusual_social18", "unusual_sex_drive18", "risky_unusual18", "unusual_money_trouble18",
                    "unrealistic_abilities18", "talk_fast18", "sexual_inappropriate18", "hear_voices18", 
                    "hyper_trouble_parents18", "irritable_parents18", "more_confidence_parents18", "not_tired_parents18", "racing_thoughts_parents18", "distracted_parents18", "more_energy_parents18", "unusually_active_parents18", "several_partners_parents18", "unusual_sex_drive_parents18",  "risky_unusual_parents18", "unusual_money_spend_parents18"]
tag = "positive"

#print(sorted(pos_y_train.value_counts()))
#print(sorted(pos_y_test.value_counts()))
train(pos_outcome_var_list, tag)

# Negative SCZ symptoms - depressive
neg_outcome_var_list = ["poor_appetite18", "depressed18", "felt_effort18", "restless18", "unhappy18", "lonely18", "others_unfriendly18", "not_enjoyed_life18", "sad18", "people_dislike_me18", "could_not_get_going18"]
tag = "negative"
#print(sorted(neg_y_train.value_counts()))
#print(sorted(neg_y_test.value_counts()))
train(neg_outcome_var_list, tag)


