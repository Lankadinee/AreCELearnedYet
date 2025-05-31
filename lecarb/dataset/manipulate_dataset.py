import random
import logging
import pickle
import numpy as np
import math
import pandas as pd
from scipy.stats import truncnorm, truncexpon, genpareto
from typing import Dict, Any, Tuple
from copy import deepcopy

from .dataset import load_table
from ..constants import DATA_ROOT, PKL_PROTO

L = logging.getLogger(__name__)

# Independence data: Random by each column
def get_random_data(dataset: str, version: str, overwrite=False) -> Tuple[pd.DataFrame, str]:
    rand_version = f"{version}_ind"
    random_file = DATA_ROOT / dataset / f"{rand_version}.pkl"
    if not overwrite and random_file.is_file():
        L.info(f"Dataset path exists, using it")
        return pd.read_pickle(random_file), rand_version
    
    df = pd.read_pickle(DATA_ROOT / dataset / f"{version}.pkl")
    for col in df.columns:
        df[col] = df[col].sample(frac=1).reset_index(drop=True)
    pd.to_pickle(df, random_file, protocol=PKL_PROTO)
    return df, rand_version

# Max Spearman correlation data: sort by each column
def get_sorted_data(dataset: str, version: str, overwrite=False) -> Tuple[pd.DataFrame, str]:
    sort_version = f"{version}_cor"
    sorted_file = DATA_ROOT / dataset / f"{sort_version}.pkl"
    if not overwrite and sorted_file.is_file():
        return pd.read_pickle(sorted_file), sort_version
    
    df = pd.read_pickle(DATA_ROOT / dataset / f"{version}.pkl")
    for col in df.columns:
        df[col] = df[col].sort_values().reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True)
    pd.to_pickle(df, sorted_file, protocol=PKL_PROTO)
    return df, sort_version

# Get skew data by tuple level frequent rank.
def get_skew_data(dataset: str = 'census', version: str = 'original', sample_ratio=0.0005, overwrite=False) -> Tuple[pd.DataFrame, str]:
    skew_version = f"{version}_skew"
    skew_file = DATA_ROOT / dataset / f"{skew_version}.pkl"
    if not overwrite and skew_file.is_file():
        return pd.read_pickle(skew_file), skew_version
    
    df = pd.read_pickle(DATA_ROOT / dataset / f"{version}.pkl")


    rank_df = pd.DataFrame(0.0, index=range(len(df)), columns=['rank_sum']).astype(np.float32)
    for col in df.columns:
        rank_df['rank_sum'] += df[col].map(df[col].value_counts().div(len(rank_df))).astype(np.float32)
        print(f"{col} frequency calculation finished!")
    selected_id = rank_df.sort_values(by='rank_sum').head(round(len(df)*sample_ratio)).index
    sk_df = df.iloc[selected_id]
    sk_df = pd.concat([sk_df] * int(1/sample_ratio + 1), ignore_index=True).head(len(df))
    pd.to_pickle(sk_df, skew_file, protocol=PKL_PROTO)
    return sk_df, skew_version



def append_data(dataset: str, version_target: str, version_from: str, interval=0.2):
    # Load the target dataset (the base dataset to append to)
    df_target = pd.read_pickle(DATA_ROOT / dataset / f"{version_target}.pkl")
    
    # Load the source dataset (the dataset to append from)
    df_from = pd.read_pickle(DATA_ROOT / dataset / f"{version_from}.pkl")

    # Get the total number of rows in the source dataset
    row_num = len(df_from)
    
    # Define the range of data to append as a fraction of the source dataset
    l = 0  # Start from the beginning (0%)
    r = l + interval  # End at the specified interval (e.g., 0.2 = 20%)
    
    # Check if the interval is valid (not exceeding 100% of the data)
    if r <= 1:
        L.info(f"Start appending {version_target} with {version_from} in [{l}, {r}]")
        
        # Append a slice of the source data to the target data
        # The slice is from l*row_num to r*row_num (e.g., 0% to 20% of source data)
        df_target = df_target.append(df_from[int(l*row_num): int(r*row_num)], ignore_index=True, sort=False)
        
        # Save the combined dataset as a pickle file
        pd.to_pickle(df_target, DATA_ROOT / dataset / f"{version_target}+{version_from}_{r:.1f}.pkl")
        
        # Save the combined dataset as a CSV file for external tools
        df_target.to_csv(DATA_ROOT / dataset / f"{version_target}+{version_from}_{r:.1f}.csv", index=False)
        
        # Load the combined dataset into the database table
        load_table(dataset, f"{version_target}+{version_from}_{r:.1f}")
    else:
        # Log an error if the batch size exceeds the available data
        L.info(f"Appending Fail! Batch size is too big!")



def gen_appended_dataset(
    seed: int, dataset: str, version: str, 
    params: Dict[str, Any], overwrite: bool
    ) -> None:
    # Set random seeds for reproducible data generation
    random.seed(seed)
    np.random.seed(seed)
    
    # Extract parameters for data update configuration
    update_type = params.get('type')  # Type of update: 'ind', 'cor', or 'skew'
    batch_ratio = params.get('batch_ratio')  # Fraction of data to append (0.0 to 1.0)
    L.info(f"Start generating appended data for {dataset}/{version}")

    if update_type == 'ind':
        # Independent update: append randomly shuffled data to simulate independent insertions
        _, rand_version = get_random_data(dataset, version, overwrite=overwrite)
        append_data(dataset, version, rand_version, interval=batch_ratio)
    elif update_type == 'cor':
        # Correlated update: append sorted data to simulate correlated insertions
        _, sort_version = get_sorted_data(dataset, version, overwrite=overwrite)
        append_data(dataset, version, sort_version, interval=batch_ratio)
    elif update_type == 'skew':
        # Skewed update: append data with high-frequency tuples to simulate skewed insertions
        _, skew_version = get_skew_data(dataset, version,
                                        sample_ratio=float(params['skew_size']), overwrite=overwrite)
        append_data(dataset, version, skew_version, interval=batch_ratio)
    else:
        raise NotImplementedError
    L.info("Finish updating data!")


