#!/usr/bin/env python3
"""
Sex Classification Dataset Creator (Single Subject)
===================================================

This script creates balanced train/validation/test splits from the ABCD dataset
for simple sex classification tasks (Single-turn).

Dataset: ABCD Study
Task: Sex classification (Male vs Female)
Strategy: Balanced stratified sampling (50% male, 50% female per split)

Output:
- train_subjects.csv
- validation_subjects.csv
- test_subjects.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configuration
METADATA_PATH = "ABCD_phenotype_total.csv"  # 메타데이터 경로 확인 필요
OUTPUT_DIR = Path("sex_classification_splits_100subjects")
RANDOM_SEED = 42

# Dataset sizes (Balanced)
# Absolute dataset sizes (Must be even numbers for perfect balance)
TRAIN_SIZE = 100  # 50 Male + 50 Female
VAL_SIZE = 100     # 50 Male + 50 Female
TEST_SIZE = 100    # 50 Male + 50 Female

def load_and_clean_data(path):
    """Load metadata and filter for valid entries."""
    df = pd.read_csv(path)
    
    # Column mapping
    if 'subjectkey' in df.columns:
        df = df.rename(columns={'subjectkey': 'subject_id'})
    
    if 'sex' not in df.columns:
        raise ValueError("Column 'sex' not found in metadata")
        
    # Drop NaNs
    df = df.dropna(subset=['subject_id', 'sex'])
    
    # Normalize sex labels
    df['sex'] = df['sex'].astype(str).map({
        '1': 'male', 'M': 'male', 'male': 'male', 'Male': 'male',
        '2': 'female', 'F': 'female', 'female': 'female', 'Female': 'female'
    })
    
    # Filter only valid labels
    df = df[df['sex'].isin(['male', 'female'])]
    
    print(f"Loaded {len(df)} valid subjects.")
    print(f"Total Sex distribution:\n{df['sex'].value_counts()}")
    
    return df

def sample_balanced(df_male, df_female, size, seed):
    """Helper to sample exact size with 50/50 balance."""
    n_male = size // 2
    n_female = size - n_male  # Handle odd sizes just in case
    
    # Check availability
    if len(df_male) < n_male or len(df_female) < n_female:
        raise ValueError(
            f"Not enough data to sample {size} subjects (Need M:{n_male}, F:{n_female} / "
            f"Have M:{len(df_male)}, F:{len(df_female)})"
        )
    
    sampled_m = df_male.sample(n=n_male, random_state=seed)
    sampled_f = df_female.sample(n=n_female, random_state=seed)
    
    # Combine and shuffle
    combined = pd.concat([sampled_m, sampled_f]).sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Return sampled and remaining
    remaining_m = df_male.drop(sampled_m.index)
    remaining_f = df_female.drop(sampled_f.index)
    
    return combined, remaining_m, remaining_f

def create_balanced_splits(df, train_size, val_size, test_size, seed=42):
    """Create splits by sampling absolute numbers sequentially."""
    
    # 1. Separate by sex
    pool_m = df[df['sex'] == 'male']
    pool_f = df[df['sex'] == 'female']
    
    print(f"\n[Sampling Process]")
    print(f"Initial Pool - Male: {len(pool_m)}, Female: {len(pool_f)}")
    
    # 2. Sample Test Set
    test_df, pool_m, pool_f = sample_balanced(pool_m, pool_f, test_size, seed)
    print(f"sampled Test: {len(test_df)} (M:{len(test_df[test_df['sex']=='male'])}, F:{len(test_df[test_df['sex']=='female'])})")
    
    # 3. Sample Validation Set
    val_df, pool_m, pool_f = sample_balanced(pool_m, pool_f, val_size, seed)
    print(f"sampled Val : {len(val_df)} (M:{len(val_df[val_df['sex']=='male'])}, F:{len(val_df[val_df['sex']=='female'])})")
    
    # 4. Sample Train Set
    train_df, pool_m, pool_f = sample_balanced(pool_m, pool_f, train_size, seed)
    print(f"sampled Train: {len(train_df)} (M:{len(train_df[train_df['sex']=='male'])}, F:{len(train_df[train_df['sex']=='female'])})")
    
    print(f"Remaining Pool - Male: {len(pool_m)}, Female: {len(pool_f)}")
    
    return {
        'train': train_df,
        'validation': val_df,
        'test': test_df
    }

def save_splits(splits, output_dir):
    """Save split dataframes to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, df in splits.items():
        save_path = output_dir / f"{split_name}_subjects.csv"
        df.to_csv(save_path, index=False)
        print(f"Saved {split_name} split to {save_path}")

def main():
    print(f"Processing data from {METADATA_PATH}...")
    
    # Load Data
    try:
        df = load_and_clean_data(METADATA_PATH)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    # Create Splits with absolute numbers
    try:
        splits = create_balanced_splits(
            df,
            train_size=TRAIN_SIZE,
            val_size=VAL_SIZE,
            test_size=TEST_SIZE,
            seed=RANDOM_SEED
        )
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        return
    
    # Save
    save_splits(splits, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("DATASET CREATION COMPLETE (Exact Numbers)")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()