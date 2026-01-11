#!/usr/bin/env python3
"""
Sex-Based sMRI Comparison Dataset Creator
==========================================

This script creates balanced train/validation/test splits from the ABCD dataset
for sex-based comparison learning tasks.

Dataset: ABCD Study (Adolescent Brain Cognitive Development)
Task: Sex classification via structural MRI comparison
Strategy: Balanced stratified sampling (50% male, 50% female per split)

Author: BrainVLM Team
Date: 2025-11-25
Updated: 2025-12-08 - Added split-specific pairing configuration
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configuration
METADATA_PATH = "ABCD_phenotype_total.csv"
OUTPUT_DIR = Path("sex_comparison_splits_100subjects_40samples")
RANDOM_SEED = 42

# Dataset sizes (balanced 50M/50F per split)
TRAIN_SIZE = 100  # 50 males + 50 females
VAL_SIZE = 100    # 50 males + 50 females
TEST_SIZE = 100   # 50 males + 50 females

# Split-specific pairing configuration
# Each split can have independent control over comparison pair counts
SPLIT_CONFIG = {
    'train': {
        'n_same_sex_pairs': 40,    # Number of same-sex comparison pairs per subject
        'n_diff_sex_pairs': 40,    # Number of different-sex comparison pairs per subject
    },
    'validation': {
        'n_same_sex_pairs': 3,
        'n_diff_sex_pairs': 3,
    },
    'test': {
        'n_same_sex_pairs': 5,
        'n_diff_sex_pairs': 5,
    }
}

# Image path template
IMAGE_DIR = "/pscratch/sd/h/heehaw/data/1.ABCD/2.sMRI_freesurfer_256"
IMAGE_TEMPLATE = "{subject_id}.nii.gz"


def validate_split_config(config):
    """
    Validate SPLIT_CONFIG to ensure all required splits and parameters are present.

    Args:
        config: SPLIT_CONFIG dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    required_splits = ['train', 'validation', 'test']
    required_params = ['n_same_sex_pairs', 'n_diff_sex_pairs']

    for split in required_splits:
        if split not in config:
            raise ValueError(f"Missing split configuration for: {split}")

        for param in required_params:
            if param not in config[split]:
                raise ValueError(f"Missing parameter '{param}' in {split} configuration")

            value = config[split][param]
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"Parameter '{param}' in {split} must be a non-negative integer, got: {value}")

    print("✓ SPLIT_CONFIG validation passed")


def load_metadata(path):
    """Load ABCD phenotype metadata."""
    print(f"Loading metadata from: {path}")
    df = pd.read_csv(path)
    print(f"Total subjects: {len(df)}")

    # Extract relevant columns
    df = df[['subjectkey', 'sex']].copy()
    df.columns = ['subject_id', 'sex']

    # Remove any missing values
    df = df.dropna()

    # Sex encoding: 1=male, 2=female
    print(f"\nSex distribution:")
    print(f"  Males (sex=1): {(df['sex'] == 1).sum()}")
    print(f"  Females (sex=2): {(df['sex'] == 2).sum()}")

    return df


def stratified_sample(df, n_per_sex, random_state):
    """
    Sample exactly n_per_sex from each sex category.

    Args:
        df: DataFrame with 'subject_id' and 'sex' columns
        n_per_sex: Number of subjects to sample per sex
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with balanced sample
    """
    males = df[df['sex'] == 1].sample(n=n_per_sex, random_state=random_state)
    females = df[df['sex'] == 2].sample(n=n_per_sex, random_state=random_state)

    # Combine and shuffle
    sampled = pd.concat([males, females], ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return sampled


def create_balanced_splits(df, train_size, val_size, test_size, random_seed):
    """
    Create balanced train/val/test splits with equal sex distribution.

    Args:
        df: Full dataset DataFrame
        train_size: Total training samples (will be split 50/50 by sex)
        val_size: Total validation samples (will be split 50/50 by sex)
        test_size: Total test samples (will be split 50/50 by sex)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with train/val/test DataFrames
    """
    # Each split needs equal males and females
    n_train_per_sex = train_size // 2
    n_val_per_sex = val_size // 2
    n_test_per_sex = test_size // 2

    print(f"\nSampling strategy:")
    print(f"  Train: {n_train_per_sex} males + {n_train_per_sex} females = {train_size} total")
    print(f"  Val:   {n_val_per_sex} males + {n_val_per_sex} females = {val_size} total")
    print(f"  Test:  {n_test_per_sex} males + {n_test_per_sex} females = {test_size} total")

    # Separate by sex
    males = df[df['sex'] == 1].copy()
    females = df[df['sex'] == 2].copy()

    # Sample males
    male_train = males.sample(n=n_train_per_sex, random_state=random_seed)
    males_remaining = males.drop(male_train.index)

    male_val = males_remaining.sample(n=n_val_per_sex, random_state=random_seed)
    males_remaining = males_remaining.drop(male_val.index)

    male_test = males_remaining.sample(n=n_test_per_sex, random_state=random_seed)

    # Sample females
    female_train = females.sample(n=n_train_per_sex, random_state=random_seed)
    females_remaining = females.drop(female_train.index)

    female_val = females_remaining.sample(n=n_val_per_sex, random_state=random_seed)
    females_remaining = females_remaining.drop(female_val.index)

    female_test = females_remaining.sample(n=n_test_per_sex, random_state=random_seed)

    # Combine and shuffle each split
    train_df = pd.concat([male_train, female_train], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    train_df['split'] = 'train'

    val_df = pd.concat([male_val, female_val], ignore_index=True)
    val_df = val_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    val_df['split'] = 'validation'

    test_df = pd.concat([male_test, female_test], ignore_index=True)
    test_df = test_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    test_df['split'] = 'test'

    return {
        'train': train_df,
        'validation': val_df,
        'test': test_df
    }


def verify_balance(splits):
    """Verify sex balance in all splits."""
    print("\n" + "="*60)
    print("SPLIT VERIFICATION")
    print("="*60)

    for split_name, df in splits.items():
        males = (df['sex'] == 1).sum()
        females = (df['sex'] == 2).sum()
        total = len(df)

        print(f"\n{split_name.upper()}:")
        print(f"  Total: {total}")
        print(f"  Males (sex=1): {males} ({males/total*100:.1f}%)")
        print(f"  Females (sex=2): {females} ({females/total*100:.1f}%)")
        print(f"  Balance check: {'✓ PASS' if males == females else '✗ FAIL'}")


def save_splits(splits, output_dir):
    """Save splits to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n" + "="*60)
    print("SAVING SPLITS")
    print("="*60)

    for split_name, df in splits.items():
        output_path = output_dir / f"{split_name}_subjects.csv"
        df.to_csv(output_path, index=False)
        print(f"  {split_name}: {output_path} ({len(df)} subjects)")

    # Also save combined metadata
    combined = pd.concat(splits.values(), ignore_index=True)
    combined_path = output_dir / "all_subjects_metadata.csv"
    combined.to_csv(combined_path, index=False)
    print(f"  combined: {combined_path} ({len(combined)} subjects)")

    return output_dir


def create_pairing_metadata(splits, output_dir, split_config):
    """
    Create pairing metadata for comparison-based learning with iterative selection.

    Each split now has independent control over the number of comparison pairs through
    the split_config dictionary.

    For each subject in each split, we'll create multiple pairs with:
    - n_same_sex_pairs reference subjects of same sex (split-specific, iteratively selected)
    - n_diff_sex_pairs reference subjects of different sex (split-specific, iteratively selected)

    Key constraint: Never select the same subject for comparison (subject_id != reference_id)

    Args:
        splits: Dictionary of train/val/test DataFrames
        output_dir: Output directory path
        split_config: Dictionary with split-specific pairing configuration
                     Format: {'train': {'n_same_sex_pairs': X, 'n_diff_sex_pairs': Y}, ...}
    """
    print(f"\n" + "="*60)
    print("CREATING PAIRING METADATA (SPLIT-SPECIFIC ITERATIVE SELECTION)")
    print("="*60)

    for split_name, df in splits.items():
        # Get split-specific configuration
        n_same_sex_pairs = split_config[split_name]['n_same_sex_pairs']
        n_diff_sex_pairs = split_config[split_name]['n_diff_sex_pairs']

        print(f"\n{split_name.upper()} pairing:")
        print(f"  Same-sex pairs per subject: {n_same_sex_pairs}")
        print(f"  Different-sex pairs per subject: {n_diff_sex_pairs}")

        pairs = []

        # Separate by sex
        males = df[df['sex'] == 1]['subject_id'].tolist()
        females = df[df['sex'] == 2]['subject_id'].tolist()

        for idx, row in df.iterrows():
            subject_id = row['subject_id']
            sex = row['sex']

            # === SAME-SEX COMPARISON PAIRS ===
            # Build pool excluding subject itself
            same_sex_pool = males if sex == 1 else females
            same_sex_pool = [s for s in same_sex_pool if s != subject_id]

            # Iteratively select n_same_sex_pairs (cycling through pool if needed)
            if same_sex_pool:
                # Use cycling to ensure we get diverse selections even with small pools
                same_sex_pool_cycling = (same_sex_pool * ((n_same_sex_pairs // len(same_sex_pool)) + 1))
                np.random.shuffle(same_sex_pool_cycling)

                for pair_idx in range(min(n_same_sex_pairs, len(set(same_sex_pool)))):
                    same_sex_ref = same_sex_pool_cycling[pair_idx]
                    pairs.append({
                        'subject_id': subject_id,
                        'sex': sex,
                        'reference_id': same_sex_ref,
                        'reference_sex': sex,
                        'comparison_type': 'same_sex',
                        'pair_index': pair_idx,
                        'split': split_name
                    })

            # === DIFFERENT-SEX COMPARISON PAIRS ===
            # Build pool of opposite sex subjects
            diff_sex_pool = females if sex == 1 else males

            # Iteratively select n_diff_sex_pairs (cycling through pool if needed)
            if diff_sex_pool:
                # Use cycling to ensure we get diverse selections even with small pools
                diff_sex_pool_cycling = (diff_sex_pool * ((n_diff_sex_pairs // len(diff_sex_pool)) + 1))
                np.random.shuffle(diff_sex_pool_cycling)

                for pair_idx in range(min(n_diff_sex_pairs, len(diff_sex_pool))):
                    diff_sex_ref = diff_sex_pool_cycling[pair_idx]
                    pairs.append({
                        'subject_id': subject_id,
                        'sex': sex,
                        'reference_id': diff_sex_ref,
                        'reference_sex': 2 if sex == 1 else 1,
                        'comparison_type': 'different_sex',
                        'pair_index': pair_idx,
                        'split': split_name
                    })

        # Save pairing metadata
        pairs_df = pd.DataFrame(pairs)
        pairs_path = output_dir / f"{split_name}_pairs.csv"
        pairs_df.to_csv(pairs_path, index=False)
        print(f"  Output: {pairs_path} ({len(pairs_df)} pairs)")

    print("\nPairing strategy:")
    print("  - SPLIT-SPECIFIC CONFIGURATION: Each split has independent control over pair counts")
    print("  - ITERATIVE SELECTION: Cycling through pools for diverse reference selection")
    print("  - CONSTRAINT: Same subject never paired with itself (subject_id != reference_id)")
    print("  - This enables rich within-sex and between-sex comparative learning per split")


def main():
    """Main execution function."""
    print("="*60)
    print("SEX-BASED sMRI COMPARISON DATASET CREATOR")
    print("="*60)

    # Validate split configuration
    validate_split_config(SPLIT_CONFIG)

    # Load metadata
    df = load_metadata(METADATA_PATH)

    # Create balanced splits
    splits = create_balanced_splits(
        df,
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        random_seed=RANDOM_SEED
    )

    # Verify balance
    verify_balance(splits)

    # Save splits
    output_dir = save_splits(splits, OUTPUT_DIR)

    # Create pairing metadata with split-specific configuration
    create_pairing_metadata(
        splits,
        output_dir,
        split_config=SPLIT_CONFIG
    )

    # Final summary
    print("\n" + "="*60)
    print("DATASET CREATION COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"Total subjects sampled: {TRAIN_SIZE + VAL_SIZE + TEST_SIZE}")
    print(f"  Train: {TRAIN_SIZE} (50M/50F)")
    print(f"  Validation: {VAL_SIZE} (50M/50F)")
    print(f"  Test: {TEST_SIZE} (50M/50F)")
    print(f"\nComparison pairs per subject (SPLIT-SPECIFIC):")
    for split_name in ['train', 'validation', 'test']:
        n_same = SPLIT_CONFIG[split_name]['n_same_sex_pairs']
        n_diff = SPLIT_CONFIG[split_name]['n_diff_sex_pairs']
        print(f"  {split_name.upper()}:")
        print(f"    Same-sex pairs: {n_same}")
        print(f"    Different-sex pairs: {n_diff}")
        print(f"    Total pairs per subject: {n_same + n_diff}")
    print(f"\nRandom seed: {RANDOM_SEED}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Image format: {IMAGE_TEMPLATE}")
    print("\nNext step: Run generate_sex_comparison_conversations.py")
    print("="*60)


if __name__ == "__main__":
    main()
