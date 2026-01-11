#!/usr/bin/env python3
"""
Unified JSONL Generator for UMBRELLA
=====================================

Generates JSONL files for T1, FA, or T1_FA modality training.
Each modality has its own prompt templates.

Usage:
    # T1 only
    python generate_conversations.py \
        --modality T1 \
        --meta_csv /path/to/metadata.csv \
        --t1_dir /path/to/T1_images \
        --output_dir ./t1_conversations

    # FA only
    python generate_conversations.py \
        --modality FA \
        --meta_csv /path/to/metadata.csv \
        --fa_dir /path/to/FA_images \
        --output_dir ./fa_conversations

    # T1 + FA (multimodal)
    python generate_conversations.py \
        --modality T1_FA \
        --meta_csv /path/to/metadata.csv \
        --t1_dir /path/to/T1_images \
        --fa_dir /path/to/FA_images \
        --output_dir ./t1_fa_conversations
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Prompt Templates (per modality)
# =============================================================================

PROMPTS = {
    'T1': {
        'sex_classification': {
            'question': "Analyze this T1-weighted brain MRI scan. Determine the biological sex of the subject.",
        },
        'age_prediction': {
            'question': "Analyze this T1-weighted brain MRI scan. Estimate the age of the subject.",
        }
    },
    'FA': {
        'sex_classification': {
            'question': "Analyze this FA (Fractional Anisotropy) map. Determine the biological sex of the subject.",
        },
        'age_prediction': {
            'question': "Analyze this FA map. Estimate the age of the subject.",
        }
    },
    'T1_FA': {
        'sex_classification': {
            'question': "Based on the T1-weighted MRI and FA map, what is the biological sex of this subject?",
        },
        'age_prediction': {
            'question': "Based on the T1-weighted MRI and FA map, estimate the age of this subject.",
        }
    }
}

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Generate JSONL for UMBRELLA training')

    # Required
    parser.add_argument('--modality', type=str, required=True,
                        choices=['T1', 'FA', 'T1_FA'],
                        help='Modality type: T1, FA, or T1_FA')
    parser.add_argument('--meta_csv', type=str, required=True,
                        help='Path to metadata CSV (must have subject_id, sex columns)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for JSONL files')

    # Image directories (required based on modality)
    parser.add_argument('--t1_dir', type=str, default=None,
                        help='Directory containing T1 images (required for T1, T1_FA)')
    parser.add_argument('--fa_dir', type=str, default=None,
                        help='Directory containing FA images (required for FA, T1_FA)')

    # Task
    parser.add_argument('--task', type=str, default='sex_classification',
                        choices=['sex_classification', 'age_prediction'],
                        help='Task type')

    # File options
    parser.add_argument('--t1_suffix', type=str, default='.nii.gz',
                        help='T1 file suffix')
    parser.add_argument('--fa_suffix', type=str, default='.nii.gz',
                        help='FA file suffix')

    # Column names
    parser.add_argument('--subject_col', type=str, default='subject_id',
                        help='Column name for subject ID')
    parser.add_argument('--sex_col', type=str, default='sex',
                        help='Column name for sex (1=male, 2=female)')
    parser.add_argument('--age_col', type=str, default='age',
                        help='Column name for age')

    # Split ratios
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)

    # Options
    parser.add_argument('--check_exists', action='store_true',
                        help='Only include subjects where images exist')

    args = parser.parse_args()

    # Validate directories based on modality
    if args.modality in ['T1', 'T1_FA'] and args.t1_dir is None:
        parser.error(f"--t1_dir is required for modality={args.modality}")
    if args.modality in ['FA', 'T1_FA'] and args.fa_dir is None:
        parser.error(f"--fa_dir is required for modality={args.modality}")

    return args


# =============================================================================
# Utility Functions
# =============================================================================

def normalize_sex(sex_value) -> Optional[str]:
    """Normalize sex value to 'male' or 'female'."""
    if pd.isna(sex_value):
        return None
    if isinstance(sex_value, (int, float)):
        if sex_value == 1:
            return 'male'
        elif sex_value == 2:
            return 'female'
    elif isinstance(sex_value, str):
        sex_lower = sex_value.lower().strip()
        if sex_lower in ['m', 'male', '1']:
            return 'male'
        elif sex_lower in ['f', 'female', '2']:
            return 'female'
    return None


def get_image_path(subject_id: str, img_dir: str, suffix: str) -> str:
    """Get image path for a subject."""
    return str(Path(img_dir) / f"{subject_id}{suffix}")


def check_image_exists(path: str) -> bool:
    """Check if image file exists."""
    return Path(path).exists()


# =============================================================================
# Sample Creation Functions
# =============================================================================

def create_sample(
    subject_id: str,
    label: str,
    modality: str,
    task: str,
    t1_path: Optional[str] = None,
    fa_path: Optional[str] = None
) -> Dict:
    """
    Create a sample for any modality.

    Args:
        subject_id: Subject identifier
        label: Label value (sex or age)
        modality: 'T1', 'FA', or 'T1_FA'
        task: 'sex_classification' or 'age_prediction'
        t1_path: Path to T1 image (for T1, T1_FA)
        fa_path: Path to FA image (for FA, T1_FA)

    Returns:
        Sample dictionary in JSONL format
    """
    # Select prompt based on modality
    question = PROMPTS[modality][task]['question']

    # Build answer
    if task == 'sex_classification':
        answer = f"{label}."
    else:  # age_prediction
        answer = f"{int(round(float(label)))} years old."

    # Build images list based on modality
    if modality == 'T1':
        images = [{"path": t1_path, "modality": "T1"}]
        image_content = [{"type": "image"}]
    elif modality == 'FA':
        images = [{"path": fa_path, "modality": "FA"}]
        image_content = [{"type": "image"}]
    else:  # T1_FA
        images = [
            {"path": t1_path, "modality": "T1"},
            {"path": fa_path, "modality": "FA"}
        ]
        image_content = [{"type": "image"}, {"type": "image"}]

    # Build sample
    return {
        "task_id": f"{subject_id}_{modality}_{task}",
        "task_type": f"{modality}_{task}",
        "subject_ids": [subject_id],
        "modalities": [modality] if modality != 'T1_FA' else ["T1", "FA"],
        "images": images,
        "conversations": [
            {
                "role": ROLE_USER,
                "content": image_content + [{"type": "text", "text": question}]
            },
            {
                "role": ROLE_ASSISTANT,
                "content": [{"type": "text", "text": answer}]
            }
        ],
        "metadata": {
            "subject_id": subject_id,
            "subject_label": label,
            "task": task,
            "modality_type": modality
        }
    }


# =============================================================================
# Main Generation Logic
# =============================================================================

def generate_samples(
    df: pd.DataFrame,
    modality: str,
    task: str,
    t1_dir: Optional[str],
    fa_dir: Optional[str],
    t1_suffix: str,
    fa_suffix: str,
    subject_col: str,
    sex_col: str,
    age_col: str,
    check_exists: bool = False
) -> List[Dict]:
    """Generate samples from dataframe."""

    samples = []
    skipped = 0

    for _, row in df.iterrows():
        subject_id = str(row[subject_col])

        # Get image paths based on modality
        t1_path = get_image_path(subject_id, t1_dir, t1_suffix) if t1_dir else None
        fa_path = get_image_path(subject_id, fa_dir, fa_suffix) if fa_dir else None

        # Check if images exist (optional)
        if check_exists:
            if modality == 'T1' and not check_image_exists(t1_path):
                skipped += 1
                continue
            elif modality == 'FA' and not check_image_exists(fa_path):
                skipped += 1
                continue
            elif modality == 'T1_FA' and (not check_image_exists(t1_path) or not check_image_exists(fa_path)):
                skipped += 1
                continue

        # Get label
        if task == 'sex_classification':
            label = normalize_sex(row.get(sex_col))
            if label is None:
                skipped += 1
                continue
        else:  # age_prediction
            label = row.get(age_col)
            if pd.isna(label):
                skipped += 1
                continue
            label = str(label)

        # Create sample
        sample = create_sample(
            subject_id=subject_id,
            label=label,
            modality=modality,
            task=task,
            t1_path=t1_path,
            fa_path=fa_path
        )
        samples.append(sample)

    if skipped > 0:
        print(f"  Skipped {skipped} subjects (missing data or images)")

    return samples


def save_jsonl(samples: List[Dict], output_path: Path):
    """Save samples to JSONL file."""
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    print(f"  Saved {len(samples)} samples to {output_path}")


def main():
    args = parse_args()

    print("=" * 60)
    print(f"UMBRELLA JSONL GENERATOR - {args.modality}")
    print("=" * 60)

    # Load metadata
    print(f"\nLoading metadata from: {args.meta_csv}")
    df = pd.read_csv(args.meta_csv)
    print(f"  Total subjects: {len(df)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split data
    print(f"\nSplitting data (train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio})")

    stratify_col = df[args.sex_col] if args.task == 'sex_classification' else None

    train_df, temp_df = train_test_split(
        df,
        train_size=args.train_ratio,
        random_state=args.seed,
        stratify=stratify_col
    )

    val_ratio_adjusted = args.val_ratio / (args.val_ratio + args.test_ratio)
    stratify_col_temp = temp_df[args.sex_col] if args.task == 'sex_classification' else None

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio_adjusted,
        random_state=args.seed,
        stratify=stratify_col_temp
    )

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Generate samples for each split
    splits = {'train': train_df, 'val': val_df, 'test': test_df}

    for split_name, split_df in splits.items():
        print(f"\nGenerating {split_name} samples...")
        samples = generate_samples(
            df=split_df,
            modality=args.modality,
            task=args.task,
            t1_dir=args.t1_dir,
            fa_dir=args.fa_dir,
            t1_suffix=args.t1_suffix,
            fa_suffix=args.fa_suffix,
            subject_col=args.subject_col,
            sex_col=args.sex_col,
            age_col=args.age_col,
            check_exists=args.check_exists
        )

        output_path = output_dir / f"{split_name}_conversations.jsonl"
        save_jsonl(samples, output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nModality: {args.modality}")
    print(f"Task: {args.task}")
    print(f"Output directory: {output_dir}")
    print("\nFiles created:")
    print("  - train_conversations.jsonl")
    print("  - val_conversations.jsonl")
    print("  - test_conversations.jsonl")

    # Print sample format
    print("\n" + "=" * 60)
    print("SAMPLE FORMAT")
    print("=" * 60)

    if args.modality == 'T1':
        print('''
{
    "task_id": "sub-001_T1_sex_classification",
    "images": [{"path": "/path/to/T1.nii.gz", "modality": "T1"}],
    "conversations": [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Analyze this T1-weighted brain MRI..."}
            ]
        },
        {"role": "assistant", "content": [{"type": "text", "text": "male."}]}
    ]
}''')
    elif args.modality == 'FA':
        print('''
{
    "task_id": "sub-001_FA_sex_classification",
    "images": [{"path": "/path/to/FA.nii.gz", "modality": "FA"}],
    "conversations": [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Analyze this FA map..."}
            ]
        },
        {"role": "assistant", "content": [{"type": "text", "text": "male."}]}
    ]
}''')
    else:  # T1_FA
        print('''
{
    "task_id": "sub-001_T1_FA_sex_classification",
    "images": [
        {"path": "/path/to/T1.nii.gz", "modality": "T1"},
        {"path": "/path/to/FA.nii.gz", "modality": "FA"}
    ],
    "conversations": [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": "Based on the T1 and FA..."}
            ]
        },
        {"role": "assistant", "content": [{"type": "text", "text": "male."}]}
    ]
}''')


if __name__ == "__main__":
    main()
