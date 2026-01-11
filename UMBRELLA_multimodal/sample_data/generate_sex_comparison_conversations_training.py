#!/usr/bin/env python3
"""
Sex-Based sMRI Comparison Conversation Generator 
====================================================

Generates multi-turn conversation JSON files for sex-based comparison learning.
Uses LLaVA-NeXT format with generic <image> tokens (NOT modality-specific).

KEY CHANGES IN :
- Generic <image> tokens (not <image_sMRI>)
- Explicit task_id, task_type, subject_ids, modalities fields
- images array with path/token/modality metadata
- Preserves modality information for data loading
- Compatible with LLaVA image token processing

Conversation Pattern:
1. User shows reference image + analysis prompt
2. Assistant acknowledges reference
3. User shows comparison image + comparison task
4. Assistant performs comparison + sex classification

Author: BrainVLM Team
Date: 2025-11-25
Version: 2.0 (LLaVA-compatible)
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Configuration
SPLITS_DIR = Path("sex_comparison_splits_100subjects_20samples")
OUTPUT_DIR = Path("sex_comparison_conversations_mixed_100subjects_20samples")
IMAGE_DIR = "/pscratch/sd/h/heehaw/data/1.ABCD/2.sMRI_freesurfer_256"
IMAGE_TEMPLATE = "{subject_id}.nii.gz"

# LLaVA-NeXT format requires lowercase roles
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# Task configuration
TASK_TYPE = "T3"  # Task 3: Multi-turn comparison
MODALITY = "sMRI"  # Structural MRI

# Prompt templates for comparison-based learning
PROMPTS = {
    #'reference_male': [
    #    "This is a T1-weighted structural MRI scan from a male subject (reference). Please analyze this brain scan carefully.",
    #    "Here is a T1-weighted brain MRI from a male participant. This will serve as your reference scan.",
    #    "Examine this structural brain MRI from a male subject. Pay attention to key anatomical features.",
    #],
    #'reference_female': [
    #    "This is a T1-weighted structural MRI scan from a female subject (reference). Please analyze this brain scan carefully.",
    #    "Here is a T1-weighted brain MRI from a female participant. This will serve as your reference scan.",
    #    "Examine this structural brain MRI from a female subject. Pay attention to key anatomical features.",
    #],

    #'acknowledgment_male': [
    #    "Understood. I've analyzed the reference male brain scan.",
    #],
    #'acknowledgment_female': [
    #    "Understood. I've analyzed the reference female brain scan.",
    #],

    #'comparison_task': [
    #    "Analyze this scan relative to the reference. Classify the biological sex and explain key differences.",
    #    "Compare this brain scan with the reference. What is the likely biological sex of this subject?",
    #    "Now analyze this T1-weighted MRI by comparing it with the reference scan. Estimate the biological sex of this subject.",
    #],

    'reference_male': [
        "Analyze this T1-weighted brain MRI scan (reference). Estimate the biological sex of the subject.",
    ],
    'reference_female': [
        "Analyze this T1-weighted brain MRI scan (reference). Estimate the biological sex of the subject.",
    ],

    'acknowledgment_male': [
        "male.",
    ],
    'acknowledgment_female': [
        "female.",
    ],

    'comparison_task': [
        "Now analyze this T1-weighted MRI by comparing it with the reference scan. Estimate the biological sex of this subject.",
    ],

    'response_male_correct': [
        "male.",
    ],
    'response_female_correct': [
        "female.",
    ],
    'response_male_contrast': [
        "male.",
    ],
    'response_female_contrast': [
        "female.",
    ],

    #'response_male_correct': [
    #    "This scan likely belongs to a male subject. Key features align with the reference male scan in terms of overall brain volume and regional proportions.",
    #    "Classification: Male. The structural features show consistency with male neuroanatomy, including similar hippocampal volumes and cortical thickness patterns.",
    #    "Based on comparison with the reference scan, this appears to be a male subject. Structural similarities include comparable gray matter volumes and white matter distribution patterns.",
    #],
    #'response_female_correct': [
    #    "This scan likely belongs to a female subject. Key features align with the reference female scan in terms of overall brain volume and regional proportions.",
    #    "Classification: Female. The structural features show consistency with female neuroanatomy, including similar hippocampal volumes and cortical thickness patterns.",
    #    "Based on comparison with the reference scan, this appears to be a female subject. Structural similarities include comparable gray matter volumes and white matter distribution patterns.",
    #],
    #'response_male_contrast': [
    #    "This scan likely belongs to a male subject. Unlike the reference female scan, this shows structural characteristics typical of male neuroanatomy.",
    #    "Classification: Male. Compared to the female reference, this scan shows distinct features including different regional volume proportions and cortical thickness patterns.",
    #    "Based on comparison with the female reference scan, this appears to be a male subject. Key differences include relatively larger overall brain volume and different white matter distribution patterns.",
    #],
    #'response_female_contrast': [
    #    "This scan likely belongs to a female subject. Unlike the reference male scan, this shows structural characteristics typical of female neuroanatomy.",
    #    "Classification: Female. Compared to the male reference, this scan shows distinct features including different regional volume proportions and cortical thickness patterns.",
    #    "Based on comparison with the male reference scan, this appears to be a female subject. Key differences include relatively different overall brain volume and distinct white matter distribution patterns.",
    #],
    
}


def get_image_path(subject_id: str) -> str:
    """Get image path for a subject."""
    return f"{IMAGE_DIR}/{IMAGE_TEMPLATE.format(subject_id=subject_id)}"


def validate_image_path(image_path: str) -> bool:
    """
    Validate that an image path exists (or at least is well-formed).

    Note: We don't check actual file existence since this may run
    on a different machine than the training environment.
    """
    return len(image_path) > 0 and image_path.endswith('.nii.gz')


def create_conversation_same_sex(subject_id: str, subject_sex: int,
                                   reference_id: str, reference_sex: int) -> Dict:
    """
    Create conversation for same-sex comparison.

    Args:
        subject_id: Target subject ID
        subject_sex: Target subject sex (1=male, 2=female)
        reference_id: Reference subject ID
        reference_sex: Reference subject sex

    Returns:
        Conversation dictionary in LLaVA-compatible format 
    """
    # Select prompts based on sex
    is_male = (subject_sex == 1)
    sex_label = "male" if is_male else "female"

    ref_prompt = np.random.choice(PROMPTS['reference_male' if is_male else 'reference_female'])
    ack_prompt = np.random.choice(PROMPTS['acknowledgment_male' if is_male else 'acknowledgment_female'])
    comp_prompt = np.random.choice(PROMPTS['comparison_task'])
    resp_prompt = np.random.choice(PROMPTS['response_male_correct' if is_male else 'response_female_correct'])

    # Get image paths
    ref_image_path = get_image_path(reference_id)
    subj_image_path = get_image_path(subject_id)

    # Validate paths
    if not validate_image_path(ref_image_path) or not validate_image_path(subj_image_path):
        raise ValueError(f"Invalid image paths for {subject_id}")

    # Build conversation in  format
    conversation = {
        "task_id": f"{subject_id}_same_sex_comparison",
        "task_type": TASK_TYPE,
        "subject_ids": [reference_id, subject_id],
        "modalities": [MODALITY, MODALITY],
        "images": [
            {
                "path": ref_image_path,
                "token": "<image>",  # Generic token, NOT <image_sMRI>
                "modality": MODALITY
            },
            {
                "path": subj_image_path,
                "token": "<image>",  # Generic token
                "modality": MODALITY
            }
        ],
        "conversations": [
            {
                "role": ROLE_USER,
                "content": [
                    {"type": "text", "text": ref_prompt},
                    {"type": "image", "modality": MODALITY, "image_path": ref_image_path}
                ]
            },
            {
                "role": ROLE_ASSISTANT,
                "content": [
                    {"type": "text", "text": ack_prompt}
                ]
            },
            {
                "role": ROLE_USER,
                "content": [
                    {"type": "text", "text": comp_prompt},
                    {"type": "image", "modality": MODALITY, "image_path": subj_image_path}
                ]
            },
            {
                "role": ROLE_ASSISTANT,
                "content": [
                    {"type": "text", "text": resp_prompt}
                ]
            }
        ],
        "metadata": {
            "subject_id": subject_id,
            "subject_label": sex_label,
            "reference_id": reference_id,
            "reference_label": sex_label,
            "comparison_type": "same",
            "task": "sex_classification_via_comparison"
        }
    }

    return conversation


def create_conversation_different_sex(subject_id: str, subject_sex: int,
                                        reference_id: str, reference_sex: int) -> Dict:
    """
    Create conversation for different-sex comparison.

    Args:
        subject_id: Target subject ID
        subject_sex: Target subject sex (1=male, 2=female)
        reference_id: Reference subject ID
        reference_sex: Reference subject sex (opposite of subject)

    Returns:
        Conversation dictionary in LLaVA-compatible format 
    """
    # Select prompts based on sex
    is_male = (subject_sex == 1)
    ref_is_male = (reference_sex == 1)

    subject_label = "male" if is_male else "female"
    reference_label = "male" if ref_is_male else "female"

    ref_prompt = np.random.choice(PROMPTS['reference_male' if ref_is_male else 'reference_female'])
    ack_prompt = np.random.choice(PROMPTS['acknowledgment_male' if ref_is_male else 'acknowledgment_female'])
    comp_prompt = np.random.choice(PROMPTS['comparison_task'])
    resp_prompt = np.random.choice(PROMPTS['response_male_contrast' if is_male else 'response_female_contrast'])

    # Get image paths
    ref_image_path = get_image_path(reference_id)
    subj_image_path = get_image_path(subject_id)

    # Validate paths
    if not validate_image_path(ref_image_path) or not validate_image_path(subj_image_path):
        raise ValueError(f"Invalid image paths for {subject_id}")

    # Build conversation in  format
    conversation = {
        "task_id": f"{subject_id}_different_sex_comparison",
        "task_type": TASK_TYPE,
        "subject_ids": [reference_id, subject_id],
        "modalities": [MODALITY, MODALITY],
        "images": [
            {
                "path": ref_image_path,
                "token": "<image>",  # Generic token
                "modality": MODALITY
            },
            {
                "path": subj_image_path,
                "token": "<image>",  # Generic token
                "modality": MODALITY
            }
        ],
        "conversations": [
            {
                "role": ROLE_USER,
                "content": [
                    {"type": "text", "text": ref_prompt},
                    {"type": "image", "modality": MODALITY, "image_path": ref_image_path}
                ]
            },
            {
                "role": ROLE_ASSISTANT,
                "content": [
                    {"type": "text", "text": ack_prompt}
                ]
            },
            {
                "role": ROLE_USER,
                "content": [
                    {"type": "text", "text": comp_prompt},
                    {"type": "image", "modality": MODALITY, "image_path": subj_image_path}
                ]
            },
            {
                "role": ROLE_ASSISTANT,
                "content": [
                    {"type": "text", "text": resp_prompt}
                ]
            }
        ],
        "metadata": {
            "subject_id": subject_id,
            "subject_label": subject_label,
            "reference_id": reference_id,
            "reference_label": reference_label,
            "comparison_type": "different",
            "task": "sex_classification_via_comparison"
        }
    }

    return conversation


def generate_conversations_for_split(split_name: str, pairs_df: pd.DataFrame) -> List[Dict]:
    """
    Generate all conversations for a given split.

    Args:
        split_name: Name of split (train/validation/test)
        pairs_df: DataFrame with pairing metadata

    Returns:
        List of conversation dictionaries
    """
    conversations = []

    print(f"  Generating conversations for {split_name}...")

    for idx, row in pairs_df.iterrows():
        subject_id = row['subject_id']
        subject_sex = row['sex']
        reference_id = row['reference_id']
        reference_sex = row['reference_sex']
        comparison_type = row['comparison_type']

        try:
            # Create conversation based on comparison type
            if comparison_type == 'same_sex':
                conv = create_conversation_same_sex(subject_id, subject_sex, reference_id, reference_sex)
            else:  # different_sex
                conv = create_conversation_different_sex(subject_id, subject_sex, reference_id, reference_sex)

            conversations.append(conv)
        except Exception as e:
            print(f"    Warning: Failed to create conversation for {subject_id}: {e}")
            continue

    print(f"    Created {len(conversations)} conversations")
    return conversations


def save_conversations(split_name: str, conversations: List[Dict], output_dir: Path):
    """
    Save conversations to JSON files.

    Args:
        split_name: Name of split
        conversations: List of conversation dictionaries
        output_dir: Output directory
    """

    # Also save all conversations as a single JSONL file
    jsonl_path = output_dir / f"{split_name}_conversations.jsonl"
    with open(jsonl_path, 'w') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')

    print(f"    Saved to: {output_dir}")
    print(f"      Individual files: {len(conversations)}")
    print(f"      JSONL file: {jsonl_path.name}")


def create_sample_outputs(output_dir: Path):
    """Create sample output files for documentation."""
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Creating sample outputs for documentation...")

    # Sample 1: Male same-sex comparison
    sample1 = create_conversation_same_sex(
        subject_id="NDARINV007W6H7B",
        subject_sex=1,
        reference_id="NDARINV00BD7VDC",
        reference_sex=1
    )

    # Sample 2: Female same-sex comparison
    sample2 = create_conversation_same_sex(
        subject_id="NDARINV003RTV85",
        subject_sex=2,
        reference_id="NDARINV00NPMHND",
        reference_sex=2
    )

    # Sample 3: Male different-sex comparison
    sample3 = create_conversation_different_sex(
        subject_id="NDARINV007W6H7B",
        subject_sex=1,
        reference_id="NDARINV003RTV85",
        reference_sex=2
    )

    # Sample 4: Female different-sex comparison
    sample4 = create_conversation_different_sex(
        subject_id="NDARINV003RTV85",
        subject_sex=2,
        reference_id="NDARINV007W6H7B",
        reference_sex=1
    )

    # Sample 5: Another male same-sex comparison
    sample5 = create_conversation_same_sex(
        subject_id="NDARINV00CY2MDM",
        subject_sex=1,
        reference_id="NDARINV00HEV6HB",
        reference_sex=1
    )

    # Sample 6: Another female same-sex comparison
    sample6 = create_conversation_same_sex(
        subject_id="NDARINV01GPLNXC",
        subject_sex=2,
        reference_id="NDARINV00NPMHND",
        reference_sex=2
    )

    # Sample 7: Another male different-sex comparison
    sample7 = create_conversation_different_sex(
        subject_id="NDARINV00CY2MDM",
        subject_sex=1,
        reference_id="NDARINV01GPLNXC",
        reference_sex=2
    )

    # Sample 8: Another female different-sex comparison
    sample8 = create_conversation_different_sex(
        subject_id="NDARINV01GPLNXC",
        subject_sex=2,
        reference_id="NDARINV00CY2MDM",
        reference_sex=1
    )

    # Sample 9: Edge case - male same-sex
    sample9 = create_conversation_same_sex(
        subject_id="NDARINV01ABCDEF",
        subject_sex=1,
        reference_id="NDARINV01GHIJKL",
        reference_sex=1
    )

    # Sample 10: Edge case - female different-sex
    sample10 = create_conversation_different_sex(
        subject_id="NDARINV02MNOPQR",
        subject_sex=2,
        reference_id="NDARINV02STUVWX",
        reference_sex=1
    )

    samples = [sample1, sample2, sample3, sample4, sample5,
               sample6, sample7, sample8, sample9, sample10]

    for i, sample in enumerate(samples, 1):
        output_path = sample_dir / f"sample_{i:02d}_{sample['task_id']}.json"
        with open(output_path, 'w') as f:
            json.dump(sample, f, indent=2)

    print(f"    Created {len(samples)} sample output files in: {sample_dir}")


def print_format_example():
    """Print an example of the  format."""
    print("\n" + "="*60)
    print(" FORMAT EXAMPLE")
    print("="*60)
    print("""
{
    "task_id": "NDARINV007W6H7B_same_sex_comparison",
    "task_type": "T3",
    "subject_ids": ["NDARINV00BD7VDC", "NDARINV007W6H7B"],
    "modalities": ["sMRI", "sMRI"],
    "images": [
        {
            "path": "/pscratch/.../NDARINV00BD7VDC.nii.gz",
            "token": "<image>",
            "modality": "sMRI"
        },
        {
            "path": "/pscratch/.../NDARINV007W6H7B.nii.gz",
            "token": "<image>",
            "modality": "sMRI"
        }
    ],
    "conversations": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "..."},
                {"type": "image", "modality": "sMRI", "image_path": "..."}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "..."}]
        }
    ],
    "metadata": {
        "subject_id": "NDARINV007W6H7B",
        "subject_label": "male",
        "reference_id": "NDARINV00BD7VDC",
        "reference_label": "male",
        "comparison_type": "same",
        "task": "sex_classification_via_comparison"
    }
}

KEY CHANGES FROM V1:
- Generic <image> token (NOT <image_sMRI>)
- Added task_id, task_type, subject_ids, modalities fields
- images array with metadata
- conversation content uses "type": "image" (NOT "image_sMRI")
- metadata uses "subject_label" and "reference_label" (NOT sex)
- comparison_type is "same" or "different" (NOT same_sex/different_sex)
    """)
    print("="*60)


def main():
    """Main execution function."""
    print("="*60)
    print("SEX-BASED sMRI COMPARISON CONVERSATION GENERATOR ")
    print("="*60)
    print("\n FEATURES:")
    print("  - Generic <image> tokens (LLaVA-compatible)")
    print("  - Modality metadata preserved for loading")
    print("  - Enhanced task tracking fields")
    print("  - Explicit image array with path/token/modality")
    print("="*60)

    # Check if splits exist
    if not SPLITS_DIR.exists():
        print(f"\nError: Splits directory not found: {SPLITS_DIR}")
        print("Please run create_sex_comparison_dataset.py first.")
        print("\nGenerating sample outputs only...")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        create_sample_outputs(OUTPUT_DIR)
        print_format_example()
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split_name in ['train']:
        print(f"\nProcessing {split_name.upper()} split:")

        # Load pairs
        pairs_path = SPLITS_DIR / f"{split_name}_pairs.csv"
        if not pairs_path.exists():
            print(f"  Warning: Pairs file not found: {pairs_path}")
            continue

        pairs_df = pd.read_csv(pairs_path)

        # Generate conversations
        conversations = generate_conversations_for_split(split_name, pairs_df)

        # Save conversations
        save_conversations(split_name, conversations, OUTPUT_DIR)

    # Create sample outputs
    create_sample_outputs(OUTPUT_DIR)

    # Print format example
    print_format_example()

    # Final summary
    print("\n" + "="*60)
    print("CONVERSATION GENERATION COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print(f"\nDirectory structure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    train/")
    print(f"      <task_id>.json")
    print(f"      train_conversations.jsonl")
    print(f"    validation/")
    print(f"      [same structure]")
    print(f"    test/")
    print(f"      [same structure]")
    print(f"    samples/")
    print(f"      sample_01_*.json")
    print(f"      sample_02_*.json")
    print(f"      ... (10 samples total)")
    print("\nFormat: LLaVA-NeXT  (generic <image> tokens)")
    print("Task: Sex classification via comparison-based learning")
    print("="*60)


if __name__ == "__main__":
    main()
