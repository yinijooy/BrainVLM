# umbrella_dataset_fixed.py
"""
UMBRELLA Dataset with Multi-Modal Sequential Multi-Subject Support.

This dataset handles:
- Multi-turn conversations with LLaVA-Next format
- Sequential multi-subject brain scans (fMRI/dMRI)
- Multiple modalities (text-only, single-scan, multi-scan)
- Training and evaluation modes
- BrainVLM-style multimodal: T1, FA, T1_FA (late fusion)

Key Features:
- Proper conversation tokenization with role-based masking
- Sequential brain scan loading with metadata tracking
- Flexible modality handling (text, single scan, multi-scan)
- Support for various task types (VQA, comparison, classification, etc.)
- T1_FA late fusion: Two separate images interleaved in batch

Format:
    Input: User question + optional brain scans
    Output: Assistant answer

Example Conversation:
    User: "What abnormalities do you see?"
    Assistant: "I observe increased activity in the frontal lobe."
    User: "Compare with scan 2."
    Assistant: "Scan 2 shows reduced activity in the same region."

T1_FA JSONL Example:
    {
        "id": "sub-001_T1_FA",
        "images": [
            {"path": "path/to/T1.nii.gz", "modality": "T1"},
            {"path": "path/to/FA.nii.gz", "modality": "FA"}
        ],
        "conversations": [...]
    }
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import nibabel as nib

from monai.transforms import (
    LoadImage,
    Compose,
    Resize,
    NormalizeIntensity,
    RandAxisFlip,
    EnsureChannelFirst,  # Use instead of AddChannel for compatibility (adds channel if missing)
    ToTensor
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    role: str  # 'user' or 'assistant'
    content: Union[str, List[Dict[str, Any]]]  # Can be string OR list of content items
    has_images: bool = False
    num_images: int = 0


@dataclass
class UMBRELLASample:
    """
    Single sample in UMBRELLA dataset.

    Attributes:
        conversation: List of conversation turns
        brain_scans: List of brain scan file paths (sequential multi-subject)
        task_type: Type of task (e.g., 'vqa', 'comparison', 'classification')
        task_id: Unique identifier for the task
        modality: 'text', 'single_scan', 'multi_scan'
        metadata: Additional metadata (subject IDs, scan types, etc.)
    """
    conversation: List[ConversationTurn]
    brain_scans: List[str] = field(default_factory=list)
    task_type: str = 'vqa'
    task_id: str = ''
    modality: str = 'text'
    metadata: Optional[Dict[str, Any]] = None


class UMBRELLADataset(Dataset):
    """
    UMBRELLA Dataset for multi-modal brain imaging tasks.

    Supports:
    - Multi-turn conversations
    - Sequential multi-subject brain scans
    - Multiple modalities (text, single scan, multi scan)
    - Training and evaluation modes

    Args:
        data_path: Path to dataset JSON file
        tokenizer: HuggingFace tokenizer
        mode: 'train' or 'eval'
        max_seq_length: Maximum sequence length
        img_size: Image size for brain scans (H, W, D)
        max_images_per_sample: Maximum number of brain scans per sample
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        mode: str = 'train',
        max_seq_length: int = 2048,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        max_images_per_sample: int = 8,
        modality_type: str = 'sMRI',  # 'sMRI', 'fMRI', 'T1', 'FA', 'T1_FA'
        **kwargs
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_seq_length = max_seq_length
        self.img_size = img_size
        self.max_images_per_sample = max_images_per_sample
        self.modality_type = modality_type

        # Setting reader=None lets MONAI auto-select reader based on file format (nii.gz, etc.)
        self.image_loader = LoadImage(reader=None, image_only=True, dtype=np.float32)

        # [Modified] Define Transform pipeline
        self.image_transform = self._define_image_augmentation(mode, img_size)

        # Load dataset
        self.samples = self._load_dataset()

        # Check if 4D data (based on img_size tuple length)
        self.is_4d = (len(img_size) == 4)

        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")
        logger.info(f"Mode: {mode}")
        logger.info(f"Modality type: {modality_type}")
        logger.info(f"Max sequence length: {max_seq_length}")
        logger.info(f"Image size: {img_size}")
        logger.info(f"4D images: {self.is_4d}")

    def _load_dataset(self) -> List[UMBRELLASample]:
        import json

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

        data = []
        with open(self.data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line: {line[:50]}...")
                        continue

        samples = []
        for item in data:
            # Conversation parsing
            conversation = [
                ConversationTurn(
                    role=turn['role'],
                    content=turn['content'],
                    has_images=any(c.get('type') == 'image' for c in turn['content']) if isinstance(turn['content'], list) else False,
                    num_images=sum(1 for c in turn['content'] if c.get('type') == 'image') if isinstance(turn['content'], list) else 0
                )
                for turn in item.get('conversations', [])  # Use 'conversations' key
            ]

            # --- Extract image paths from 'images' key ---
            # Note: The unified JSONL generator (generate_conversations.py) already creates
            # correct images and prompts per modality, so we simply load all images here.
            brain_scans = []
            if 'images' in item:
                brain_scans = [img['path'] for img in item['images'] if 'path' in img]
            elif 'brain_scans' in item:
                # Backwards compatibility
                brain_scans = item['brain_scans']
            # ---------------------------------------------------

            sample = UMBRELLASample(
                conversation=conversation,
                brain_scans=brain_scans,
                task_type=item.get('task_type', 'vqa'),
                task_id=item.get('task_id', ''),
                modality="multi_scan" if len(brain_scans) > 1 else "single_scan" if brain_scans else "text",  # Auto-infer modality
                metadata=item.get('metadata')
            )

            samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.

        Returns:
            Dictionary with keys:
            - input_ids: Tokenized input
            - attention_mask: Attention mask
            - labels: Labels for loss computation
            - pixel_values: Brain scan images (if applicable)
            - modality_type: Modality type for patch embedding ('sMRI', 'fMRI', 'T1', 'FA', 'T1_FA')
            - task_type: Task type
            - task_id: Task ID
            - modality: Modality type (text/single_scan/multi_scan)
            - metadata: Additional metadata
        """
        sample = self.samples[idx]

        # Tokenize conversation
        tokenized = self._tokenize_conversation(sample)

        # Load brain scans (if applicable)
        pixel_values = None
        if sample.brain_scans:
            pixel_values = self._load_brain_scans(sample.brain_scans)

        # Return batch item
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['labels'],
            'pixel_values': pixel_values,
            'modality_type': self.modality_type,  # For patch embedding dispatch
            'task_type': sample.task_type,
            'task_id': sample.task_id,
            'modality': sample.modality,
            'metadata': sample.metadata,
            'sample_index': idx
        }

    def _define_image_augmentation(self, mode: str, img_size: Tuple[int, ...]) -> Compose:
        """
        Define MONAI transform pipeline based on mode (train/eval).
        Logic adopted from BaseDataset_T1.
        """
        transforms = []
        
        # 1. Add channel dimension (H, W, D) -> (C, H, W, D)
        # channel_dim='no_channel' adds channel dim at front when input has no channel (3D volume, etc.)
        # Replaces deprecated AddChannel
        transforms.append(EnsureChannelFirst(channel_dim='no_channel'))

        # 2. Resizing (match image size)
        # spatial_size should be spatial dimensions excluding channel
        transforms.append(Resize(spatial_size=img_size))

        if mode == 'train':
            # 3. [Train Only] Data augmentation: Random axis flip
            transforms.append(RandAxisFlip(prob=0.5))

        # 4. Intensity normalization
        transforms.append(NormalizeIntensity())
        
        # 5. Tensor conversion (optional in MONAI 0.9+, but explicit here)
        transforms.append(ToTensor())

        return Compose(transforms)

    def _load_brain_scans(self, scan_paths: List[str]) -> torch.Tensor:
        """
        Load and transform sequential multi-subject brain scans using MONAI.

        Args:
            scan_paths: List of paths to brain scan files

        Returns:
            Tensor of shape (num_scans, C, H, W, D) containing transformed brain scans
        """
        scans = []

        # Load only up to max allowed count
        target_paths = scan_paths[:self.max_images_per_sample]

        for scan_path in target_paths:
            try:
                # 1. Load with MONAI LoadImage (returns Numpy array)
                data = self.image_loader(scan_path)

                # 2. Apply Transform pipeline
                # Composed transform is a callable object
                if self.image_transform:
                    data = self.image_transform(data)
                
                # data shape expected: (C, H, W, D)
                # Convert to Tensor if transform result is not Tensor
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data)

                scans.append(data)

            except Exception as e:
                logger.warning(f"Failed to load brain scan {scan_path}: {e}")
                # On load failure, add zero-filled tensor (maintain shape: C, H, W, D)
                # Assuming EnsureChannelFirst was applied: (1, *img_size)
                fallback_shape = (1, *self.img_size) 
                scans.append(torch.zeros(fallback_shape))

        if not scans:
            # If scan paths are empty or all failed
            return torch.zeros((1, 1, *self.img_size))

        # 3. Stacking
        # List[ (C,H,W,D) ] -> Tensor (N, C, H, W, D)
        return torch.stack(scans, dim=0)

    def _resize_3d(self, data: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """
        Resize 3D brain scan to target size.

        Args:
            data: 3D numpy array
            target_size: Target size (H, W, D)

        Returns:
            Resized 3D array
        """
        from scipy.ndimage import zoom

        # Calculate zoom factors
        zoom_factors = [
            target_size[i] / data.shape[i]
            for i in range(3)
        ]

        # Resize
        resized = zoom(data, zoom_factors, order=1)

        return resized

    def _get_zero_image(self) -> torch.Tensor:
        """Return zero tensor for missing images."""
        if len(self.img_size) == 3:
            # 3D brain scan
            return torch.zeros(1, *self.img_size)
        else:
            # 2D image
            return torch.zeros(1, *self.img_size)

    def _extract_text_from_content(self, content: Union[str, List[Dict[str, Any]]]) -> str:
        """
        Extract ONLY text content from turn content.
        Image placeholders are handled separately in _tokenize_conversation to ensure consistent positioning.
        """
        if isinstance(content, str):
            return content

        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get('type', 'text')
                    if item_type == 'text':
                        text_parts.append(item.get('text', ''))
                    # [Modified] Don't add anything for 'type': 'image' (added separately in front later)
            return ''.join(text_parts)

        else:
            logger.warning(f"Unexpected content type: {type(content)}")
            return str(content)

    def _tokenize_conversation(self, sample: UMBRELLASample) -> Dict[str, torch.Tensor]:
        """
        Tokenize multi-turn conversation with LLaVA-Next format.

        LLaVA-Next Format:
        <|im_start|>user <image><image>
        Compare these scans.<|im_end|><|im_start|>assistant
        Based on comparison...<|im_end|><|im_start|>user
        What about...?<|im_end|><|im_start|>assistant

        Masking Strategy:
        - User turns: Masked with -100 (not in loss)
        - Assistant turns: Active in loss computation
        - Image tokens: Placeholder during tokenization

        Evaluation Mode:
        - Last assistant turn should NOT have <|im_end|> token
        - This signals to model to continue generating from that point
        - Format: "<|im_start|>assistant " (with trailing space, no end token)

        Returns:
            Dict with input_ids, attention_mask, labels
        """
        # Build full conversation in LLaVA-Next format
        conversation_parts = []
        turn_boundaries = []  # Track (start_idx, end_idx, role) for masking

        # Process each turn
        for i, turn in enumerate(sample.conversation):
            # [1] Skip if Eval mode & last Assistant turn (prevent answer leakage & duplicate trigger)
            if self.mode == 'eval':
                is_last_turn = (i == len(sample.conversation) - 1)
                is_assistant = (turn.role == 'assistant')
                if is_last_turn and is_assistant:
                    continue
            
            # [Important] Record turn start position (this was missing)
            turn_start = len(conversation_parts)

            # [2] Start Token + Role + Newline
            conversation_parts.append(f"<|im_start|>{turn.role}\n")

            # [3] Add Image Tokens (Pre-pend: before text)
            # If turn has images, add <image> tokens right after Role
            if turn.num_images > 0:
                # e.g.: if 2 images, "<image><image>"
                conversation_parts.append("<image>" * turn.num_images)

            # [4] Content (pure text only)
            text_content = self._extract_text_from_content(turn.content)
            conversation_parts.append(text_content)

            # [5] End Token + Newline
            conversation_parts.append("<|im_end|>\n")

            turn_end = len(conversation_parts)
            turn_boundaries.append((turn_start, turn_end, turn.role))


        # Debug: Verify all items are strings before joining
        for idx, part in enumerate(conversation_parts):
            if not isinstance(part, str):
                logger.error(f"ERROR: conversation_parts[{idx}] is {type(part)}, not str!")
                logger.error(f"Content: {part}")
                raise TypeError(f"conversation_parts[{idx}] must be str, got {type(part)}")

        # Join all parts into full text
        full_text = ''.join(conversation_parts)

        # Tokenize full text
        encoding = self.tokenizer(
            full_text,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Initialize labels
        labels = input_ids.clone()

        # Mask padding tokens
        labels[attention_mask == 0] = -100

        # Mask user turns (only assistant turns contribute to loss)
        if 'offset_mapping' in encoding:
            offset_mapping = encoding['offset_mapping'].squeeze(0)

            # Convert character-level turn boundaries to token-level
            for part_start, part_end, role in turn_boundaries:
                # Calculate character positions
                char_start = len(''.join(conversation_parts[:part_start]))
                char_end = len(''.join(conversation_parts[:part_end]))

                # Mask tokens in this character range if role is 'user'
                if role == 'user':
                    for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                        if token_start >= char_start and token_end <= char_end:
                            labels[token_idx] = -100
                        elif token_start < char_end and token_end > char_start:
                            # Partial overlap - mask conservatively
                            labels[token_idx] = -100
        else:
            # Fallback: mask based on <|im_start|>assistant pattern
            self._mask_user_turns_fallback(input_ids, labels)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def _mask_user_turns_fallback(self, input_ids: torch.Tensor, labels: torch.Tensor):
        """
        Fallback method to mask user turns when offset_mapping is not available.

        Masks all tokens until we see <|im_start|>assistant pattern.

        Args:
            input_ids: Input token IDs
            labels: Labels tensor to modify in-place
        """
        # Encode the assistant start pattern
        assistant_start = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

        # Find positions where assistant turns start
        seq_len = input_ids.size(0)
        pattern_len = len(assistant_start)

        in_assistant_turn = False

        for i in range(seq_len - pattern_len + 1):
            # Check if this position matches the pattern
            if torch.equal(input_ids[i:i+pattern_len], torch.tensor(assistant_start)):
                in_assistant_turn = True
                # Mask the pattern itself (the turn marker)
                labels[i:i+pattern_len] = -100
                continue

            # Check for turn end
            im_end = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
            if i + len(im_end) <= seq_len:
                if torch.equal(input_ids[i:i+len(im_end)], torch.tensor(im_end)):
                    in_assistant_turn = False
                    labels[i:i+len(im_end)] = -100

            # Mask if not in assistant turn
            if not in_assistant_turn:
                labels[i] = -100

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate batch of samples.

        Args:
            batch: List of sample dictionaries

        Returns:
            Batched dictionary
        """
        # Stack tensors
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])

        # Handle pixel values (optional)
        pixel_values = None
        if batch[0]['pixel_values'] is not None:
            # Stack brain scans - handle variable number of scans per sample
            max_scans = max(item['pixel_values'].size(0) for item in batch)
            batch_size = len(batch)

            # Create padded tensor
            pixel_values = torch.zeros(
                batch_size,
                max_scans,
                *self.img_size
            )

            for i, item in enumerate(batch):
                num_scans = item['pixel_values'].size(0)
                pixel_values[i, :num_scans] = item['pixel_values']

        # Collect metadata
        task_types = [item['task_type'] for item in batch]
        task_ids = [item['task_id'] for item in batch]
        modalities = [item['modality'] for item in batch]
        metadata_list = [item['metadata'] for item in batch]
        sample_indices = [item['sample_index'] for item in batch]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pixel_values': pixel_values,
            'task_types': task_types,
            'task_ids': task_ids,
            'modalities': modalities,
            'metadata': metadata_list,
            'sample_indices': sample_indices
        }


def create_umbrella_dataloader(
    data_path: str,
    tokenizer: Any,
    mode: str = 'train',
    batch_size: int = 4,
    num_workers: int = 4,
    **dataset_kwargs
):
    """
    Create UMBRELLA dataloader.

    Args:
        data_path: Path to dataset JSON file
        tokenizer: HuggingFace tokenizer
        mode: 'train' or 'eval'
        batch_size: Batch size
        num_workers: Number of dataloader workers
        **dataset_kwargs: Additional arguments for UMBRELLADataset

    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader

    dataset = UMBRELLADataset(
        data_path=data_path,
        tokenizer=tokenizer,
        mode=mode,
        **dataset_kwargs
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )

    return dataloader


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # Create dataset
    dataset = UMBRELLADataset(
        data_path="data/umbrella_train.json",
        tokenizer=tokenizer,
        mode='train',
        max_seq_length=2048,
        img_size=(128, 128, 128),
        max_images_per_sample=8
    )

    # Test single sample
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Input IDs shape:", sample['input_ids'].shape)
    print("Attention mask shape:", sample['attention_mask'].shape)
    print("Labels shape:", sample['labels'].shape)

    if sample['pixel_values'] is not None:
        print("Pixel values shape:", sample['pixel_values'].shape)

    # Create dataloader
    dataloader = create_umbrella_dataloader(
        data_path="data/umbrella_train.json",
        tokenizer=tokenizer,
        mode='train',
        batch_size=4,
        num_workers=4
    )

    # Test batch
    batch = next(iter(dataloader))
    print("\nBatch keys:", batch.keys())
    print("Batch input IDs shape:", batch['input_ids'].shape)
    print("Batch attention mask shape:", batch['attention_mask'].shape)
    print("Batch labels shape:", batch['labels'].shape)

    if batch['pixel_values'] is not None:
        print("Batch pixel values shape:", batch['pixel_values'].shape)
