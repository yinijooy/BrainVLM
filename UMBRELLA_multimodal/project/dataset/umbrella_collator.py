"""
UMBRELLA Data Collator (Fixed)

Key Features:
- Integrates Tokenization and Masking in one pass for accuracy
- Correctly handles variable images per sample
- Supports ChatML format for multi-turn conversations
"""
import logging
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

@dataclass
class UMBRELLABatch:
    """Batch object compatible with HF Trainer."""
    pixel_values: Optional[torch.Tensor]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    # Metadata for Trainer/Logging
    num_images_per_sample: List[int]
    task_types: List[str]
    metadata: List[Dict[str, Any]]

    # Optional fields
    image_mask: Optional[torch.Tensor] = None
    task_ids: Optional[torch.Tensor] = None
    sample_indices: Optional[List[int]] = None
    modality_type: Optional[str] = None  # 'sMRI', 'fMRI', 'T1', 'FA', 'T1_FA'

    def __len__(self):
        return self.input_ids.shape[0]

    def keys(self):
        return [k for k in self.__dict__.keys() if getattr(self, k) is not None]

    def __getitem__(self, key):
        if isinstance(key, int):
            raise IndexError("UMBRELLABatch does not support integer indexing.")
        return getattr(self, key)
    
    def __contains__(self, key):
        return hasattr(self, key) and getattr(self, key) is not None

    def get(self, key, default=None):
        return getattr(self, key, default)

    def pop(self, key, default=None):
        if hasattr(self, key):
            val = getattr(self, key)
            object.__setattr__(self, key, None)
            return val
        return default

    def values(self):
        return [getattr(self, k) for k in self.keys()]

    def items(self):
        return [(k, getattr(self, k)) for k in self.keys()]


class UMBRELLACollator:
    """
    Unified Collator: Handles both pre-tokenized data and raw conversations.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_token = "<image>"
        self.ignore_index = -100
        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"
        self.nl = "\n"

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _process_turn(self, role: str, content: Any) -> Tuple[List[int], List[int]]:
        """Legacy support: Tokenize raw conversation if needed."""
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if item.get('type') == 'text': text_parts.append(item['text'])
                elif item.get('type') == 'image': text_parts.append(self.image_token)
            text_content = "".join(text_parts)
        else:
            text_content = str(content)

        header = f"{self.im_start}{role}{self.nl}"
        footer = f"{self.im_end}{self.nl}"

        header_ids = self.tokenizer.encode(header, add_special_tokens=False)
        content_ids = self.tokenizer.encode(text_content, add_special_tokens=False)
        footer_ids = self.tokenizer.encode(footer, add_special_tokens=False)

        input_ids = header_ids + content_ids + footer_ids

        if role.lower() in ["user", "human", "system"]:
            labels = [self.ignore_index] * len(input_ids)
        else:
            labels = [self.ignore_index] * len(header_ids) + content_ids + footer_ids
            
        return input_ids, labels

    def __call__(self, batch: List[Dict[str, Any]]) -> UMBRELLABatch:
        # Check if data is already tokenized (Dataset returned input_ids)
        is_pre_tokenized = "input_ids" in batch[0]

        input_ids_list = []
        labels_list = []
        pixel_values_list = []
        num_images_list = []
        task_types = []
        metadata_list = []
        sample_indices = []
        modality_type = batch[0].get('modality_type', 'sMRI')  # Get from first item (all same in batch)

        for item in batch:
            # 1. Metadata
            task_types.append(item.get('task_type', 'unknown'))
            metadata_list.append(item.get('metadata', {}))
            sample_indices.append(item.get('sample_index', -1))
            
            # 2. Images
            imgs = item.get('pixel_values')
            if imgs is not None:
                if isinstance(imgs, dict): imgs = list(imgs.values())[0]
                pixel_values_list.append(imgs)
                num_images_list.append(imgs.shape[0])
            else:
                num_images_list.append(0)

            # 3. Input IDs & Labels
            if is_pre_tokenized:
                # [CASE A] Already tokenized by Dataset
                cur_ids = item['input_ids']
                cur_lbl = item['labels']
                
                # Convert to tensor if list
                if not isinstance(cur_ids, torch.Tensor):
                    cur_ids = torch.tensor(cur_ids, dtype=torch.long)
                if not isinstance(cur_lbl, torch.Tensor):
                    cur_lbl = torch.tensor(cur_lbl, dtype=torch.long)
                    
                input_ids_list.append(cur_ids)
                labels_list.append(cur_lbl)
            else:
                # [CASE B] Raw conversations (Need tokenization)
                conversation = item.get('conversations', [])
                if isinstance(conversation, str): conversation = json.loads(conversation)
                
                cur_input_ids = []
                cur_labels = []

                for turn in conversation:
                    ids, lbls = self._process_turn(turn.get('role', 'user'), turn.get('content', ''))
                    cur_input_ids.extend(ids)
                    cur_labels.extend(lbls)
                
                # Truncate
                if len(cur_input_ids) > self.max_length:
                    cur_input_ids = cur_input_ids[:self.max_length]
                    cur_labels = cur_labels[:self.max_length]
                
                input_ids_list.append(torch.tensor(cur_input_ids, dtype=torch.long))
                labels_list.append(torch.tensor(cur_labels, dtype=torch.long))

        # 4. Padding
        input_ids = pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = pad_sequence(
            labels_list, batch_first=True, padding_value=self.ignore_index
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        # 5. Image Batching
        pixel_values = torch.cat(pixel_values_list, dim=0) if pixel_values_list else None

        return UMBRELLABatch(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            num_images_per_sample=num_images_list,
            task_types=task_types,
            metadata=metadata_list,
            sample_indices=sample_indices,
            modality_type=modality_type
        )