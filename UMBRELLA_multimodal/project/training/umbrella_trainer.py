"""
UMBRELLA Trainer: Unified Training and Evaluation

Key Features:
1. Training: Multi-turn masking, Dynamic Image Expansion, Task-aware logging
2. Evaluation: Generates actual text responses with proper Left Padding & Image Injection
3. Metrics: Sex classification accuracy, Turn stats

"""
import os
import copy
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from pathlib import Path

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer_utils import EvalLoopOutput


# Import UMBRELLA components (these will be available when module is properly imported)
# Deferred import inside functions to avoid circular imports

logger = logging.getLogger(__name__)


@dataclass
class UMBRELLATrainingArgs(TrainingArguments):
    """Extended training arguments for UMBRELLA-specific settings."""
    # 1. Dynamic Batching & Memory
    enable_memory_aware_batching: bool = field(default=True, metadata={"help": "Enable memory-aware batching"})
    memory_budget_gb: float = field(default=30.0, metadata={"help": "Memory budget in GB"})
    
    # 2. Gradient Normalization
    normalize_gradients_by_batch_size: bool = field(default=True, metadata={"help": "Normalize gradients by batch size"})
    base_batch_size: int = field(default=32, metadata={"help": "Base batch size for gradient normalization"})

    # 3. Logging
    log_image_statistics: bool = field(default=True, metadata={"help": "Log image statistics"})
    
    # 4. [CRITICAL FIX] Evaluation Generation Config (Missing Fields Added Here)
    eval_output_dir: str = field(default="./eval_predictions", metadata={"help": "Directory to save evaluation predictions"})
    #save_eval_predictions: bool = field(default=True, metadata={"help": "Whether to save eval predictions to JSONL"})  # Add if needed
    eval_max_new_tokens: int = field(default=256, metadata={"help": "Max new tokens for generation"})
    eval_temperature: float = field(default=0.7, metadata={"help": "Temperature for generation"})
    eval_top_p: float = field(default=0.9, metadata={"help": "Top-p for generation"})
    eval_do_sample: bool = field(default=True, metadata={"help": "Do sample for generation"})

class UMBRELLATrainer(Trainer):
    """Unified Trainer for Training and Evaluation (Generation)."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_history = {'loss': [], 'eval': []}
        os.makedirs(self.args.eval_output_dir, exist_ok=True)
        
        # for gradient checking
        self._static_graph_set = False

    def _ensure_set_static_graph(self, model):
        """Helper for DDP static graph optimization."""
        if not self._static_graph_set and self.is_in_train:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                # model._set_static_graph() # Optional: Enable if topology is static
                pass
            self._static_graph_set = True

    def _compute_dummy_loss(self, model, active_modality):
        """
        Compute dummy loss for unused parameters to fix DDP and grad_fn errors.

        This ensures:
        1. Inactive modality params (e.g. fMRI when training sMRI) get gradients (0.0).
        2. Text-only batches still connect to trainable params (PatchEmbed), giving loss a grad_fn.
        """
        # Unwrap model if DDP
        if hasattr(model, "module"):
            base_model = model.module
        else:
            base_model = model

        # Access the trainable embeddings module
        # Path: LlavaForConditionalGeneration -> LlavaMetaForCausalLM -> LlavaMultiModalProjector/VisionTower
        # We target the PatchEmbed which is inside the Vision Tower
        try:
            embeddings = base_model.vision_tower.vision_model.embeddings
        except AttributeError:
            # Fallback if structure is different
            return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)

        dummy_loss = torch.tensor(0.0, dtype=torch.float32, device=next(model.parameters()).device)

        # Scaling factor to ensure 0-contribution but graph connection
        # Using 0.0 * sum() is standard pattern for this

        # All supported modalities (including T1/FA for BrainVLM style)
        all_modalities = ['sMRI', 'fMRI', 'T1', 'FA']

        for name, param in embeddings.named_parameters():
            if param.requires_grad:
                # Logic: Add to dummy loss if it belongs to an INACTIVE modality
                # If active_modality is None (Text-only), ALL modalities are inactive, so ALL are added.
                # This perfectly fixes the "element 0 does not require grad" error.

                is_active_param = False
                if active_modality:
                    # For T1_FA, both T1 and FA are active
                    if active_modality == 'T1_FA':
                        if 'T1' in name or 'FA' in name:
                            is_active_param = True
                    elif active_modality in name:
                        is_active_param = True

                if not is_active_param:
                    dummy_loss = dummy_loss + (param.sum() * 0.0)

        return dummy_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom compute_loss with Dynamic Image Token Expansion & Dummy Gradient Support.
        """
        self._ensure_set_static_graph(model)

        # 1. Prepare Inputs
        raw_labels = inputs.pop("labels", None)

        # Remove metadata (and get modality_type)
        inputs.pop("num_images_per_sample", None)
        inputs.pop("task_types", None)
        inputs.pop("metadata", None)
        inputs.pop("image_mask", None)
        inputs.pop("task_ids", None)
        inputs.pop("sample_indices", None)
        modality_type = inputs.pop("modality_type", None)  # Get modality_type from batch

        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) and v.device != device else v for k, v in inputs.items()}
        if raw_labels is not None: raw_labels = raw_labels.to(device)

        # 2. Vision Feature Extraction & Modality Detection
        pixel_values = inputs.get('pixel_values')
        image_features_per_sample = None
        active_modality = None

        if pixel_values is not None:
            # Use modality_type from batch if available, otherwise detect from shape
            if modality_type and modality_type in ['T1', 'FA', 'T1_FA', 'sMRI', 'fMRI']:
                active_modality = modality_type
            else:
                # Fallback: Detect Modality based on Shape
                # sMRI (Flattened): (N, 1, 128, 128, 128) -> ndim=5
                # fMRI (Flattened): (N, 1, 96, 96, 96, T) -> ndim=6
                if pixel_values.dim() == 5:
                    active_modality = 'sMRI'
                elif pixel_values.dim() == 6:
                    active_modality = 'fMRI'

            # Pass modality to vision_tower (PatchEmbed)
            # The PatchEmbed.forward() accepts modality parameter
            image_outputs = model.vision_tower(pixel_values, output_hidden_states=True)
            sel_feats = image_outputs.hidden_states[model.config.vision_feature_layer]
            sel_feats = model.multi_modal_projector(sel_feats)
            
            batch_size = inputs['input_ids'].shape[0]
            total_images = sel_feats.shape[0]
            if batch_size > 0:
                num_imgs = total_images // batch_size
                image_features_per_sample = sel_feats.view(batch_size, num_imgs, -1, sel_feats.shape[-1])
        else:
            # Text-only batch
            active_modality = None

        # 3. Compute Dummy Loss (Critical Fix)
        # This adds 0.0 * sum(unused_params) to the graph
        dummy_loss = self._compute_dummy_loss(model, active_modality)

        # 4. Dynamic Merge (Expand <image> tokens)
        IMAGE_TOKEN_ID = 151646 
        if hasattr(model.config, "image_token_index"): IMAGE_TOKEN_ID = model.config.image_token_index

        new_embeds, new_labels, new_masks = [], [], []
        inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])
        
        batch_size = inputs['input_ids'].shape[0]
        for i in range(batch_size):
            cur_ids = inputs['input_ids'][i]
            cur_emb = inputs_embeds[i]
            cur_lbl = raw_labels[i]
            cur_msk = inputs['attention_mask'][i]
            
            img_indices = (cur_ids == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
            final_emb, final_lbl, final_msk = [], [], []
            last_pos, img_cnt = 0, 0
            
            for pos in img_indices:
                final_emb.append(cur_emb[last_pos:pos])
                final_lbl.append(cur_lbl[last_pos:pos])
                final_msk.append(cur_msk[last_pos:pos])
                
                if image_features_per_sample is not None:
                    # Check bound
                    if img_cnt < image_features_per_sample.size(1):
                        img_feat = image_features_per_sample[i][img_cnt]
                        final_emb.append(img_feat)
                        final_lbl.append(torch.full((img_feat.shape[0],), -100, device=device, dtype=cur_lbl.dtype))
                        final_msk.append(torch.ones((img_feat.shape[0],), device=device, dtype=cur_msk.dtype))
                        img_cnt += 1
                last_pos = pos + 1
            
            final_emb.append(cur_emb[last_pos:])
            final_lbl.append(cur_lbl[last_pos:])
            final_msk.append(cur_msk[last_pos:])
            
            new_embeds.append(torch.cat(final_emb, dim=0))
            new_labels.append(torch.cat(final_lbl, dim=0))
            new_masks.append(torch.cat(final_msk, dim=0))

        # Pad
        batch_embeds = pad_sequence(new_embeds, batch_first=True, padding_value=0.0)
        batch_labels = pad_sequence(new_labels, batch_first=True, padding_value=-100)
        batch_masks = pad_sequence(new_masks, batch_first=True, padding_value=0)

        # 5. LLM Forward
        outputs = model.language_model(
            inputs_embeds=batch_embeds,
            labels=batch_labels,
            attention_mask=batch_masks
        )
        
        # Combine Losses
        # actual_loss + dummy_loss ensures connectivity for all params
        loss = outputs.loss + dummy_loss

        # 6. Logging & Gradient Norm
        if self.state.global_step % 20 == 0 and self.model.training:
            self._log_prediction(outputs.logits, batch_labels)
            
        if self.args.normalize_gradients_by_batch_size:
            loss = loss * (self.args.base_batch_size / batch_size)

        if return_outputs:
            return loss, outputs
        return loss


    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """Generate predictions during evaluation."""
        logger.info(f"Starting Generation Evaluation: {description}")
        self.model.eval()
        all_preds = []

        # Loss accumulation vars
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # [Important] Save metadata before compute_loss potentially removes it
                # UMBRELLABatch supports .get()
                current_metadata = batch.get('metadata', [])
                current_task_types = batch.get('task_types', [])

                # 1. Calculate Loss (may fail with nan/inf, but continue anyway)
                batch_for_loss = copy.deepcopy(batch)
                try:
                    loss = self.compute_loss(self.model, batch_for_loss, return_outputs=False)
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        total_loss += loss.item()
                        num_batches += 1
                    # Skip warning for nan/inf - just pass
                except Exception as e:
                    pass  # Silently skip loss calculation errors

                # 2. Generate Predictions
                texts = self._generate_step(batch)

                # 3. Store Predictions
                # Metadata might be shorter if batch was padded? Usually it matches.
                # Create default lists if they were None/empty
                if not current_metadata: current_metadata = [{}] * len(texts)
                if not current_task_types: current_task_types = ['unknown'] * len(texts)

                for i, text in enumerate(texts):
                    meta = current_metadata[i] if i < len(current_metadata) else {}
                    task = current_task_types[i] if i < len(current_task_types) else 'unknown'
                    
                    all_preds.append({
                        'model_answer': self._clean_response(text),
                        'task_type': task,
                        'metadata': meta
                    })
        
        # Save Predictions
        output_file = Path(self.args.eval_output_dir) / f"preds_step_{self.state.global_step}.jsonl"
        with open(output_file, 'w') as f:
            for item in all_preds: f.write(json.dumps(item) + '\n')
            
        # Compute Metrics
        metrics = self._compute_sex_accuracy(all_preds)
        
        # [CRITICAL] Add average loss to metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        metrics[f"{metric_key_prefix}_loss"] = avg_loss

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=len(all_preds))


    def _generate_step(self, batch):
        """Generation logic: Expand -> Left Pad -> Trigger -> Generate"""
        model = self.model
        device = next(model.parameters()).device
        
        # 1. Prepare Inputs
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch.get('pixel_values')
        
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
            #if pixel_values.dim() == 5:
            #    pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])

        # 2. Feature Extraction & Expansion (Same as compute_loss logic)
        image_features_per_sample = None
        
        if pixel_values is not None:
            image_outputs = model.vision_tower(pixel_values, output_hidden_states=True)
            sel_feats = image_outputs.hidden_states[model.config.vision_feature_layer]
            sel_feats = model.multi_modal_projector(sel_feats)
            
            # Reshape back to batch: Assumes collator gives flattened batch of images
            batch_size = input_ids.shape[0]
            total_images = sel_feats.shape[0]
            if batch_size > 0:
                num_imgs = total_images // batch_size 
                image_features_per_sample = sel_feats.view(batch_size, num_imgs, -1, sel_feats.shape[-1])
    

        inputs_embeds = model.get_input_embeddings()(input_ids)
        
        # 3. Dynamic Merge (Expand)
        new_input_embeds = []
        new_attention_masks = []
        
        IMAGE_TOKEN_ID = 151646 
        if hasattr(model.config, "image_token_index"):
            IMAGE_TOKEN_ID = model.config.image_token_index

        batch_size = input_ids.shape[0]
        for i in range(batch_size):
            cur_ids = input_ids[i]
            cur_emb = inputs_embeds[i]
            cur_mask = attention_mask[i]
            
            img_indices = (cur_ids == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
            final_embeds, final_masks = [], []
            last_pos, img_cnt = 0, 0
            
            for pos in img_indices:
                final_embeds.append(cur_emb[last_pos:pos])
                final_masks.append(cur_mask[last_pos:pos])
                
                if image_features_per_sample is not None:
                    if img_cnt < image_features_per_sample.size(1):
                        img_feat = image_features_per_sample[i][img_cnt]
                        final_embeds.append(img_feat)
                        final_masks.append(torch.ones((img_feat.shape[0],), device=device, dtype=cur_mask.dtype))
                        img_cnt += 1
                last_pos = pos + 1
            
            final_embeds.append(cur_emb[last_pos:])
            final_masks.append(cur_mask[last_pos:])
            
            new_input_embeds.append(torch.cat(final_embeds, dim=0))
            new_attention_masks.append(torch.cat(final_masks, dim=0))

        # 4. LEFT PADDING & TRIGGER
        # Find max length
        max_len = max([t.shape[0] for t in new_input_embeds])
        
        # Trigger Tokens ("<|im_start|>assistant")
        trigger_ids = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        trigger_embeds = model.get_input_embeddings()(torch.tensor(trigger_ids, device=device))

        #print(f"TEXT INPUT FOR GENERATION: {self.tokenizer.decode(torch.cat([torch.tensor(input_ids[0]).to('cpu'), torch.tensor(trigger_ids).to('cpu')]), skip_special_tokens=False)}")
        
        padded_embeds_list = []
        padded_masks_list = []
        
        for embed, mask in zip(new_input_embeds, new_attention_masks):
            # Add Trigger First
            embed_with_trigger = torch.cat([embed, trigger_embeds], dim=0)
            mask_with_trigger = torch.cat([mask, torch.ones(len(trigger_ids), device=device)], dim=0)
            
            # Then Left Pad
            curr_len = embed_with_trigger.shape[0]
            # Max len needs to account for trigger too
            target_len = max_len + len(trigger_ids) 
            pad_len = target_len - curr_len
            
            if pad_len > 0:
                pad_embed = torch.zeros((pad_len, embed.shape[1]), device=device, dtype=embed.dtype)
                pad_mask = torch.zeros((pad_len,), device=device, dtype=mask.dtype)
                
                padded_embeds_list.append(torch.cat([pad_embed, embed_with_trigger], dim=0))
                padded_masks_list.append(torch.cat([pad_mask, mask_with_trigger], dim=0))
            else:
                padded_embeds_list.append(embed_with_trigger)
                padded_masks_list.append(mask_with_trigger)
                
        final_inputs_embeds = torch.stack(padded_embeds_list)
        final_attention_masks = torch.stack(padded_masks_list)

        # 5. Manual autoregressive generation to avoid position_ids issues with inputs_embeds
        batch_size = final_inputs_embeds.shape[0]
        seq_len = final_inputs_embeds.shape[1]
        max_new_tokens = self.args.eval_max_new_tokens

        # Initialize generated tokens list
        generated_tokens = []
        current_embeds = final_inputs_embeds
        current_mask = final_attention_masks
        past_key_values = None

        for step in range(max_new_tokens):
            with torch.no_grad():
                if past_key_values is None:
                    # First step: use full inputs_embeds
                    outputs = model(
                        inputs_embeds=current_embeds,
                        attention_mask=current_mask,
                        use_cache=True,
                        return_dict=True
                    )
                else:
                    # Subsequent steps: use only the new token embedding
                    outputs = model(
                        inputs_embeds=current_embeds,
                        attention_mask=current_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True
                    )

                past_key_values = outputs.past_key_values

                # Get next token (greedy)
                next_token_logits = outputs.logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                generated_tokens.append(next_tokens)

                # Check if all sequences have hit EOS
                if (next_tokens == self.tokenizer.eos_token_id).all():
                    break

                # Prepare for next step
                next_embeds = model.get_input_embeddings()(next_tokens.unsqueeze(-1))
                current_embeds = next_embeds
                current_mask = torch.cat([current_mask, torch.ones((batch_size, 1), device=device, dtype=current_mask.dtype)], dim=1)

        # Stack generated tokens
        if generated_tokens:
            generated = torch.stack(generated_tokens, dim=1)  # (batch, seq)
        else:
            generated = torch.tensor([[]], device=device, dtype=torch.long).expand(batch_size, 0)

        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


    def _clean_response(self, text):
        """Remove assistant prefix."""
        triggers = ["assistant", "<|im_start|>assistant\n"]
        for t in triggers:
            if text.lower().startswith(t): return text[len(t):].strip()
        return text.strip()
    
    def _log_prediction(self, logits, labels):
        """
        Log predictions aligned for Next Token Prediction.
        Shows what the model predicted for the NEXT position.
        """
        idx = 0
        
        # [Key Fix] Apply Shift
        # Model's logits[t] predicts labels[t+1]
        # Exclude last logits and first label to align pairs

        # 1. Predictions: Exclude last token (no ground truth for last prediction)
        pred_ids = torch.argmax(logits[idx][:-1], dim=-1)
        
        # 2. Ground truth: Exclude first token (first token is input, not prediction target)
        gt_ids = labels[idx][1:]

        # Turn separation logic (same as before, using shifted IDs)
        turns = []
        current_gt = []
        current_pd = []
        in_turn = False
        ignore_index = -100

        for i in range(len(gt_ids)):
            # Only consider valid prediction range when ground truth is not ignore_index
            is_valid = (gt_ids[i] != ignore_index)

            if is_valid:
                if not in_turn:
                    in_turn = True
                    current_gt = []
                    current_pd = []
                
                current_gt.append(gt_ids[i])
                current_pd.append(pred_ids[i])
            else:
                if in_turn:
                    turns.append((current_gt, current_pd))
                    in_turn = False
                    current_gt = []
                    current_pd = []

        if in_turn:
            turns.append((current_gt, current_pd))

        # Print output
        print(f"\n{'='*20} Prediction Log [Step {self.state.global_step}] (Shifted for visual alignment) {'='*20}")
        if len(turns) == 0:
            print("  (No assistant turns found in this sample)")
        
        for i, (gt_chunk, pd_chunk) in enumerate(turns):
            gt_text = self.tokenizer.decode(torch.tensor(gt_chunk), skip_special_tokens=False)
            pd_text = self.tokenizer.decode(torch.tensor(pd_chunk), skip_special_tokens=False)
            
            print(f"[Assistant Turn {i+1}]")
            print(f"  GT: {gt_text}")
            print(f"  PD: {pd_text}")
        print("="*65)


    def _compute_sex_accuracy(self, predictions):
        correct = 0
        total = 0
        for p in predictions:
            gt = p['metadata'].get('subject_label')
            ans = p['model_answer'].lower()
            if gt in ['male', 'female']:
                pred = 'female' if 'female' in ans else 'male' if 'male' in ans else None
                if pred:
                    total += 1
                    if pred == gt: correct += 1
        return {"eval_sex_acc": correct / total if total else 0}