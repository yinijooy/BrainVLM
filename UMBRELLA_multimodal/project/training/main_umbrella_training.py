"""
UMBRELLA Training Script (Refactored)

- Integrated UMBRELLATrainingPipeline
- Uses Unified UMBRELLATrainer for both training and generation-based evaluation
- Simplified Data Collator instantiation
"""

import sys
import logging
import argparse
import yaml
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

from transformers import (
    AutoTokenizer,
    LlavaForConditionalGeneration
)

# Import UMBRELLA components
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataset.umbrella_dataset import UMBRELLADataset
from dataset.umbrella_collator import UMBRELLACollator
# Updated Trainer import (Unified version)
from training.umbrella_trainer import UMBRELLATrainer, UMBRELLATrainingArgs
from model.patch_embed import PatchEmbed

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


@dataclass
class UMBRELLATrainingConfig:
    """Unified configuration for UMBRELLA training."""

    # Model settings
    model_name: str = "llava-hf/llava-interleave-qwen-0.5b-hf"
    tokenizer_name: Optional[str] = None

    # Training data
    train_json_path: str = "./data/train.json"
    eval_json_path: Optional[str] = None

    # Task filtering
    task_filter: Optional[str] = None

    # Modality settings: 'sMRI', 'fMRI', 'T1', 'FA', 'T1_FA'
    modality: str = "sMRI"
    img_size: List[int] = None
    patch_size: List[int] = None

    # Multi-modality settings (for custom PatchEmbed)
    sMRI_img_size: List[int] = None
    sMRI_patch_size: List[int] = None
    fMRI_img_size: List[int] = None
    fMRI_patch_size: List[int] = None

    # T1/FA multimodal settings (BrainVLM style)
    T1_img_size: List[int] = None
    T1_patch_size: List[int] = None
    FA_img_size: List[int] = None
    FA_patch_size: List[int] = None

    # Training hyperparameters
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    num_epochs: int = 50
    max_seq_length: int = 2048
    max_images_per_sample: int = 2

    # Memory and performance
    enable_memory_aware_batching: bool = True
    memory_budget_gb: float = 30.0
    gradient_checkpointing: bool = True
    mixed_precision: str = "fp32"

    # Logging and saving
    output_dir: str = "./hf_results/umbrella"
    logging_steps: int = 1
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    warmup_steps: int = 100

    # Advanced logging
    log_image_statistics: bool = True
    
    # Gradient normalization
    normalize_gradients_by_batch_size: bool = True
    base_batch_size: int = 32

    # Weights & Biases
    use_wandb: bool = True
    wandb_project: str = "umbrella-training"
    wandb_api_key: Optional[str] = None

    # Evaluation generation settings
    eval_max_new_tokens: int = 256
    eval_temperature: float = 0.7
    eval_top_p: float = 0.9
    eval_output_dir: str = "./eval_predictions"

    @classmethod
    def from_yaml(cls, config_path: str) -> 'UMBRELLATrainingConfig':
        yaml_config = load_config(config_path)
        dataset_config = yaml_config.get('dataset', {})
        model_config = yaml_config.get('model', {})
        trainer_config = yaml_config.get('trainer', {})

        modality = yaml_config.get('modality', 'sMRI')  # Read from config, default sMRI
        modality_dataset_config = dataset_config.get(modality, {})
        img_size = modality_dataset_config.get('img_size', [96, 96, 96])
        patch_size = model_config.get(modality, {}).get('patch_size', [16, 16, 16])

        sMRI_config = dataset_config.get('sMRI', {})
        sMRI_model_config = model_config.get('sMRI', {})
        fMRI_config = dataset_config.get('fMRI', {})
        fMRI_model_config = model_config.get('fMRI', {})

        # T1/FA configs (BrainVLM style)
        T1_config = dataset_config.get('T1', {})
        T1_model_config = model_config.get('T1', {})
        FA_config = dataset_config.get('FA', {})
        FA_model_config = model_config.get('FA', {})

        return cls(
            model_name=model_config.get('hf_name', 'llava-hf/llava-interleave-qwen-0.5b-hf'),
            modality=modality,
            img_size=img_size,
            patch_size=patch_size,
            sMRI_img_size=sMRI_config.get('img_size', [96, 96, 96]),
            sMRI_patch_size=sMRI_model_config.get('patch_size', [16, 16, 16]),
            fMRI_img_size=fMRI_config.get('img_size', [96, 96, 96, 24]),
            fMRI_patch_size=fMRI_model_config.get('patch_size', [16, 16, 16, 3]),
            T1_img_size=T1_config.get('img_size', [120, 120, 120]),
            T1_patch_size=T1_model_config.get('patch_size', [10, 10, 10]),
            FA_img_size=FA_config.get('img_size', [120, 120, 120]),
            FA_patch_size=FA_model_config.get('patch_size', [10, 10, 10]),
            batch_size=trainer_config.get('per_device_batch_size', 2),
            gradient_accumulation_steps=trainer_config.get('gradient_accumulation_steps', 1),
            learning_rate=trainer_config.get('learning_rate', 5e-5),
            num_epochs=trainer_config.get('max_epochs', 50),
            max_seq_length=2048,
            gradient_checkpointing=trainer_config.get('gradient_checkpointing', True),
            warmup_steps=trainer_config.get('warmup_steps', 500),
            output_dir=trainer_config.get('output_dir', './hf_results/umbrella'),
            logging_steps=trainer_config.get('logging_steps', 1),
            eval_steps=trainer_config.get('eval_steps', 100),
            save_steps=trainer_config.get('save_steps', 500),
            wandb_api_key=yaml_config.get('wandb', {}).get('API_KEY'),
            wandb_project=yaml_config.get('wandb', {}).get('project', 'umbrella-training')
        )

    def to_training_args(self, eval_dataset_available: bool = False) -> UMBRELLATrainingArgs:
        """Convert config to UMBRELLATrainingArgs."""
        return UMBRELLATrainingArgs(
            # HF Args
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            logging_steps=self.logging_steps,
            eval_steps=self.eval_steps if eval_dataset_available else None,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            fp16=self.mixed_precision == "fp16",
            bf16=self.mixed_precision == "bf16",
            save_strategy="steps",
            evaluation_strategy="steps" if eval_dataset_available else "no",
            logging_strategy="steps",
            report_to="wandb" if self.use_wandb else "none",
            load_best_model_at_end=eval_dataset_available,
            #metric_for_best_model="loss" if eval_dataset_available else None,
            metric_for_best_model="sex_acc",    # NOTE: YOU SHOULD CHANGE REGARDING YOUR TASK
            greater_is_better=True if eval_dataset_available else None,     # NOTE: YOU SHOULD CHANGE REGARDING YOUR TASK
            gradient_checkpointing=self.gradient_checkpointing,
            remove_unused_columns=False,

            # UMBRELLA-specific Args
            enable_memory_aware_batching=self.enable_memory_aware_batching,
            memory_budget_gb=self.memory_budget_gb,
            log_image_statistics=self.log_image_statistics,
            normalize_gradients_by_batch_size=self.normalize_gradients_by_batch_size,
            base_batch_size=self.base_batch_size,
            
            # Eval Generation Args
            eval_output_dir=self.eval_output_dir,
            eval_max_new_tokens=self.eval_max_new_tokens,
            eval_temperature=self.eval_temperature,
            eval_top_p=self.eval_top_p
        )


def create_llava_model_with_custom_patch_embed(config: UMBRELLATrainingConfig) -> LlavaForConditionalGeneration:
    """Helper to load model and replace PatchEmbed."""
    logger.info(f"Loading pre-trained model: {config.model_name}")
    model = LlavaForConditionalGeneration.from_pretrained(config.model_name)

    # Create custom PatchEmbed
    original_patch_embedding = model.vision_tower.vision_model.embeddings.patch_embedding
    embed_dim = int(original_patch_embedding.out_channels)

    patch_embed = PatchEmbed(
        sMRI_size=config.sMRI_img_size or [96, 96, 96],
        sMRI_patch_size=config.sMRI_patch_size or [16, 16, 16],
        fMRI_size=config.fMRI_img_size or [96, 96, 96, 24],
        fMRI_patch_size=config.fMRI_patch_size or [16, 16, 16, 3],
        T1_size=config.T1_img_size or [120, 120, 120],
        T1_patch_size=config.T1_patch_size or [10, 10, 10],
        FA_size=config.FA_img_size or [120, 120, 120],
        FA_patch_size=config.FA_patch_size or [10, 10, 10],
        embed_dim=embed_dim,
        modality_type=config.modality  # Pass modality type
    )

    logger.info(f"Created PatchEmbed with modality_type={config.modality}")
    if config.modality in ['T1', 'FA', 'T1_FA']:
        logger.info(f"T1 patches: {patch_embed.T1_num_patches}, FA patches: {patch_embed.FA_num_patches}")

    # Replace & Freeze strategy
    setattr(model.vision_tower.vision_model, "embeddings", patch_embed)

    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the new embeddings
    for name, param in model.vision_tower.vision_model.named_parameters():
        if 'embeddings' in name:
            param.requires_grad = True

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model


class UMBRELLATrainingPipeline:
    """Unified Pipeline for Model, Data, and Trainer setup."""

    def __init__(self, config: UMBRELLATrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("UMBRELLA Training Pipeline Initialized")

    def setup_model(self) -> Tuple[LlavaForConditionalGeneration, AutoTokenizer]:
        tokenizer_name = self.config.tokenizer_name or self.config.model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.padding_side = "left" # Crucial for generation

        # Load Model
        model = create_llava_model_with_custom_patch_embed(self.config)


        # Add special tokens if needed 
        # 필요한 특수 토큰 목록
        tokens_to_add = ['<|im_start|>', '<|im_end|>', '<image>']
        missing_tokens = [t for t in tokens_to_add if t not in tokenizer.get_vocab()]

        # 없는 토큰만 추가
        if missing_tokens:
            print(f"Adding missing tokens: {missing_tokens}")
            tokenizer.add_special_tokens({'additional_special_tokens': missing_tokens})
            
            # [중요] 토큰을 새로 추가했을 때만 임베딩 크기를 조절해야 함
            model.resize_token_embeddings(len(tokenizer))
        else:
            print("All special tokens already exist. Skipping add_special_tokens.")
                
        
        return model, tokenizer

    def create_dataset(self, json_path: str, tokenizer, mode: str = 'train') -> UMBRELLADataset:
        # Determine correct img_size based on modality
        if self.config.modality in ['T1', 'T1_FA']:
            img_size = tuple(self.config.T1_img_size) if self.config.T1_img_size else (120, 120, 120)
        elif self.config.modality == 'FA':
            img_size = tuple(self.config.FA_img_size) if self.config.FA_img_size else (120, 120, 120)
        elif self.config.modality == 'fMRI':
            img_size = tuple(self.config.fMRI_img_size) if self.config.fMRI_img_size else (96, 96, 96, 24)
        else:  # sMRI or default
            img_size = tuple(self.config.img_size) if self.config.img_size else (96, 96, 96)

        logger.info(f"Creating dataset with modality={self.config.modality}, img_size={img_size}")

        dataset = UMBRELLADataset(
            data_path=json_path,
            tokenizer=tokenizer,
            mode=mode,
            img_size=img_size,
            max_seq_length=self.config.max_seq_length,
            max_images_per_sample=self.config.max_images_per_sample,
            modality_type=self.config.modality  # Pass modality_type for patch embedding dispatch
        )
        return dataset

    def create_collator(self, tokenizer) -> UMBRELLACollator:
        collator = UMBRELLACollator(
            tokenizer=tokenizer,
        )
        return collator

    def train(self):
        # 1. Setup W&B
        if self.config.use_wandb and self.config.wandb_api_key:
            import wandb
            wandb.login(key=self.config.wandb_api_key)
            wandb.init(project=self.config.wandb_project, config=vars(self.config))

        # 2. Setup Model & Tokenizer
        model, tokenizer = self.setup_model()

        # 3. Create Datasets
        train_dataset = self.create_dataset(self.config.train_json_path, tokenizer, mode='train')
        eval_dataset = None
        if self.config.eval_json_path:
            eval_dataset = self.create_dataset(self.config.eval_json_path, tokenizer, mode='eval')

        # 4. Create Collator
        collator = self.create_collator(tokenizer)

        # 5. Create Training Args
        training_args = self.config.to_training_args(eval_dataset_available=(eval_dataset is not None))

        # 6. Initialize Unified Trainer
        # No more 'create_evaluation_trainer' - directly use UMBRELLATrainer
        trainer = UMBRELLATrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            tokenizer=tokenizer
        )
        
        logger.info("Starting Training...")
        trainer.train()

        # 7. Save Final Model
        final_output_dir = Path(self.config.output_dir) / "final_model"
        trainer.save_model(str(final_output_dir))
        tokenizer.save_pretrained(str(final_output_dir))
        logger.info(f"Training Complete. Model saved to {final_output_dir}")


def main():
    parser = argparse.ArgumentParser(description="UMBRELLA Training Pipeline")
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config')
    parser.add_argument('--train-data', type=str, required=True)
    parser.add_argument('--eval-data', type=str)
    parser.add_argument('--modality', type=str, default='sMRI')
    parser.add_argument('--task-filter', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--eval-output-dir', type=str)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--no-wandb', action='store_true')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Loading config from {args.config}")
    
    config = UMBRELLATrainingConfig.from_yaml(args.config)
    
    # Overrides
    config.train_json_path = args.train_data
    config.eval_json_path = args.eval_data
    config.modality = args.modality
    if args.task_filter: config.task_filter = args.task_filter
    if args.output_dir: config.output_dir = args.output_dir
    if args.batch_size: config.batch_size = args.batch_size
    if args.learning_rate: config.learning_rate = args.learning_rate
    if args.eval_output_dir: config.eval_output_dir = args.eval_output_dir
    if args.no_wandb: config.use_wandb = False

    # Create output dir
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Run Pipeline
    pipeline = UMBRELLATrainingPipeline(config)
    pipeline.train()


if __name__ == "__main__":
    main()