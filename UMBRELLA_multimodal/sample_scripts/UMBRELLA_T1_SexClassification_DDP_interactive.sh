#!/bin/bash
# UMBRELLA: LLaVA-based Brain MRI Training Script
# Distributed Data Parallel (DDP) Interactive Mode

set +x

cd /pscratch/sd/h/heehaw/BrainVLM/UMBRELLA_gemini  #TODO: Change to your own scratch space

module load python
#module load pytorch/1.13.1
module load cpe/23.03

conda activate /pscratch/sd/h/heehaw/anaconda/BrainVLM_llava   #TODO: Change to your own conda env
# conda activate py39
# pip install timm
#export MASTER_ADDR=`/bin/hostname -s`
#export MASTER_PORT=29500
#export MASTER_PORT=$(shuf -i 29500-65535 -n 1)

export LIBRARY_PATH=$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=/pscratch/sd/h/heehaw   #TODO: Change to your own scratch space
export HF_HOME=/pscratch/sd/h/heehaw/huggingface   #TODO: Change to your own scratch space
export TORCH_HOME=/pscratch/sd/h/heehaw/   #TODO: Change to your own scratch space

# #recent version (24.3.30)
python project/training/main_umbrella_training.py \
      --config /pscratch/sd/h/heehaw/BrainVLM/UMBRELLA_gemini/project/config/umbrella_llava_train_SexClassification.yaml \
      --train-data ./sample_data/sex_classification_conversations_100subjects/train_conversations.jsonl \
      --eval-data ./sample_data/sex_classification_conversations_100subjects/validation_conversations.jsonl \
      --modality sMRI \
      --output-dir ./hf_results/umbrella_SexClassification \
      --eval-output-dir /pscratch/sd/h/heehaw/BrainVLM/UMBRELLA_gemini/eval_predictions_SexClassification_100subjects

# --limit_training_samples 1000
# --mask_patch_size 12 12 12 20
 
