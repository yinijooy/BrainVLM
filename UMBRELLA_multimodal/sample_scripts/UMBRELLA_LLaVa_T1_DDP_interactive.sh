#!/bin/bash
# UMBRELLA: LLaVA-based Brain MRI Training Script
# Distributed Data Parallel (DDP) Interactive Mode

set +x

cd /pscratch/sd/h/heehaw/UMBRELLA/project   #TODO: Change to your own scratch space


module load python
module load cpe/23.03

conda activate /pscratch/sd/h/heehaw/anaconda/BrainVLM_llava   #TODO: Change to your own conda env

export LIBRARY_PATH=$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=/pscratch/sd/h/heehaw   #TODO: Change to your own scratch space
export HF_HOME=/pscratch/sd/h/heehaw/huggingface   #TODO: Change to your own scratch space
export TORCH_HOME=/pscratch/sd/h/heehaw/   #TODO: Change to your own scratch space


# Run UMBRELLA training with LLaVA architecture
torchrun --nnodes 1 --nproc_per_node 1 main_umbrella_llava_T1.py
