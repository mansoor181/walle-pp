#!/bin/bash

# Initialize mode variable
mode=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      mode="$2"
      shift 2  # Shift past the flag and its value
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Check if the mode argument is provided
if [[ -z ${mode} ]]; then
  echo "mode is not provided, choices: dev, train"
  exit 1
fi

# Source the setup-wandb.sh script to set environment variables
# source ~/Documents/GSU/Projects/Antibody-Design/epitope-prediction/epitope-pred/asepcode/train-walle/scripts/setup-wandb.sh

# ------------------------------------------------------------------------------
# dev
# ------------------------------------------------------------------------------
if [[ ${mode} == 'dev' ]]; then
  python3 train.py \
    mode='dev' \
    "wandb_init.project=retrain-walle-dev" \
    "wandb_init.notes='BLOSUM62'" \
    hparams.max_epochs=300 \
    hparams.pos_weight=100 \
    hparams.train_batch_size=128 \
    hparams.val_batch_size=32 \
    hparams.test_batch_size=32 \
    dataset.node_feat_type='custom' \
    dataset.ab.custom_embedding_method_src.script_path='asep/data/embedding/blosum62.py' \
    dataset.ab.custom_embedding_method_src.method_name='embed_blosum62' \
    dataset.ab.custom_embedding_method_src.name='blosum62' \
    dataset.ag.custom_embedding_method_src.script_path='asep/data/embedding/blosum62.py' \
    dataset.ag.custom_embedding_method_src.method_name='embed_blosum62' \
    dataset.ag.custom_embedding_method_src.name='blosum62' \
    hparams.input_ab_dim=24 \
    hparams.input_ag_dim=24
fi

# ------------------------------------------------------------------------------
# train
# ------------------------------------------------------------------------------
if [[ ${mode} == 'train' ]]; then
  python3 train.py \
    mode='train' \
    "wandb_init.project=retrain-walle-group" \
    "wandb_init.notes='blosum62'" \
    "wandb_init.tags='blosum62'" \
    hparams.max_epochs=300 \
    hparams.pos_weight=100 \
    hparams.train_batch_size=128 \
    hparams.val_batch_size=32 \
    hparams.test_batch_size=32 \
    dataset.node_feat_type='custom' \
    dataset.ab.custom_embedding_method_src.script_path='asep/data/embedding/blosum62.py' \
    dataset.ab.custom_embedding_method_src.method_name='embed_blosum62' \
    dataset.ab.custom_embedding_method_src.name='BLOSUM62' \
    dataset.ag.custom_embedding_method_src.script_path='asep/data/embedding/blosum62.py' \
    dataset.ag.custom_embedding_method_src.method_name='embed_blosum62' \
    dataset.ag.custom_embedding_method_src.name='BLOSUM62' \
    hparams.input_ab_dim=24 \
    hparams.input_ag_dim=24 \
    hparams.model_type='graph'	
    # "callbacks.early_stopping=null"
fi


# ./train-walle/scripts/train-blosum62.sh --mode train 
