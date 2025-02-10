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
    "wandb_init.notes='one_hot'" \
    hparams.max_epochs=300 \
    hparams.pos_weight=100 \
    hparams.train_batch_size=128 \
    hparams.val_batch_size=32 \
    hparams.test_batch_size=32 \
    dataset.node_feat_type='one_hot' \
    hparams.input_ab_dim=20 \
    hparams.input_ag_dim=20
fi

# ------------------------------------------------------------------------------
# train
# ------------------------------------------------------------------------------
if [[ ${mode} == 'train' ]]; then
  python3 train.py \
    mode='train' \
    "wandb_init.project=retrain-walle-group" \
    "wandb_init.notes='one_hot'" \
    "wandb_init.tags='one_hot'" \
    dataset.node_feat_type='one_hot' \
    dataset.ab.embedding_model='one_hot' \
    dataset.ag.embedding_model='one_hot' \
    dataset.split_method='epitope_group' \
    hparams.max_epochs=300 \
    hparams.pos_weight=100 \
    hparams.train_batch_size=128 \
    hparams.val_batch_size=32 \
    hparams.test_batch_size=32 \
    hparams.input_ab_dim=20 \
    hparams.input_ag_dim=20 \
    hparams.model_type='graph'
    # "callbacks.early_stopping=null" &
fi

# TODO: turn off early stopping
