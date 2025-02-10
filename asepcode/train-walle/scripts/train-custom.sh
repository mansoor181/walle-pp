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
    "wandb_init.project=retrain-walle" \
    "wandb_init.notes='esm2'" \
    hparams.max_epochs=5 \
    hparams.pos_weight=100 \
    hparams.train_batch_size=128 \
    hparams.val_batch_size=32 \
    hparams.test_batch_size=32 \
    dataset.node_feat_type='custom' \
    dataset.ab.custom_embedding_method_src.script_path='asep/data/embedding/esm2.py' \
    dataset.ab.custom_embedding_method_src.method_name='esm2_residue_embedding' \
    dataset.ab.custom_embedding_method_src.name='ESM2' \
    dataset.ag.custom_embedding_method_src.script_path='asep/data/embedding/esm2.py' \
    dataset.ag.custom_embedding_method_src.method_name='esm2_residue_embedding' \
    dataset.ag.custom_embedding_method_src.name='ESM2' \
    hparams.input_ab_dim=640 \
    hparams.input_ag_dim=640
fi

# ------------------------------------------------------------------------------
# train
# ------------------------------------------------------------------------------
if [[ ${mode} == 'train' ]]; then
  python train.py \
    mode='train' \
    "wandb_init.project=retrain-walle-group" \
    "wandb_init.notes='protbert-esm2'" \
    hparams.max_epochs=300 \
    hparams.pos_weight=100 \
    hparams.train_batch_size=128 \
    hparams.val_batch_size=32 \
    hparams.test_batch_size=32 \
    hparams.input_ab_dim=1024 \
    hparams.input_ag_dim=1280 \
    dataset.node_feat_type='custom' \
    dataset.ab.custom_embedding_method_src.script_path='asep/data/embedding/protbert.py' \
    dataset.ab.custom_embedding_method_src.method_name='embed_protbert' \
    dataset.ab.custom_embedding_method_src.name='ProtBERT' \
    dataset.ag.custom_embedding_method_src.script_path='asep/data/embedding/esm2.py' \
    dataset.ag.custom_embedding_method_src.method_name='embed_esm2' \
    dataset.ag.custom_embedding_method_src.name='ESM2' \
    hparams.model_type='linear'
    # "callbacks.early_stopping=null"
fi


############# test ############
# chmod +x train-walle/scripts/train-custom.sh
# ./train-walle/scripts/train-custom.sh --mode train  


