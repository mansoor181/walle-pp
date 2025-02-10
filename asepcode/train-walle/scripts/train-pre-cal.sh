#!/bin/zsh

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
source ~/Documents/GSU/Projects/Antibody-Design/epitope-prediction/epitope-pred/asepcode/train-walle/scripts/setup-wandb.sh

# echo "Mode: ${mode}"

# ------------------------------------------------------------------------------
# dev
# ------------------------------------------------------------------------------
if [[ ${mode} == 'dev' ]]; then
  python3 train.py \
    mode='dev' \
    "wandb_init.project=retrain-walle-dev" \
    "wandb_init.notes='pre_cal'" \
    hparams.max_epochs=5 \
    hparams.pos_weight=100 \
    hparams.train_batch_size=128 \
    hparams.val_batch_size=32 \
    hparams.test_batch_size=32 \
    dataset.node_feat_type='pre_cal' \
    dataset.ab.embedding_model='igfold' \
    dataset.ag.embedding_model='esm2' \
    hparams.input_ab_dim=512 \
    hparams.input_ag_dim=480
fi

# ------------------------------------------------------------------------------
# train
# ------------------------------------------------------------------------------
if [[ ${mode} == 'train' ]]; then
  python3.10 train.py \
    mode='train' \
    "wandb_init.project=retrain-walle" \
    "wandb_init.notes='pre_cal'" \
    dataset.node_feat_type='pre_cal' \
    dataset.ab.embedding_model='igfold' \
    dataset.ag.embedding_model='esm2' \
    dataset.split_method='epitope_ratio' \
    hparams.max_epochs=3 \
    hparams.pos_weight=100 \
    hparams.train_batch_size=32 \
    hparams.val_batch_size=32 \
    hparams.test_batch_size=32 \
    hparams.input_ab_dim=480 \
    hparams.input_ag_dim=480 \
    hparams.model_type='linear'
    # "callbacks.early_stopping=null" &
fi


# split_method=null  # choices: [null, epitope_ratio, epitope_group]
# if null, default to epitope_ratio

# Print the mode for debugging
# echo "Mode: ${mode}"


############# test ############
# chmod +x train-pre-cal.sh
# ./train-walle/scripts/train-pre-cal.sh --mode train  


# export WANDB_API_KEY='228bf764d83d4fcb93bb729541c0b58e262ac25a'
# export WANDB_ENTITY='alibilab-gsu'
# export WANDB_PROJECT='retrain-walle'
# export WANDB_MODE='online'
