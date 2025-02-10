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

# ------------------------------------------------------------------------------
# dev
# ------------------------------------------------------------------------------
if [[ ${mode} == 'dev' ]]; then
  python3.10 train.py \
    mode='dev' \
    "wandb_init.project=retrain-walle" \
    "wandb_init.notes='protbert'" \
    "wandb_init.tags='protbert'" \
    hparams.max_epochs=2 \
    hparams.pos_weight=100 \
    hparams.train_batch_size=128 \
    hparams.val_batch_size=32 \
    hparams.test_batch_size=32 \
    dataset.node_feat_type='custom' \
    dataset.ab.custom_embedding_method_src.script_path='asep/data/embedding/protbert.py' \
    dataset.ab.custom_embedding_method_src.method_name='protbert_residue_embedding' \
    dataset.ab.custom_embedding_method_src.name='protbert' \
    dataset.ag.custom_embedding_method_src.script_path='asep/data/embedding/protbert.py' \
    dataset.ag.custom_embedding_method_src.method_name='protbert_residue_embedding' \
    dataset.ag.custom_embedding_method_src.name='protbert' \
    hparams.input_ab_dim=1024 \
    hparams.input_ag_dim=1024
fi

# ------------------------------------------------------------------------------
# train
# ------------------------------------------------------------------------------
if [[ ${mode} == 'train' ]]; then
  python3.10 train.py \
    mode='train' \
    "wandb_init.project=retrain-walle" \
    "wandb_init.notes='esmif-esm2'" \
    "wandb_init.tags='esmif-esm2'" \
    hparams.max_epochs=3 \
    hparams.pos_weight=100 \
    hparams.train_batch_size=128 \
    hparams.val_batch_size=32 \
    hparams.test_batch_size=32 \
    dataset.node_feat_type='custom' \
    dataset.ab.embedding_model='custom' \
    dataset.ag.embedding_model='custom' \
    dataset.ab.custom_embedding_method_src.script_path='asep/data/embedding/esm_if1.py' \
    dataset.ab.custom_embedding_method_src.method_name='embed_esm_if' \
    dataset.ab.custom_embedding_method_src.name='esmif' \
    dataset.ag.custom_embedding_method_src.script_path='asep/data/embedding/esm2.py' \
    dataset.ag.custom_embedding_method_src.method_name='embed_esm2' \
    dataset.ag.custom_embedding_method_src.name='esm2' \
    hparams.input_ab_dim=512 \
    hparams.input_ag_dim=1280
    # "callbacks.early_stopping=null"
fi

# chmod +x train-walle/scripts/train-esmif.sh
# ./train-walle/scripts/train-esmif.sh --mode train 