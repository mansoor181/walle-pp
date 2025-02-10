import subprocess
from concurrent.futures import ThreadPoolExecutor

# Define the embedding methods and their dimensions
embedding_methods = {
    'ESM2': {'script': 'esm2.py', 'method': 'embed_esm2', 'dim': 1280, 'name': 'ESM2'},
    'ESM-IF': {'script': 'esm_if1.py', 'method': 'embed_esm_if', 'dim': 512, 'name': 'ESM-IF'},
    'AntiBERTy': {'script': 'antiberty.py', 'method': 'embed_antiberty', 'dim': 512, 'name': 'AntiBERTy'},
    'ProtBERT': {'script': 'protbert.py', 'method': 'embed_protbert', 'dim': 1024, 'name': 'ProtBERT'}
    # 'BLOSUM62': {'script': 'blosum62.py', 'method': 'embed_blosum62', 'dim': 24, 'name': 'BLOSUM62'},
    # 'One-Hot': {'script': None, 'method': None, 'dim': 20, 'name': 'One-Hot'}  # Special case
}

# Define the configurations
ab_models = ['ESM2', 'ESM-IF', 'AntiBERTy', 'ProtBERT']
ag_models = ['ESM2', 'ESM-IF', 'ProtBERT']
base_models = ['BLOSUM62', 'One-Hot']

# Run the setup-wandb.sh script once
# subprocess.run("source ~/Documents/GSU/Projects/Antibody-Design/epitope-prediction/epitope-pred/asepcode/train-walle/scripts/setup-wandb.sh", 
#                shell=True, executable="/bin/zsh")

def run_training(ab_model, ag_model):
    # Set up the command for train.py
    command = [
        "python", "train.py",
        "mode=train",
        f"wandb_init.project=retrain-walle-group",
        f"wandb_init.notes={ab_model.lower()}-{ag_model.lower()}",
        f"wandb_init.tags={ab_model.lower()}-{ag_model.lower()}",
        "dataset.node_feat_type=custom",
        f"dataset.ab.custom_embedding_method_src.script_path=asep/data/embedding/{embedding_methods[ab_model]['script']}",
        f"dataset.ab.custom_embedding_method_src.method_name={embedding_methods[ab_model]['method']}",
        f"dataset.ab.custom_embedding_method_src.name={embedding_methods[ab_model]['name']}",
        f"dataset.ag.custom_embedding_method_src.script_path=asep/data/embedding/{embedding_methods[ag_model]['script']}",
        f"dataset.ag.custom_embedding_method_src.method_name={embedding_methods[ag_model]['method']}",
        f"dataset.ag.custom_embedding_method_src.name={embedding_methods[ag_model]['name']}",
        f"hparams.input_ab_dim={embedding_methods[ab_model]['dim']}",
        f"hparams.input_ag_dim={embedding_methods[ag_model]['dim']}",
        "hparams.model_type=linear",
        "dataset.split_method=epitope_group",
        "hparams.max_epochs=300",
        "hparams.pos_weight=100",
        "hparams.train_batch_size=128",
        "hparams.val_batch_size=32",
        "hparams.test_batch_size=32"
    ]

    # # Special handling for base models (BLOSUM62 and One-Hot)
    # if ab_model in base_models and ag_model in base_models:
    #     command.extend([
    #         "dataset.node_feat_type=one_hot",
    #         "dataset.ab.embedding_model=one_hot",
    #         "dataset.ag.embedding_model=one_hot"
    #     ])

    # Execute the command
    print(f"Running configuration: {ab_model}-{ag_model}")
    subprocess.run(command, check=True)

# Create a pool of workers
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = []
    # Run all combinations of ab_models and ag_models
    for ab in ab_models:
        for ag in ag_models:
            futures.append(executor.submit(run_training, ab, ag))
    # # Run base models (BLOSUM62-BLOSUM62 and One-Hot-One-Hot)
    # for base in base_models:
    #     futures.append(executor.submit(run_training, base, base))

    # Wait for all futures to complete
    for future in futures:
        future.result()
