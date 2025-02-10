import os
import json
import wandb
from concurrent.futures import ProcessPoolExecutor, as_completed

# Initialize WandB API
api = wandb.Api()

# Set your WandB entity and project name
entity = "alibilab-gsu"
project = "retrain-walle-group"

# Fetch all runs in the project
runs = api.runs(f"{entity}/{project}")

# Function to check if a run failed
def is_failed_run(run):
    return run.state == "crashed" or run.state == "failed"

# Function to parse the config if it's a string
def parse_config(config):
    if isinstance(config, str):
        try:
            return json.loads(config)
        except json.JSONDecodeError:
            print(f"Failed to parse config: {config}")
            return None
    return config

# Function to re-run a configuration
def rerun_config(config):
    try:
        # Construct the command to run the script
        command = (
            f"python train.py "
            f"mode={config['mode']} "
            f"wandb_init.project={config['wandb_init']['project']} "
            f"wandb_init.entity={config['wandb_init']['entity']} "
            f"wandb_init.notes={config['wandb_init']['notes']} "
            f"wandb_init.tags={config['wandb_init']['tags']} "
            f"dataset.node_feat_type={config['dataset']['node_feat_type']} "
            f"dataset.ab.custom_embedding_method_src.script_path={config['dataset']['ab']['custom_embedding_method_src']['script_path']} "
            f"dataset.ab.custom_embedding_method_src.method_name={config['dataset']['ab']['custom_embedding_method_src']['method_name']} "
            f"dataset.ab.custom_embedding_method_src.name={config['dataset']['ab']['custom_embedding_method_src']['name']} "
            f"dataset.ag.custom_embedding_method_src.script_path={config['dataset']['ag']['custom_embedding_method_src']['script_path']} "
            f"dataset.ag.custom_embedding_method_src.method_name={config['dataset']['ag']['custom_embedding_method_src']['method_name']} "
            f"dataset.ag.custom_embedding_method_src.name={config['dataset']['ag']['custom_embedding_method_src']['name']} "
            f"hparams.input_ab_dim={config['hparams']['input_ab_dim']} "
            f"hparams.input_ag_dim={config['hparams']['input_ag_dim']} "
            f"hparams.model_type={config['hparams']['model_type']} "
            f"dataset.split_method={config['dataset']['split_method']} "
            f"hparams.max_epochs={config['hparams']['max_epochs']} "
            f"hparams.pos_weight={config['hparams']['pos_weight']} "
            f"hparams.train_batch_size={config['hparams']['train_batch_size']} "
            f"hparams.val_batch_size={config['hparams']['val_batch_size']} "
            f"hparams.test_batch_size={config['hparams']['test_batch_size']}"
        )
        print(f"Running: {command}")
        os.system(command)
    except KeyError as e:
        print(f"KeyError in config: {e}")
    except Exception as e:
        print(f"Error in rerun_config: {e}")

# Main function to re-run all failed runs in parallel
def rerun_failed_runs():
    # Collect configurations of failed runs
    failed_configs = []
    for run in runs:
        if is_failed_run(run):
            print(f"Found failed run: {run.name} (ID: {run.id})")
            config = parse_config(run.config)
            if config:
                failed_configs.append(config)

    # Re-run failed configurations in parallel
    num_workers = 40  # Adjust based on your CPU cores
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(rerun_config, config) for config in failed_configs]
        for future in as_completed(futures):
            try:
                future.result()
                print("Task completed successfully.")
            except Exception as e:
                print(f"Task failed with error: {e}")

# Execute the script
if __name__ == "__main__":
    rerun_failed_runs()
