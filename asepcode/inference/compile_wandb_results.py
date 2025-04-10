import argparse
import os
import pandas as pd
import wandb
import warnings
warnings.filterwarnings("ignore")

def compute_metrics(tp, tn, fp, fn):
    """
    Compute precision, recall, f1, accuracy, and balanced accuracy (BACC) given tp, tn, fp, fn.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    bacc = (sensitivity + specificity) / 2
    return precision, recall, f1, accuracy, bacc

def fetch_wandb_results(project_name, api_key):
    """
    Fetch results from wandb for a given project.
    """
    # Authenticate with wandb
    wandb.login(key=api_key)

    # Initialize wandb API
    api = wandb.Api()

    # Fetch all runs in the project
    runs = api.runs(project_name)

    # Extract metrics for each run
    results = []
    for run in runs:
        # Get run configuration and metrics
        config = run.config
        summary = run.summary

        # Check if node_feat_type is custom
        node_feat_type = config.get("dataset", {}).get("node_feat_type", "pre_cal")
        if node_feat_type == "custom":
            # Use custom embedding names if available
            ab_embedding = config.get("dataset", {}).get("ab", {}).get("custom_embedding_method_src", {}).get("name", "custom")
            ag_embedding = config.get("dataset", {}).get("ag", {}).get("custom_embedding_method_src", {}).get("name", "custom")
        else:
            # Use default embedding models
            ab_embedding = config.get("dataset", {}).get("ab", {}).get("embedding_model", "unknown")
            ag_embedding = config.get("dataset", {}).get("ag", {}).get("embedding_model", "unknown")

        # Replace underscores with hyphens in embedding names
        ab_embedding = ab_embedding.replace("_", "-")
        ag_embedding = ag_embedding.replace("_", "-")

        # Compute node prediction metrics
        node_tp = summary.get("testEpochFinal/avg_epi_node_tp", 0)
        node_tn = summary.get("testEpochFinal/avg_epi_node_tn", 0)
        node_fp = summary.get("testEpochFinal/avg_epi_node_fp", 0)
        node_fn = summary.get("testEpochFinal/avg_epi_node_fn", 0)
        node_precision, node_recall, node_f1, node_accuracy, node_bacc = compute_metrics(node_tp, node_tn, node_fp, node_fn)

        # Compute link prediction metrics
        link_tp = summary.get("testEpochFinal/avg_edge_index_bg_tp", 0)
        link_tn = summary.get("testEpochFinal/avg_edge_index_bg_tn", 0)
        link_fp = summary.get("testEpochFinal/avg_edge_index_bg_fp", 0)
        link_fn = summary.get("testEpochFinal/avg_edge_index_bg_fn", 0)
        link_precision, link_recall, link_f1, link_accuracy, link_bacc = compute_metrics(link_tp, link_tn, link_fp, link_fn)

        # Map model type to algorithm name
        model_type = config.get("hparams", {}).get("model_type", "unknown")
        if model_type == "graph":
            algorithm = "GCN"
        elif model_type == "linear":
            algorithm = "GCN-L"
        elif model_type == "transformer":
            algorithm = "GAT"
        else:
            algorithm = model_type  # Fallback to original name if not mapped

        # Extract relevant metrics
        result = {
            "Algorithm": algorithm,
            "Model Name": run.name,
            "Embedding (Ab/Ag)": f"{ab_embedding}/{ag_embedding}",
            # Node Prediction Metrics
            "Node MCC": summary.get("testEpochFinal/avg_epi_node_mcc", None),
            "Node AUPRC": summary.get("testEpochFinal/avg_epi_node_auprc", None),
            "Node AUC-ROC": summary.get("testEpochFinal/avg_epi_node_auroc", None),  # Added AUC-ROC
            "Node Precision": node_precision,
            "Node Recall": node_recall,
            "Node F1": node_f1,
            "Node Accuracy": node_accuracy,
            "Node BACC": node_bacc,
            # Bipartite Link Prediction Metrics
            "Link MCC": summary.get("testEpochFinal/avg_edge_index_bg_mcc", None),
            "Link AUPRC": summary.get("testEpochFinal/avg_edge_index_bg_auprc", None),
            "Link AUC-ROC": summary.get("testEpochFinal/avg_edge_index_bg_auroc", None),  # Added AUC-ROC
            "Link Precision": link_precision,
            "Link Recall": link_recall,
            "Link F1": link_f1,
            "Link Accuracy": link_accuracy,
            "Link BACC": link_bacc,
            "Loss": summary.get("testEpochFinal/avg_loss", None),
            "Training Time (s)": summary.get("_runtime", None),
        }
        results.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(results)
    return df

def save_results_to_csv(df, output_path):
    """
    Save the results DataFrame to a CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def save_results_to_latex(df, output_dir, proj_name):
    """
    Save the results DataFrame to a single LaTeX file with three tables:
    - One combined table.
    - One table for Epitope Node Prediction.
    - One table for Bipartite Link Prediction.
    Each table has a proper caption and label.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sort the DataFrame by Algorithm to group rows by model type
    df = df.sort_values(by="Algorithm")

    # Define the LaTeX document structure
    latex_document = r"""
"""

    # Combined Table
    latex_document += r"""
\begin{table}[h!]
\centering
\caption{Combined Results for Epitope Node Prediction and Bipartite Link Prediction}
\label{tab:combined_results}
\begin{tabular}{llrrrrrrrrrrrrrrr}
\toprule
 & \multicolumn{7}{c}{Epitope Node Prediction} & \multicolumn{7}{c}{Bipartite Link Prediction} & \\
\cmidrule(lr){2-9} \cmidrule(lr){10-16}
Algorithm & Embedding (Ab/Ag) & MCC & AUPRC & AUC-ROC & Precision & Recall & F1 & BACC & MCC & AUPRC & AUC-ROC & Precision & Recall & F1 & BACC & Training Time (s) \\
\midrule
"""
    current_algorithm = None
    for _, row in df.iterrows():
        if row["Algorithm"] != current_algorithm:
            latex_document += r"""\cmidrule(lr){2-17}
            """
            current_algorithm = row["Algorithm"]
            latex_document += f"\\multirow{{{len(df[df['Algorithm'] == current_algorithm])}}}{{*}}{{{current_algorithm}}} & "
        else:
            latex_document += " & "
        latex_document += f"{row['Embedding (Ab/Ag)']} & "
        latex_document += f"{row['Node MCC']:.3f} & {row['Node AUPRC']:.3f} & {row['Node AUC-ROC']:.3f} & {row['Node Precision']:.3f} & {row['Node Recall']:.3f} & {row['Node F1']:.3f} & {row['Node BACC']:.3f} & "
        latex_document += f"{row['Link MCC']:.3f} & {row['Link AUPRC']:.3f} & {row['Link AUC-ROC']:.3f} & {row['Link Precision']:.3f} & {row['Link Recall']:.3f} & {row['Link F1']:.3f} & {row['Link BACC']:.3f} & "
        latex_document += f"{row['Training Time (s)']:.1f} \\\\\n"
    latex_document += r"""\bottomrule
\end{tabular}
\end{table}

"""

    # Epitope Node Prediction Table
    latex_document += r"""
\begin{table}[h!]
\centering
\caption{Results for Epitope Node Prediction}
\label{tab:node_results}
\begin{tabular}{llrrrrrrr}
\toprule
 & \multicolumn{7}{c}{Epitope Node Prediction} \\
\cmidrule(lr){2-9}
Algorithm & Embedding (Ab/Ag) & MCC & AUPRC & AUC-ROC & Precision & Recall & F1 & BACC \\
\midrule
"""
    current_algorithm = None
    for _, row in df.iterrows():
        if row["Algorithm"] != current_algorithm:
            latex_document += r"""\cmidrule(lr){2-9}
            """
            current_algorithm = row["Algorithm"]
            latex_document += f"\\multirow{{{len(df[df['Algorithm'] == current_algorithm])}}}{{*}}{{{current_algorithm}}} & "
        else:
            latex_document += " & "
        latex_document += f"{row['Embedding (Ab/Ag)']} & "
        latex_document += f"{row['Node MCC']:.3f} & {row['Node AUPRC']:.3f} & {row['Node AUC-ROC']:.3f} & {row['Node Precision']:.3f} & {row['Node Recall']:.3f} & {row['Node F1']:.3f} & {row['Node BACC']:.3f} \\\\\n"
    latex_document += r"""\bottomrule
\end{tabular}
\end{table}

"""

    # Bipartite Link Prediction Table
    latex_document += r"""
\begin{table}[h!]
\centering
\caption{Results for Bipartite Link Prediction}
\label{tab:link_results}
\begin{tabular}{llrrrrrrr}
\toprule
 & \multicolumn{7}{c}{Bipartite Link Prediction} \\
\cmidrule(lr){2-9}
Algorithm & Embedding (Ab/Ag) & MCC & AUPRC & AUC-ROC & Precision & Recall & F1 & BACC \\
\midrule
"""
    current_algorithm = None
    for _, row in df.iterrows():
        if row["Algorithm"] != current_algorithm:
            latex_document += r"""\cmidrule(lr){2-9}
            """
            current_algorithm = row["Algorithm"]
            latex_document += f"\\multirow{{{len(df[df['Algorithm'] == current_algorithm])}}}{{*}}{{{current_algorithm}}} & "
        else:
            latex_document += " & "
        latex_document += f"{row['Embedding (Ab/Ag)']} & "
        latex_document += f"{row['Link MCC']:.3f} & {row['Link AUPRC']:.3f} & {row['Link AUC-ROC']:.3f} & {row['Link Precision']:.3f} & {row['Link Recall']:.3f} & {row['Link F1']:.3f} & {row['Link BACC']:.3f} \\\\\n"
    latex_document += r"""\bottomrule
\end{tabular}
\end{table}

"""

    # Save the LaTeX document
    tex_path = os.path.join(output_dir, f"{proj_name}_wandb_results.tex")
    with open(tex_path, "w") as f:
        f.write(latex_document)
    print(f"LaTeX document saved to {tex_path}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compile wandb results into a summary CSV and LaTeX table.")
    parser.add_argument(
        "--project_name",
        type=str,
        required=True,
        help="Name of the wandb project."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="Your wandb API key."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the results."
    )
    args = parser.parse_args()

    # Fetch results from wandb
    df = fetch_wandb_results(args.project_name, args.api_key)
    
    proj_name = args.project_name.split("/")[1]
    # Save results to CSV
    csv_path = os.path.join(args.output_dir, f"{proj_name}_wandb_results_summary.csv")
    save_results_to_csv(df, csv_path)

    # Save results to LaTeX
    save_results_to_latex(df, args.output_dir, proj_name)

if __name__ == "__main__":
    main()


########## test ###########
"""
python3 inference/compile_wandb_results.py \         
    --project_name "alibilab-gsu/retrain-walle" \
    --api_key "" \
    --output_dir ""
"""
