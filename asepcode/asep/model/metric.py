from typing import Dict

import torch
from torch import Tensor
from torcheval.metrics import BinaryAUPRC, BinaryAUROC, BinaryConfusionMatrix
from torcheval.metrics.functional import binary_auprc, binary_auroc
from torchmetrics.functional import matthews_corrcoef


def cal_edge_index_bg_auprc(
    edge_index_bg_pred: Tensor,
    edge_index_bg_true: Tensor,
    edge_cutoff: float = 0.5,
) -> Tensor:
    """
    Compute AUPRC for bipartite link prediction.
    """
    with torch.no_grad():
        t = edge_index_bg_true.reshape(-1).long().cpu()
        p = (edge_index_bg_pred > edge_cutoff).reshape(-1).long().cpu()
        return binary_auprc(p, t)


def cal_edge_index_bg_auroc(
    edge_index_bg_pred: Tensor,
    edge_index_bg_true: Tensor,
) -> Tensor:
    """
    Compute AUC-ROC for bipartite link prediction.
    """
    with torch.no_grad():
        t = edge_index_bg_true.reshape(-1).long().cpu()
        p = edge_index_bg_pred.reshape(-1).cpu()
        return binary_auroc(p, t)


def cal_epitope_node_auprc(
    edge_index_bg_pred: Tensor,
    edge_index_bg_true: Tensor,
    num_edge_cutoff: int,  # used to determine epitope residue from edges,
) -> Tensor:
    """
    Compute AUPRC for epitope node prediction.
    """
    with torch.no_grad():
        # Get epitope idx
        t = (edge_index_bg_true.sum(dim=0) > 0).reshape(-1).long()
        p = (edge_index_bg_pred.sum(dim=0) > num_edge_cutoff).reshape(-1).long()
        return binary_auprc(p, t)


def cal_epitope_node_auroc(
    edge_index_bg_pred: Tensor,
    edge_index_bg_true: Tensor,
) -> Tensor:
    """
    Compute AUC-ROC for epitope node prediction.
    """
    with torch.no_grad():
        # Get epitope idx
        t = (edge_index_bg_true.sum(dim=0) > 0).reshape(-1).long().cpu()
        p = edge_index_bg_pred.sum(dim=0).reshape(-1).cpu()
        return binary_auroc(p, t)


def cal_edge_index_bg_metrics(
    edge_index_bg_pred: Tensor,
    edge_index_bg_true: Tensor,
    edge_cutoff: float = 0.5,
) -> Dict:
    """
    Compute metrics for bipartite link prediction:
    - AUPRC, AUC-ROC, MCC, TP, TN, FP, FN.
    """
    with torch.no_grad():
        t = edge_index_bg_true.reshape(-1).long().cpu()
        p = edge_index_bg_pred.reshape(-1).cpu()

        # AUPRC
        auprc = BinaryAUPRC().update(input=p, target=t).compute()

        # AUC-ROC
        auroc = BinaryAUROC().update(input=p, target=t).compute()

        # Confusion matrix
        tn, fp, fn, tp = (
            BinaryConfusionMatrix(threshold=edge_cutoff)
            .update(input=p, target=t)
            .compute()
            .reshape(-1)
        )

        # MCC
        mcc = (tp * tn - fp * fn) / (
            torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-7
        )

        return {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "auprc": auprc,
            "auroc": auroc,
            "mcc": mcc,
        }


def cal_epitope_node_metrics(
    edge_index_bg_pred: Tensor,
    edge_index_bg_true: Tensor,
    num_edge_cutoff: int,  # used to determine epitope residue from edges,
) -> Dict:
    """
    Compute metrics for epitope node prediction:
    - AUPRC, AUC-ROC, MCC, TP, TN, FP, FN.
    """
    with torch.no_grad():
        # Get epitope idx
        t = (edge_index_bg_true.sum(dim=0) > 0).reshape(-1).long().cpu()
        p = (edge_index_bg_pred.sum(dim=0) > num_edge_cutoff).reshape(-1).long().cpu()

        # AUPRC
        auprc = BinaryAUPRC().update(input=p, target=t).compute()

        # AUC-ROC
        auroc = BinaryAUROC().update(input=p, target=t).compute()

        # Confusion matrix
        tn, fp, fn, tp = (
            BinaryConfusionMatrix().update(input=p, target=t).compute().reshape(-1)
        )

        # MCC
        mcc = (tp * tn - fp * fn) / torch.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-7
        )

        return {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "auprc": auprc,
            "auroc": auroc,
            "mcc": mcc,
        }