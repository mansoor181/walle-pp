
This repository contains an extended version of [WALLE](https://github.com/biochunan/AsEP-dataset.git) for improved antibody-aware epitope prediction using graph convolutional network (GCN) coupled with protein language model (PLM)-based embedding methods such as ESM2, ProtBERT, and AntiBERTy.


## Embedding methods:
Specifically, we employed the following residue and protein folding embedding methods:
- [ESM2 and ESM-IF1](https://github.com/facebookresearch/esm.git)
- [ProtBert](https://github.com/agemagician/ProtTrans.git)
- BLOSUM62
- One-hot encoding
- [AntiBERTy](https://github.com/jeffreyruffolo/AntiBERTy.git)


## Baseline epitope prediction methods:
- [EpiGraph](https://github.com/sj584/EpiGraph.git)

## Dataset:
- [AsEP](https://github.com/biochunan/AsEP-dataset.git)


## Evaluation Results Schema

# Performance Comparison on Different Dataset Splits

| Algorithm     | Epitope Ratio Split                                                 | Epitope Group Split                                                |
|---------------|----------------------------------------------------------------------|----------------------------------------------------------------------|
|               | MCC     | Prec.   | Recall  | AUCROC  | F1      | MCC     | Prec.   | Recall  | AUCROC  | F1      |
|---------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| **WALLE++**   | **0.263** | **0.281** | **0.457** | **0.650** | **0.348** | **0.123** | 0.162   | **0.280** | **0.569** | **0.205** |
| WALLE         | 0.210   | 0.235   | 0.258   | 0.635   | 0.145   | 0.077   | 0.143   | 0.266   | 0.544   | 0.145   |
| EpiPred       | 0.029   | 0.122   | 0.142   | —       | 0.112   | -0.006  | 0.089   | 0.158   | —       | 0.112   |
| ESMFold       | 0.028   | 0.137   | 0.060   | —       | 0.046   | 0.018   | 0.113   | 0.034   | —       | 0.046   |
| ESMBind       | 0.016   | 0.106   | 0.090   | 0.506   | 0.064   | 0.002   | 0.082   | 0.076   | 0.500   | 0.064   |
| MaSIF-site    | 0.037   | 0.125   | 0.114   | —       | 0.128   | 0.046   | **0.164** | 0.174   | —       | 0.128   |

**Note:** The best values in each column are highlighted in bold.
