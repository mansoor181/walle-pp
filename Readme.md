## Description


This repository contains an extended version of [WALLE](https://github.com/biochunan/AsEP-dataset.git) for improved antibody-aware epitope prediction using graph convolutional network (GCN) coupled with protein language model (PLM)-based embedding methods. Specifically, we employed the following residue and protein folding embedding methods: [ESM2 and ESM-IF1](https://github.com/facebookresearch/esm.git), [ProtBert](https://github.com/agemagician/ProtTrans.git), [AntiBERTy](https://github.com/jeffreyruffolo/AntiBERTy.git), BLOSUM62, and One-hot encoding.



## Baseline epitope prediction methods:
- [EpiGraph](https://github.com/sj584/EpiGraph.git)
<!-- 
## Dataset:
- [AsEP](https://github.com/biochunan/AsEP-dataset.git)
 -->


## Performance Comparison on Different Dataset Splits


|    Algorithm  | MCC       | Prec.     | Recall    | AUCROC    | F1        |
|---------------|-----------|-----------|-----------|-----------|-----------|
| **Our approach**   | **0.263** | **0.281** | **0.457** | **0.650** | **0.348** |
| WALLE         | 0.210     | 0.235     | 0.258     | 0.635     | 0.145     |
| EpiPred       | 0.029     | 0.122     | 0.142     | —         | 0.112     |
| ESMBind       | 0.016     | 0.106     | 0.090     | 0.506     | 0.064     |
| MaSIF-site    | 0.037     | 0.125     | 0.114     | —         | 0.128     |

**Note:** The best values in each column are highlighted in bold.
