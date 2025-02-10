Contents

- [Evaluate pre-trained model](#evaluate-pre-trained-model)
  - [Download ckpts from wandb](#download-ckpts-from-wandb)
  - [Evaluate ckpts](#evaluate-ckpts)
  - [Calculate mean metrics from evaluation results](#calculate-mean-metrics-from-evaluation-results)
  - [Evaluation procedures](#evaluation-procedures)
    - [Validate config](#validate-config)
  - [Evaluation Results Schema](#evaluation-results-schema)
    - [Config modification for old previous versions of config](#config-modification-for-old-previous-versions-of-config)
  - [Pre-trained model and config](#pre-trained-model-and-config)
    - [Split by epitope/surf ratio](#split-by-epitopesurf-ratio)
    - [Split by epitope group](#split-by-epitope-group)


## Embedding methods:
- [ESM2 and ESM-IF1](https://github.com/facebookresearch/esm.git)
- [ProtBert](https://github.com/agemagician/ProtTrans.git)
- BLOSUM62
- One-hot encoding
- [AntiBERTy](https://github.com/jeffreyruffolo/AntiBERTy.git)


## Baseline epitope prediction methods:
- [EpiGraph](https://github.com/sj584/EpiGraph.git)

## Dataset:
- [AsEP](https://github.com/biochunan/AsEP-dataset.git)


## Evaluation procedures



## Evaluation Results Schema



### Config modification for old previous versions of config

- may need to add a field `model_type:"graph"` to hparams field
- may need to add a field `split_method:null` to dataset field

## Pre-trained model and config

### Split by epitope/surf ratio

|    Run Name    | Model  | Embedding (Ab/Ag) | \# GCN Layers |
| :------------: | :----: | :---------------: | :-----------: |
| clear-sweep-30 | graph  |      onehot       |       2       |
| comic-sweep-9  | graph  |     ESM2/ESM2     |       2       |
| decent-sweep-2 | graph  |     BLOSUM62      |       2       |
| lilac-sweep-16 | linear |     ESM2/ESM2     |       2       |
|  sage-sweep-9  | linear |    IgFold/ESM2    |       2       |
| whole-sweep-72 | graph  |    IgFold/ESM2    |       2       |

Best performing model on `valEpoch/avg_edge_index_bg_mcc`:

|     Run Name      | Model | Embedding (Ab/Ag) | \# GCN Layers |
| :---------------: | :---: | :---------------: | :-----------: |
| treasured-sweep-8 | graph |    IgFold/ESM2    |       2       |


### Split by epitope group

|     Run Name     | Model | Embedding (Ab/Ag) | \# GCN layers |
| :--------------: | :---: | :---------------: | :-----------: |
| jumping-sweep-22 | graph |    IgFold/ESM2    |       2       |
|   wild-sweep-1   | graph |    IgFold/ESM2    |       3       |
