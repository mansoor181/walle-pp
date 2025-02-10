# EpiGraph
B cell epitope prediction by capturing spatial clustering property of the epitopes using graph attention network  <br/> 
<br/> (Sungjin Choi, Dongsup Kim)

# Webserver
User friendly webserver is available at https://epigraph.kaist.ac.kr/

# Installation (In Ubuntu)
1. Generate the conda environment
```python
conda env create --file conda_env.yaml --name Bepitope
conda activate Bepitope 
```
 
2. Install related packages  
```python
pip install wget
pip install biopython
pip install biotite
pip install fair-esm  
sudo apt-get install dssp  # for generating RSA using Biopython
```

3. Install models
```python
# at first, it may take some time to download the esm-2, esm-if model  
import torch
_, _ = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D") # load esm-2 model

import esm
_, _ = esm.pretrained.esm_if1_gvp4_t16_142M_UR50() # load esm-if model
```


# BUG FIX from esm

ImportError: cannot import name 'esmfold_structure_module_only_8M' from 'esm.pretrained' (/home/{user}/anaconda3/envs/Bepitope/lib/python3.8/site-packages/esm/pretrained.py) 

simply copy-paste the functions starting from esmfold_structure_module_only_8M into the pretrained.py
(https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/esm/pretrained.py#L274)

or

cp pretrained.py /home/{user}/anaconda3/envs/Bepitope/lib/python3.8/site-packages/esm/pretrained.py

Then everything will be ok.

# Download dataset
1. Preprocessed training/test set from epitope3D dataset is available at https://drive.google.com/drive/folders/1pqPEUxVgo4vCn5Q_a4AR_G4lq-vq24aI?usp=sharing
2. Original file from epitop3D https://biosig.lab.uq.edu.au/epitope3d/data
3. Original file from DiscoTope-3 https://services.healthtech.dtu.dk/services/DiscoTope-3.0/
 

# Prediction


```python
# predict the epitopes from pdb (fetched)
python inference.py --pdb 1cfi
```



```python
# for multiple inference... in bash
for pdb in pdb1 pdb2 pdb3 pdb4 pdb5 pdb6 ... pdb10
> do
> python inference.py --pdb $pdb
> done
```


```python
# for inference using pdb file from local computer...
# the pdb file must be located in Custom_PDB(default) directory 
# or any directory you assign with --pdb_path
python inference_CustomPDB.py --pdb 6FNZ
```

# Replication

```python
# simply evaluate the trained model(in checkpoint directory) on the epitope3d test set (45 PDB)
python evaluate.py
```

```python
# to train the models and save
# in model training, we recommend using GPU. CPU work is quite slow.
python train.py --model_save_dir models
```


```python
# assign new directory with --model_checkpoint
# python evaluate.py simply evaluate the models saved in checkpoint
python evaluate.py --model_checkpoint models
```

Each python program contains more arguments.
You can check with -h option

ex) python inference.py/inference_CustomPDB.py/train.py/evaluate.py -h
