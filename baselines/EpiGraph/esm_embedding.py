from Bio.PDB import PDBParser
import esm.inverse_folding
import networkx as nx
import numpy as np
import torch
import os

# load esm2 model
def _load_esm_model(model_name: str = "esm2_t33_650M_UR50D"): 
    return torch.hub.load("facebookresearch/esm", model_name)


def esm2_residue_embedding(
    sequence : str,
    model_name: str = "esm2_t33_650M_UR50D", 
    output_layer: int = 33,
) -> torch.Tensor:                               
    
    
    embedding_total = np.empty((0, 1280))
    
    # sequence length > 1022, truncate the sequence region after position 1022
    # only the sequence position upto 1022 are embedded using esm-2 model
  
    embedding = compute_esm_embedding(
    sequence,
    representation="residue",
    model_name=model_name,
    output_layer=output_layer,
    )

    # remove start and end tokens from per-token residue embeddings
    embedding = embedding[0, 1:-1]
       
    return embedding_total


def compute_esm_embedding(
    sequence: str,
    representation: str,
    model_name: str = "esm2_t33_650M_UR50D",
    output_layer: int = 33,
) -> torch.Tensor:

    
    model, alphabet = _load_esm_model(model_name)
    batch_converter = alphabet.get_batch_converter()

    data = [
        ("protein1", sequence),
    ]
    

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(
            batch_tokens, repr_layers=[output_layer], return_contacts=True
        )
    token_representations = results["representations"][output_layer]

        
    if representation == "residue":
        return token_representations.numpy()
        
    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    elif representation == "sequence":
        sequence_representations = []
        for i, (_, seq) in enumerate(data):
            sequence_representations.append(
                token_representations[i, 1 : len(seq) + 1].mean(0)
            )

            return sequence_representations[0]

def esm_if_2_embedding(pdb_id, path):
    
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()    
    
    
    # parse the PDB file usign biopython
    parser = PDBParser()
    
    pdb_file_path = os.path.join(f"{path}", f"{pdb_id}.pdb")
    pdb_structure = parser.get_structure(f"{pdb_id}", pdb_file_path)
    pdb_model = pdb_structure[0]
    
    # get the chain_list
    chain_list = []
    for chain in pdb_model:
        chain_id = chain.get_id()
        chain_list.append(chain_id)
    fpath = os.path.join(f"{path}/" +  f"{pdb_id}.pdb")
    
    
    chain_node_list = []
    chain_esm_if_list = []
    chain_esm2_list = []
    chain_coord_list = []
    
    # get node_id, esm_if, esm2, coord information per-chain
    for target_chain_id in chain_list:
        structure = esm.inverse_folding.util.load_structure(fpath, target_chain_id)
        coords, seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
        
        esm_if_rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)
        chain_esm_if_list.append(torch.tensor(esm_if_rep).float())
        
        esm2_rep = esm2_residue_embedding(seq)
        chain_esm2_list.append(torch.tensor(esm2_rep).float())
        
        node_list = []
        coord_list = []
        for idx, i in enumerate(structure):
            
            # collect the residue and coordinates (N, Ca, C) 
            if idx % 3 == 0:
                
                chain = i._annot["chain_id"]
                res_id = i._annot["res_id"]
                res_name = i._annot["res_name"]
                residue = chain + ":" + res_name + ":" + str(res_id)
                node_list.append(residue)

                # backbone coordinates: from Ca, (order of N, Ca, C per residue) 
                Ca_coord =  coords[idx // 3][1]
                coord_list.append(Ca_coord)
                
        chain_node_list.append(node_list)
        chain_coord_list.append(coord_list)

    chain_esm_if_list = torch.concat(chain_esm_if_list)
    chain_esm2_list = torch.concat(chain_esm2_list)
        
        
    return chain_esm_if_list, chain_esm2_list, chain_node_list, chain_coord_list

