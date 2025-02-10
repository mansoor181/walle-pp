import os
from Bio.PDB import PDBParser
import esm
import esm.inverse_folding
import torch
import warnings
warnings.filterwarnings("ignore")

def embed_esm_if(sequence, pdb_id = None, target_chain_id: str = None) -> torch.Tensor:
    """
    Generate ESM-IF embeddings for a given PDB file.

    Args:
        pdb_file_path (str): Path to the PDB file.
        target_chain_id (str, optional): Specific chain ID to extract embeddings for. 
                                         If None, embeddings for all chains are returned.

    Returns:
        torch.Tensor: A tensor of shape (L, 512) where L is the length of the sequence.
    """
    if pdb_id:
        # change the structures directory path accordingly
        structures_dir = "/Users/mansoor/Documents/GSU/Projects/Antibody-Design/epitope-prediction/data/asep/structures/"
        pdb_file_path = f"{structures_dir}{pdb_id}.pdb" 
        # Load the ESM-IF model
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model = model.eval()

        # Parse the PDB file using BioPython
        parser = PDBParser()
        pdb_structure = parser.get_structure("pdb_structure", pdb_file_path)
        pdb_model = pdb_structure[0]

        # If a specific chain ID is provided, extract embeddings for that chain
        if target_chain_id:
            chain_ids = [target_chain_id]
        else:
            # Extract embeddings for all chains
            chain_ids = [chain.get_id() for chain in pdb_model]
            chain_ids = chain_ids[2] # get the antigen chain id

        # List to store embeddings for all chains
        chain_embeddings = []

        for chain_id in chain_ids:
            # Load the structure and extract coordinates
            structure = esm.inverse_folding.util.load_structure(pdb_file_path, chain_id)
            coords, _ = esm.inverse_folding.util.extract_coords_from_structure(structure)

            # Get ESM-IF embeddings for the chain
            esm_if_rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)
            chain_embeddings.append(torch.tensor(esm_if_rep).float())

        # Concatenate embeddings for all chains
        if chain_embeddings:
            embeddings = torch.cat(chain_embeddings, dim=0)
        else:
            raise ValueError("No embeddings were generated. Check the PDB file and chain IDs.")

    else:
        embeddings = torch.zeros(len(sequence),512)

    return embeddings


# Example usage
if __name__ == "__main__":
    # Path to the PDB file
    proj_dir = "/Users/mansoor/Documents/GSU/Projects/Antibody-Design/epitope-prediction/"
    asep_data_dir = os.path.join(proj_dir, "data/asep/")
    structures_asep_path = asep_data_dir + "structures/"

    pdb_file_path = structures_asep_path + "3v6o_1P.pdb"  # Replace with PDB file path
    pdb_id = "3v6o_1P"
    # Generate embeddings for a specific chain or all chains
    target_chain_id = "a"  # Set to None to process all chains
    embeddings = embed_esm_if(pdb_id, target_chain_id)
    print(f"Embeddings shape: {embeddings.shape}")









# import os
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd
# import torch
# from Bio.PDB import PDBParser, Polypeptide
# import esm
# import esm.inverse_folding
# from typing import Any, Callable, Dict, List, Optional, Tuple, Union
# from biotite.structure.io import load_structure
# from Bio.PDB import Polypeptide

# from asep.utils.utils import run_seqres2atmseq, extract_seqres_from_pdb

# PDBDataFrame = pd.DataFrame
# AllAtomDataFrame = pd.DataFrame
# AdjMatrix = np.ndarray
# BinaryMask = np.ndarray



# # map seqres -> atmseq -> surf_mask
# def map_seqres_to_atmseq_with_downstream_mask(
#     seqres2atmseq_mask: Dict[str, Any], target_mask: np.ndarray
# ) -> AdjMatrix:
#     """
#     Map SEQRES to ATMSEQ then to Surface residue mask.

#     Args:
#         seqres2atmseq_mask (Dict[str, Any]): a dictionary with keys: 'seqres', 'atmseq', 'mask'. This is mapping from SEQRES to ATMSEQ.
#             All three str values have the same length L.
#         target_mask (np.ndarray): e.g. surface residue mask, this is mapping from ATMSEQ to a downstream residue mask.
#             The length of this array must equal to the length of 'atmseq' (exclude '-') in `seqres2atmseq_mask`.

#     Returns:
#         np.ndarray: a binary mask mapping SEQRES to Surface residues, shape (L, )
#     """
#     assert len(target_mask) == len(seqres2atmseq_mask["seq"]["atmseq"].replace("-", ""))
#     seqres2target_mask, i = [], 0
#     for c in seqres2atmseq_mask["seq"]["mask"]:
#         if (
#             c == "1" or c == 1 or c is True
#         ):  # residue exists in the structure with a surface mask
#             seqres2target_mask.append(target_mask[i])
#             i += 1
#         elif c == "0" or c == 0 or c is False:  # missing residue
#             seqres2target_mask.append(0)
#     return np.array(seqres2target_mask)

# def esm_if_residue_embedding(
#     pdb_file_path: str,
#     target_chain_id: Optional[str] = None,
#     seqres: Optional[Dict[str, str]] = None,
# ) -> torch.Tensor:
#     """
#     Generate ESM-IF embeddings for a given PDB file, considering seqres2surf mapping.

#     Args:
#         pdb_file_path (str): Path to the PDB file.
#         target_chain_id (str, optional): Specific chain ID to extract embeddings for. 
#                                          If None, embeddings for all chains are returned.
#         seqres (Dict[str, str], optional): SEQRES sequences for the chains. If not provided,
#                                            they will be extracted from the PDB file.

#     Returns:
#         torch.Tensor: A tensor of shape (L, 512) where L is the length of the sequence.
#     """
#     # Load the ESM-IF model
#     model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
#     model = model.eval()

#     # Load the PDB file using biotite
#     structure = load_structure(pdb_file_path)

#     # If a specific chain ID is provided, extract embeddings for that chain
#     if target_chain_id:
#         chain_ids = [target_chain_id]
#     else:
#         # Extract embeddings for all chains
#         chain_ids = list(set(structure.chain_id))

#     # If SEQRES is not provided, extract it from the PDB file
#     if seqres is None:
#         seqres = extract_seqres_from_pdb(Path(pdb_file_path))

#     # List to store embeddings for all chains
#     chain_embeddings = []

#     for chain_id in chain_ids:
#         # Filter the structure for the target chain
#         chain_structure = structure[structure.chain_id == chain_id]

#         # Extract coordinates and residue names
#         coords = chain_structure.coord
#         residue_names = chain_structure.res_name  # Access residue names directly

#         # Convert residue names to one-letter codes
#         atmseq = "".join([Polypeptide.three_to_one(name) for name in residue_names])

#         # Get ESM-IF embeddings for the chain
#         esm_if_rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)
#         esm_if_rep = torch.tensor(esm_if_rep, dtype=torch.float32)  # Ensure correct tensor type

#         # Map SEQRES to ATMSEQ and then to surface residues
#         seqres2atmseq_mask = run_seqres2atmseq(seqres=list(seqres.values())[0], atmseq=atmseq)
#         surf_mask = np.ones(len(atmseq))  # Assuming all residues are surface residues for simplicity
#         seqres2surf_mask = map_seqres_to_atmseq_with_downstream_mask(
#             seqres2atmseq_mask=seqres2atmseq_mask, target_mask=surf_mask
#         )

#         # Apply the mask to the embeddings
#         esm_if_rep = esm_if_rep[seqres2surf_mask == 1, :]
#         chain_embeddings.append(esm_if_rep)

#     # Concatenate embeddings for all chains
#     if chain_embeddings:
#         embeddings = torch.cat(chain_embeddings, dim=0)
#     else:
#         raise ValueError("No embeddings were generated. Check the PDB file and chain IDs.")

#     return embeddings

# # Example usage
# if __name__ == "__main__":
#     # Path to the PDB file
#     # Path to the PDB file
#     proj_dir = "/Users/mansoor/Documents/GSU/Projects/Antibody-Design/epitope-prediction/"
#     asep_data_dir = os.path.join(proj_dir, "data/asep/")
#     structures_asep_path = asep_data_dir + "structures/"

#     pdb_file_path = structures_asep_path + "3v6o_1P.pdb"  # Replace with PDB file path

#     # Generate embeddings for a specific chain or all chains
#     target_chain_id = "a"  # Set to None to process all chains
#     seqres = None  # Provide SEQRES if available, otherwise it will be extracted from the PDB file
#     embeddings = esm_if_residue_embedding(pdb_file_path, target_chain_id, seqres)

#     print(f"Embeddings shape: {embeddings.shape}")
